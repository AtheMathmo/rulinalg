use matrix::{Matrix, BaseMatrix, BaseMatrixMut};
use matrix::{back_substitution};
use matrix::PermutationMatrix;
use vector::Vector;
use error::{Error, ErrorKind};

use std::any::Any;

use libnum::{Float, Zero, One};

use matrix::decomposition::Decomposition;

/// Result of unpacking an instance of
/// [PartialPivLu](struct.PartialPivLu.html).
#[derive(Debug, Clone)]
pub struct LUP<T> {
    /// The lower triangular matrix in the decomposition.
    pub l: Matrix<T>,
    /// The upper triangular matrix in the decomposition.
    pub u: Matrix<T>,
    /// The permutation matrix in the decomposition.
    pub p: PermutationMatrix<T>
}

/// LU decomposition with partial pivoting.
///
/// For any square matrix A, there exist a permutation matrix
/// `P`, a lower triangular matrix `L` and an upper triangular
/// matrix `U` such that
///
/// ```text
/// PA = LU.
/// ```
///
/// However, due to the way partial pivoting algorithms work,
/// LU decomposition with partial pivoting is in general
/// *only numerically stable for well-conditioned invertible matrices*.
///
/// That said, partial pivoting is sufficient in the vast majority
/// of practical applications, and it is also the fastest of the
/// pivoting schemes in existence.
///
///
/// # Applications
///
/// Given a matrix `x`, computing the LU(P) decomposition is simple:
///
/// ```
/// use rulinalg::matrix::decomposition::{PartialPivLu, LUP, Decomposition};
/// use rulinalg::matrix::Matrix;
///
/// let x = Matrix::<f64>::identity(4);
///
/// // The matrix is consumed and its memory
/// // re-purposed for the decomposition
/// let lu = PartialPivLu::decompose(x).expect("Matrix is invertible.");
///
/// // See below for applications
/// // ...
///
/// // The factors L, U and P can be obtained by unpacking the
/// // decomposition, for example by destructuring as seen here
/// let LUP { l, u, p } = lu.unpack();
///
/// ```
///
/// ## Solving linear systems
///
/// Arguably the most common use case of LU decomposition
/// is the computation of solutions to (multiple) linear systems
/// that share the same coefficient matrix.
///
/// ```
/// # #[macro_use] extern crate rulinalg;
/// # use rulinalg::matrix::decomposition::PartialPivLu;
/// # use rulinalg::matrix::Matrix;
/// # fn main() {
/// # let x = Matrix::identity(4);
/// # let lu = PartialPivLu::decompose(x).unwrap();
/// let b = vector![3.0, 4.0, 2.0, 1.0];
/// let y = lu.solve(b)
///           .expect("Matrix is invertible.");
/// assert_vector_eq!(y, vector![3.0, 4.0, 2.0, 1.0], comp = float);
///
/// // We can efficiently solve multiple such systems
/// let c = vector![0.0, 0.0, 0.0, 0.0];
/// let z = lu.solve(c).unwrap();
/// assert_vector_eq!(z, vector![0.0, 0.0, 0.0, 0.0], comp = float);
/// # }
/// ```
///
/// ## Computing the inverse of a matrix
///
/// The LU decomposition provides a convenient way to obtain
/// the inverse of the decomposed matrix. However, please keep
/// in mind that explicitly computing the inverse of a matrix
/// is *usually* a bad idea. In many cases, one might instead simply
/// solve multiple systems using `solve`.
///
/// For example, a common misconception is that when one needs
/// to solve multiple linear systems `Ax = b` for different `b`,
/// one should pre-compute the inverse of the matrix for efficiency.
/// In fact, this is practically never a good idea! A far more efficient
/// and accurate method is to perform the LU decomposition once, and
/// then solve each system as shown in the examples of the previous
/// subsection.
///
/// That said, there are definitely cases where an explicit inverse is
/// needed. In these cases, the inverse can easily be obtained
/// through the `inverse()` method.
///
/// # Computing the determinant of a matrix
///
/// Once the LU decomposition has been obtained, computing
/// the determinant of the decomposed matrix is a very cheap
/// operation.
///
/// ```
/// # #[macro_use] extern crate rulinalg;
/// # use rulinalg::matrix::decomposition::PartialPivLu;
/// # use rulinalg::matrix::Matrix;
/// # fn main() {
/// # let x = Matrix::<f64>::identity(4);
/// # let lu = PartialPivLu::decompose(x).unwrap();
/// assert_eq!(lu.det(), 1.0);
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct PartialPivLu<T> {
    lu: Matrix<T>,
    p: PermutationMatrix<T>
}

impl<T: Clone + One + Zero> Decomposition for PartialPivLu<T> {
    type Factors = LUP<T>;

    fn unpack(self) -> LUP<T> {
        let l = unit_lower_triangular_part(&self.lu);
        let u = nullify_lower_triangular_part(self.lu);

        LUP {
            l: l,
            u: u,
            p: self.p
        }
    }
}

impl<T: 'static + Float> PartialPivLu<T> {
    /// Performs the decomposition.
    ///
    /// # Panics
    ///
    /// The matrix must be square.
    ///
    /// # Errors
    ///
    /// An error will be returned if the matrix
    /// is singular to working precision (badly conditioned).
    pub fn decompose(matrix: Matrix<T>) -> Result<Self, Error> {
        let n = matrix.cols;
        assert!(matrix.rows == n, "Matrix must be square for LU decomposition.");
        let mut lu = matrix;
        let mut p = PermutationMatrix::identity(n);

        for index in 0..n {
            let mut curr_max_idx = index;
            let mut curr_max = lu[[curr_max_idx, curr_max_idx]];

            for i in (curr_max_idx+1)..n {
                if lu[[i, index]].abs() > curr_max.abs() {
                    curr_max = lu[[i, index]];
                    curr_max_idx = i;
                }
            }
            if curr_max.abs() < T::epsilon() {
                return Err(Error::new(ErrorKind::DivByZero,
                    "The matrix is too ill-conditioned for
                     LU decomposition with partial pivoting."));
            }

            lu.swap_rows(index, curr_max_idx);
            p.swap_rows(index, curr_max_idx);
            for i in (index+1)..n {
                let mult = lu[[i, index]] / curr_max;
                lu[[i, index]] = mult;
                for j in (index+1)..n {
                    lu[[i, j]] = lu[[i,j]] - mult*lu[[index, j]];
                }
            }
        }
        Ok(PartialPivLu {
            lu: lu,
            p: p.inverse()
        })
    }
}

// TODO: Remove Any bound (cannot for the time being, since
// back substitution uses Any bound)
impl<T> PartialPivLu<T> where T: Any + Float {
    /// Solves the linear system `Ax = b`.
    ///
    /// Here, `A` is the decomposed matrix satisfying
    /// `PA = LU`. Note that this method is particularly
    /// well suited to solving multiple such linear systems
    /// involving the same `A` but different `b`.
    ///
    /// # Errors
    ///
    /// If the matrix is very ill-conditioned, the function
    /// might fail to obtain the solution to the system.
    ///
    /// # Panics
    ///
    /// The right-hand side vector `b` must have compatible size.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg;
    /// # use rulinalg::matrix::decomposition::PartialPivLu;
    /// # use rulinalg::matrix::Matrix;
    /// # fn main() {
    /// let x = Matrix::identity(4);
    /// let lu = PartialPivLu::decompose(x).unwrap();
    /// let b = vector![3.0, 4.0, 2.0, 1.0];
    /// let y = lu.solve(b)
    ///           .expect("Matrix is invertible.");
    /// assert_vector_eq!(y, vector![3.0, 4.0, 2.0, 1.0], comp = float);
    /// # }
    /// ```
    pub fn solve(&self, b: Vector<T>) -> Result<Vector<T>, Error> {
        assert!(b.size() == self.lu.rows(),
            "Right-hand side vector must have compatible size.");
        // Note that applying p here implicitly incurs a clone.
        // TODO: Is it possible to avoid the clone somehow?
        // To my knowledge, applying a permutation matrix
        // in-place in O(n) time requires O(n) storage for bookkeeping.
        // However, we might be able to get by with something like
        // O(n log n) for the permutation as the forward/backward
        // substitution algorithms are O(n^2), if this helps us
        // avoid the memory overhead.
        let b = lu_forward_substitution(&self.lu, &self.p * b);
        back_substitution(&self.lu, b)
    }

    /// Computes the inverse of the matrix which this LUP decomposition
    /// represents.
    ///
    /// # Errors
    /// The inversion might fail if the matrix is very ill-conditioned.
    pub fn inverse(&self) -> Result<Matrix<T>, Error> {
        let n = self.lu.rows();
        let mut inv = Matrix::zeros(n, n);
        let mut e = Vector::zeros(n);

        // To compute the inverse of a matrix A, note that
        // we can simply solve the system
        // AX = I,
        // where X is the inverse of A, and I is the identity
        // matrix of appropriate dimension.
        //
        // Note that this is not optimal in terms of performance,
        // and there is likely significant potential for improvement.
        //
        // A more performant technique is usually to compute the
        // triangular inverse of each of the L and U triangular matrices,
        // but this again requires efficient algorithms (blocked/recursive)
        // to invert triangular matrices, which at this point
        // we do not have available.

        // Solve for each column of the inverse matrix
        for i in 0 .. n {
            e[i] = T::one();

            let col = try!(self.solve(e));

            for j in 0 .. n {
                inv[[j, i]] = col[j];
            }

            e = col.apply(&|_| T::zero());
        }

        Ok(inv)
    }

    /// Computes the determinant of the decomposed matrix.
    pub fn det(&self) -> T {
        // Recall that the determinant of a triangular matrix
        // is the product of its diagonal entries. Also,
        // the determinant of L is implicitly 1.
        let u_det = self.lu.diag().fold(T::one(), |x, &y| x * y);
        // Note that the determinant of P is equal to the
        // determinant of P^T, so we don't have to invert it
        let p_det = self.p.clone().det();
        p_det * u_det
    }
}

/// Performs forward substitution using the LU matrix
/// for which L has an implicit unit diagonal. That is,
/// the strictly lower triangular part of LU corresponds
/// to the strictly lower triangular part of L.
///
/// This is equivalent to solving the system Lx = b.
fn lu_forward_substitution<T: Float>(lu: &Matrix<T>, b: Vector<T>) -> Vector<T> {
    assert!(lu.rows() == lu.cols(), "LU matrix must be square.");
    assert!(b.size() == lu.rows(), "LU matrix and RHS vector must be compatible.");
    let mut x = b;

    for (i, row) in lu.row_iter().enumerate().skip(1) {
        // Note that at time of writing we need raw_slice here for
        // auto-vectorization to kick in
        let adjustment = row.raw_slice()
                            .iter()
                            .take(i)
                            .cloned()
                            .zip(x.iter().cloned())
                            .fold(T::zero(), |sum, (l, x)| sum + l * x);

        x[i] = x[i] - adjustment;
    }
    x
}

fn unit_lower_triangular_part<T, M>(matrix: &M) -> Matrix<T>
    where T: Zero + One + Clone, M: BaseMatrix<T> {
    let (m, n) = (matrix.rows(), matrix.cols());
    let mut data = Vec::<T>::with_capacity(m * n);

    for (i, row) in matrix.row_iter().enumerate() {
        for element in row.iter().take(i).cloned() {
            data.push(element);
        }

        if i < n {
            data.push(T::one());
        }

        for _ in (i + 1) .. n {
            data.push(T::zero());
        }
    }

    Matrix::new(m, n, data)
}

fn nullify_lower_triangular_part<T: Zero>(mut matrix: Matrix<T>) -> Matrix<T> {
    for (i, mut row) in matrix.row_iter_mut().enumerate() {
        for element in row.iter_mut().take(i) {
            *element = T::zero();
        }
    }
    matrix
}


impl<T> Matrix<T> where T: Any + Float
{
    /// Computes L, U, and P for LUP decomposition.
    ///
    /// Returns L,U, and P respectively.
    ///
    /// This function is deprecated.
    /// Please see [PartialPivLu](decomposition/struct.PartialPivLu.html)
    /// for a replacement.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::matrix::Matrix;
    ///
    /// let a = matrix![1.0, 2.0, 0.0;
    ///                 0.0, 3.0, 4.0;
    ///                 5.0, 1.0, 2.0];
    ///
    /// let (l, u, p) = a.lup_decomp().expect("This matrix should decompose!");
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// - Matrix is not square.
    ///
    /// # Failures
    ///
    /// - Matrix cannot be LUP decomposed.
    #[deprecated]
    pub fn lup_decomp(self) -> Result<(Matrix<T>, Matrix<T>, Matrix<T>), Error> {
        let n = self.cols;
        assert!(self.rows == n, "Matrix must be square for LUP decomposition.");
        let mut l = Matrix::<T>::zeros(n, n);
        let mut u = self;
        let mut p = Matrix::<T>::identity(n);

        for index in 0..n {
            let mut curr_max_idx = index;
            let mut curr_max = u[[curr_max_idx, curr_max_idx]];

            for i in (curr_max_idx+1)..n {
                if u[[i, index]].abs() > curr_max.abs() {
                    curr_max = u[[i, index]];
                    curr_max_idx = i;
                }
            }
            if curr_max.abs() < T::epsilon() {
                return Err(Error::new(ErrorKind::DivByZero,
                    "Singular matrix found in LUP decomposition. \
                    A value in the diagonal of U == 0.0."));
            }

            if curr_max_idx != index {
                l.swap_rows(index, curr_max_idx);
                u.swap_rows(index, curr_max_idx);
                p.swap_rows(index, curr_max_idx);
            }
            l[[index, index]] = T::one();
            for i in (index+1)..n {
                let mult = u[[i, index]]/curr_max;
                l[[i, index]] = mult;
                u[[i, index]] = T::zero();
                for j in (index+1)..n {
                    u[[i, j]] = u[[i,j]] - mult*u[[index, j]];
                }
            }
        }
        Ok((l, u, p))
    }
}

#[cfg(test)]
mod tests {
    use matrix::{Matrix, PermutationMatrix};
    use testsupport::{is_lower_triangular, is_upper_triangular};

    use super::{PartialPivLu, LUP};
    use matrix::decomposition::Decomposition;

    use libnum::Float;

    #[allow(deprecated)]
    #[test]
    #[should_panic]
    fn test_non_square_lup_decomp() {
        let a: Matrix<f64> = Matrix::ones(2, 3);

        let _ = a.lup_decomp();
    }

    #[allow(deprecated)]
    #[test]
    fn test_lup_decomp() {
        use error::ErrorKind;
        let a: Matrix<f64> = matrix![
            1., 2., 3., 4.;
            0., 0., 0., 0.;
            0., 0., 0., 0.;
            0., 0., 0., 0.
        ];

        match a.lup_decomp() {
            Err(e) => assert!(*e.kind() == ErrorKind::DivByZero),
            Ok(_) => panic!()
        }
    }

    #[test]
    fn partial_piv_lu_decompose_arbitrary() {
        // Since the LUP decomposition is not in general unique,
        // we can not test against factors directly, but
        // instead we must rely on the fact that the
        // matrices P, L and U together construct the
        // original matrix
        let x = matrix![ -3.0,   0.0,   4.0,   1.0;
                        -12.0,   5.0,  17.0,   1.0;
                         15.0,   0.0, -18.0,  -5.0;
                          6.0,  20.0, -10.0, -15.0 ];

        let LUP { l, u, p } = PartialPivLu::decompose(x.clone())
                                           .unwrap()
                                           .unpack();
        let y = p.inverse() * &l * &u;
        assert_matrix_eq!(x, y, comp = float);
        assert!(is_lower_triangular(&l));
        assert!(is_upper_triangular(&u));
    }

    #[test]
    pub fn partial_piv_lu_inverse_identity() {
        let lu = PartialPivLu::<f64> {
            lu: Matrix::identity(3),
            p: PermutationMatrix::identity(3)
        };

        let inv = lu.inverse().expect("Matrix is invertible.");

        assert_matrix_eq!(inv, Matrix::identity(3), comp = float);
    }

    #[test]
    pub fn partial_piv_lu_inverse_arbitrary_invertible_matrix() {
        let x = matrix![5.0, 0.0, 0.0, 1.0;
                        2.0, 2.0, 2.0, 1.0;
                        4.0, 5.0, 5.0, 5.0;
                        1.0, 6.0, 4.0, 5.0];

        let inv = matrix![1.85185185185185203e-01,   1.85185185185185175e-01, -7.40740740740740561e-02, -1.02798428206033007e-17;
                          1.66666666666666630e-01,   6.66666666666666519e-01, -6.66666666666666519e-01,  4.99999999999999833e-01;
                         -3.88888888888888840e-01,   1.11111111111111174e-01,  5.55555555555555358e-01, -4.99999999999999833e-01;
                          7.40740740740740838e-02,  -9.25925925925925819e-01,  3.70370370370370294e-01,  5.13992141030165006e-17];

        let lu = PartialPivLu::decompose(x).unwrap();

        assert_matrix_eq!(lu.inverse().unwrap(), inv, comp = float);
    }

    #[test]
    pub fn partial_piv_lu_det_identity() {
        let lu = PartialPivLu::<f64> {
            lu: Matrix::identity(3),
            p: PermutationMatrix::identity(3)
        };

        assert_eq!(lu.det(), 1.0);
    }

    #[test]
    pub fn partial_piv_lu_det_arbitrary_invertible_matrix() {
        let x = matrix![ 5.0,  0.0,  0.0,  1.0;
                         0.0,  2.0,  2.0,  1.0;
                        15.0,  4.0,  7.0, 10.0;
                         5.0,  2.0, 17.0, 32.0];

        let lu = PartialPivLu::decompose(x).unwrap();

        let expected_det = 149.99999999999997;
        let diff = lu.det() - expected_det;
        assert!(diff.abs() < 1e-12);
    }

    #[test]
    pub fn partial_piv_lu_solve_arbitrary_matrix() {
        let x = matrix![ 5.0, 0.0, 0.0, 1.0;
                         2.0, 2.0, 2.0, 1.0;
                         4.0, 5.0, 5.0, 5.0;
                         1.0, 6.0, 4.0, 5.0 ];
        let b = vector![9.0, 16.0, 49.0, 45.0];
        let expected = vector![1.0, 2.0, 3.0, 4.0];

        let lu = PartialPivLu::decompose(x).unwrap();
        let y = lu.solve(b).unwrap();
        // Need to up the tolerance to take into account
        // numerical error. Ideally there'd be a more systematic
        // way to test this.
        assert_vector_eq!(y, expected, comp = ulp, tol = 100);
    }

    #[test]
    pub fn lu_forward_substitution() {
        use super::lu_forward_substitution;

        {
            let lu: Matrix<f64> = matrix![];
            let b = vector![];
            let x = lu_forward_substitution(&lu, b);
            assert!(x.size() == 0);
        }

        {
            let lu = matrix![3.0];
            let b = vector![1.0];
            let x = lu_forward_substitution(&lu, b);
            assert_eq!(x, vector![1.0]);
        }

        {
            let lu = matrix![3.0, 2.0;
                             2.0, 2.0];
            let b = vector![1.0, 2.0];
            let x = lu_forward_substitution(&lu, b);
            assert_eq!(x, vector![1.0, 0.0]);
        }
    }

}
