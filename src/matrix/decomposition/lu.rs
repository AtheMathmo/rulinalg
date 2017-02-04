use matrix::{Matrix, BaseMatrix, BaseMatrixMut};
use matrix::{forward_substitution, back_substitution};
use vector::Vector;
use error::{Error, ErrorKind};

use std::any::Any;

use libnum::Float;

use matrix::decomposition::Decomposition;

/// TODO: docs
#[derive(Debug, Clone)]
pub struct LUP<T> {
    /// The lower triangular matrix in the decomposition.
    pub l: Matrix<T>,
    /// The upper triangular matrix in the decomposition.
    pub u: Matrix<T>,
    /// The permutation matrix in the decomposition.
    pub p: Matrix<T>
}

/// TODO: Docs
#[derive(Debug, Clone)]
pub struct PartialPivLu<T> {
    // For now, we store the full matrices, but
    // we can improve this by storing the decomposition
    // in the input matrix such that L and U can be stored
    // in the space of a single matrix
    lup: LUP<T>
}

impl<T> Decomposition for PartialPivLu<T> {
    type Factors = LUP<T>;

    fn unpack(self) -> LUP<T> {
        self.lup
    }
}

impl<T: 'static + Float> PartialPivLu<T> {
    /// TODO
    pub fn decompose(matrix: Matrix<T>) -> Result<Self, Error> {
        matrix.lup_decomp().map(|(l, u, p)|
            PartialPivLu {
                lup: LUP {
                    l: l,
                    u: u,
                    p: p
                }
            }
        )
    }
}

impl<T> PartialPivLu<T> where T: Any + Float {
    /// TODO
    pub fn solve(&self, b: Vector<T>) -> Result<Vector<T>, Error> {
        let b = try!(forward_substitution(&self.lup.l, &self.lup.p * b));
        back_substitution(&self.lup.u, b)
    }

    /// Computes the inverse of the matrix which this LUP decomposition
    /// represents.
    pub fn inverse(&self) -> Result<Matrix<T>, Error> {
        let n = self.lup.u.rows();
        let mut inv = Matrix::zeros(n, n);
        let mut e = Vector::zeros(n);

        // To compute the inverse of a matrix A, note that
        // we can simply solve the system
        // AX = I,
        // where X is the inverse of A, and I is the identity
        // matrix of appropriate dimension.

        // Solve for each column of the inverse matrix
        for i in 0 .. n {
            e[i] = T::one();

            let y = try!(self.solve(e));

            for j in 0 .. n {
                inv[[j, i]] = y[j];
            }

            e = Vector::zeros(n);
        }

        Ok(inv)
    }

    /// Computes the determinant of the decomposed matrix.
    pub fn det(&self) -> T {
        use matrix::parity;

        // Recall that the determinant of a triangular matrix
        // is the product of its diagonal entries
        let u_det = self.lup.u.diag().fold(T::one(), |x, &y| x * y);
        let l_det = self.lup.l.diag().fold(T::one(), |x, &y| x * y);
        // The determinant of a permutation matrix is simply its parity
        let p_det = parity(&self.lup.p);
        p_det * u_det * l_det
    }
}


impl<T> Matrix<T> where T: Any + Float
{
    /// Computes L, U, and P for LUP decomposition.
    ///
    /// Returns L,U, and P respectively.
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
    use matrix::{Matrix, BaseMatrix};
    use testsupport::{is_lower_triangular, is_upper_triangular};

    use super::{PartialPivLu, LUP};
    use matrix::decomposition::Decomposition;

    use libnum::Float;

    #[test]
    #[should_panic]
    fn test_non_square_lup_decomp() {
        let a: Matrix<f64> = Matrix::ones(2, 3);

        let _ = a.lup_decomp();
    }

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
    fn lup_decompose_arbitrary() {
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
        let y = p.transpose() * &l * &u;
        assert_matrix_eq!(x, y, comp = float);
        assert!(is_lower_triangular(&l));
        assert!(is_upper_triangular(&u));
    }

    #[test]
    pub fn partial_piv_lu_inverse_identity() {
        let lu = PartialPivLu::<f64> {
            lup: LUP {
                l: Matrix::identity(3),
                u: Matrix::identity(3),
                p: Matrix::identity(3)
            }
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
            lup: LUP {
                l: Matrix::identity(3),
                u: Matrix::identity(3),
                p: Matrix::identity(3)
            }
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

}
