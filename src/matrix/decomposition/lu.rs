use matrix::{
    Matrix, BaseMatrix, BaseMatrixMut,
    forward_substitution, back_substitution
};
use vector::Vector;
use error::{Error};
use matrix::Decomposition;

use std::any::Any;

use libnum::Float;

/// The L, U and P matrices in the LUP decomposition.
#[derive(Debug, Clone)]
pub struct LU<T> {
    /// The lower triangular matrix in the LUP decomposition.
    pub l: Matrix<T>,
    /// The upper triangular matrix in the LUP decomposition.
    pub u: Matrix<T>,
    /// The permutation matrix in the LUP decomposition.
    pub p: Matrix<T>
}

/// TODO
#[derive(Debug, Clone)]
pub struct LuDecomposition<T> {
    // For now, we store the separate factors, but in the future
    // we can greatly reduce memory usage by storing both L and U
    // in the space of a single matrix
    lu: LU<T>
}

impl<T> Decomposition for LuDecomposition<T> {
    type Factors = LU<T>;

    fn decompose(self) -> Self::Factors {
        self.lu
    }
}

impl<T> Matrix<T> where T: Any + Float {
    /// Computes the LUP decomposition.
    ///
    /// For a given square matrix `A`, returns L, U, P such that
    ///
    /// ```text
    /// PA = LU
    /// ```
    ///
    /// where `P` is a permutation matrix, and `L` and `U` are
    /// lower and upper triangular matrices, respectively.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, Decomposition, LU};
    ///
    /// let a = Matrix::new(3,3, vec![1.0, 2.0, 0.0,
    ///                               0.0, 3.0, 4.0,
    ///                               5.0, 1.0, 2.0]);
    ///
    /// let LU { l, u, p } = a.lu().decompose();
    /// ```
    ///
    /// # Panics
    ///
    /// - Matrix is not square.
    pub fn lu(self) -> LuDecomposition<T> {
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

            // Recall that a pivot column means that there is a 1
            // in the diagonal position in the reduced echelon form.
            // Here we consider values smaller than machine epsilon to
            // be zero.
            let is_pivot_col = curr_max.abs() > T::epsilon();

            if is_pivot_col && curr_max_idx != index {
                l.swap_rows(index, curr_max_idx);
                u.swap_rows(index, curr_max_idx);
                p.swap_rows(index, curr_max_idx);
            }

            l[[index, index]] = T::one();
            for i in (index + 1)..n {
                let mult = if is_pivot_col { u[[i, index]] / curr_max }
                           else { T::zero() };
                l[[i, index]] = mult;
                u[[i, index]] = T::zero();
                for j in (index + 1)..n {
                    u[[i, j]] = u[[i,j]] - mult * u[[index, j]];
                }
            }
        }

        LuDecomposition {
            lu: LU {
                l: l,
                u: u,
                p: p
            }
        }
    }
}


impl<T> LuDecomposition<T> where T: Any + Float {
    /// TODO
    pub fn solve(&self, b: Vector<T>) -> Result<Vector<T>, Error> {
        let b = try!(forward_substitution(&self.lu.l, &self.lu.p * b));
        back_substitution(&self.lu.u, b)
    }

    /// Computes the inverse of the matrix which this LUP decomposition
    /// represents.
    pub fn inverse(&self) -> Result<Matrix<T>, Error> {
        let n = self.lu.u.rows();
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
        use matrix::DiagOffset::Main;

        // Recall that the determinant of a triangular matrix
        // is the product of its diagonal entries
        let u_det = self.lu.u.iter_diag(Main).fold(T::one(), |x, &y| x * y);
        let l_det = self.lu.l.iter_diag(Main).fold(T::one(), |x, &y| x * y);
        // The determinant of a permutation matrix is simply its parity
        let p_det = parity(&self.lu.p);
        p_det * u_det * l_det
    }
}

#[cfg(test)]
mod tests {
    use super::LuDecomposition;
    use matrix::{Matrix, Decomposition};
    use matrix::LU;
    use libnum::Float;

    #[test]
    #[should_panic]
    fn test_non_square_lup_decomp() {
        let a: Matrix<f64> = Matrix::ones(2, 3);

        let _ = a.lu();
    }

    #[test]
    fn test_lu_singular_matrix() {
        // The LUP decomposition for a singular matrix
        // is not unique, so we can not explicitly check the
        // L, U and P matrices against known values directly.
        // However, we can verify that X = P^T L U,
        // and that P is a valid permutation matrix,
        // L is a lower triangular matrix and that U
        // is upper triangular.
        // However, there are some plans for a more general
        // type-level framework for verifying structural matrix constraints,
        // and so for now we only check that X = P^T L U.

        {
            let x = matrix![2.0, 5.0, 3.0;
                            0.0, 0.0, 1.0;
                            0.0, 0.0, 3.0];
            let LU { l, u, p } = x.clone().lu().decompose();
            assert_matrix_eq!(x, p.transpose() * l * u, comp = float);
        }

        {
            let x = matrix![2.0, 0.0, 0.0;
                            5.0, 0.0, 0.0;
                            3.0, 1.0, 3.0];
            let LU { l, u, p } = x.clone().lu().decompose();
            assert_matrix_eq!(x, p.transpose() * l * u, comp = float);
        }

        {
            let x = matrix![1.0, 2.0, 3.0;
                            3.0, 2.0, 1.0;
                            4.0, 4.0, 4.0];
            let LU { l, u, p } = x.clone().lu().decompose();
            assert_matrix_eq!(x, p.transpose() * l * u, comp = float);
        }
    }

    #[test]
    pub fn lu_inverse_identity() {
        let lu = LuDecomposition::<f64> {
            lu: LU {
                l: Matrix::identity(3),
                u: Matrix::identity(3),
                p: Matrix::identity(3)
            }
        };

        let inv = lu.inverse().expect("Matrix is invertible.");

        assert_matrix_eq!(inv, Matrix::identity(3), comp = float);
    }

    #[test]
    pub fn lu_inverse_arbitrary_invertible_matrix() {
        let x = matrix![5.0, 0.0, 0.0, 1.0;
                        2.0, 2.0, 2.0, 1.0;
                        4.0, 5.0, 5.0, 5.0;
                        1.0, 6.0, 4.0, 5.0];

        let inv = matrix![1.85185185185185203e-01,   1.85185185185185175e-01, -7.40740740740740561e-02, -1.02798428206033007e-17;
                          1.66666666666666630e-01,   6.66666666666666519e-01, -6.66666666666666519e-01,  4.99999999999999833e-01;
                         -3.88888888888888840e-01,   1.11111111111111174e-01,  5.55555555555555358e-01, -4.99999999999999833e-01;
                          7.40740740740740838e-02,  -9.25925925925925819e-01,  3.70370370370370294e-01,  5.13992141030165006e-17];

        assert_matrix_eq!(inv, x.lu().inverse().unwrap(), comp = float);
    }

    #[test]
    pub fn lu_inverse_arbitrary_singular_matrix() {
        let x = matrix![ 5.0,  0.0,  0.0,  1.0;
                         0.0,  2.0,  2.0,  1.0;
                        15.0,  4.0,  4.0, 10.0;
                         5.0,  2.0,  2.0, 32.0];
        assert!(x.lu().inverse().is_err());
    }

    #[test]
    pub fn lu_det_identity() {
        let lu = LuDecomposition::<f64> {
            lu: LU {
                l: Matrix::identity(3),
                u: Matrix::identity(3),
                p: Matrix::identity(3)
            }
        };

        assert_eq!(lu.det(), 1.0);
    }

    #[test]
    pub fn lu_det_arbitrary_invertible_matrix() {
        let x = matrix![ 5.0,  0.0,  0.0,  1.0;
                         0.0,  2.0,  2.0,  1.0;
                        15.0,  4.0,  7.0, 10.0;
                         5.0,  2.0, 17.0, 32.0];

        let expected_det = 149.99999999999997;
        let diff = x.lu().det() - expected_det;
        assert!(diff.abs() < 1e-12);
    }

    #[test]
    pub fn lu_det_arbitrary_singular_matrix() {
        let x = matrix![ 5.0,  0.0,  0.0,  1.0;
                         0.0,  2.0,  2.0,  1.0;
                        15.0,  4.0,  4.0, 10.0;
                         5.0,  2.0,  2.0, 32.0];
        assert_eq!(x.lu().det(), 0.0);
    }

    #[test]
    pub fn lu_det_arbitrary_matrices() {
        let x = matrix![-3.0, -1.0, -5.0,  1.0;
                         4.0,  3.0,  2.0, -3.0;
                         1.0, -1.0,  4.0, -5.0;
                         4.0,  3.0,  2.0,  5.0];

        let expected_det = 56.0;
        let diff = x.lu().det() - expected_det;
        assert!(diff.abs() < 1e-13);
    }
}
