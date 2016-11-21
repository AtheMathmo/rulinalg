use matrix::{Matrix, BaseMatrixMut, forward_substitution, back_substitution};
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
}

#[cfg(test)]
mod tests {
    use matrix::{Matrix, Decomposition};
    use matrix::LU;

    #[test]
    #[should_panic]
    fn test_non_square_lup_decomp() {
        let a: Matrix<f64> = Matrix::ones(2, 3);

        let _ = a.lu();
    }

    #[test]
    fn test_lup_decomp_singular_matrix() {
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
}
