use matrix::{Matrix, BaseMatrixMut};
use error::{Error};

use std::any::Any;

use libnum::Float;

impl<T> Matrix<T> where T: Any + Float
{
    /// Computes the LUP decomposition.
    ///
    /// For a given square matrix `A`, returns `(L, U, P)` such that
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
    /// use rulinalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(3,3, vec![1.0, 2.0, 0.0,
    ///                               0.0, 3.0, 4.0,
    ///                               5.0, 1.0, 2.0]);
    ///
    /// let (l,u,p) = a.lup_decomp().expect("This matrix should decompose!");
    /// ```
    ///
    /// # Panics
    ///
    /// - Matrix is not square.
    ///
    /// # Failures
    ///
    /// - Can not fail. `Result` will be removed in future releases.
    pub fn lup_decomp(&self) -> Result<(Matrix<T>, Matrix<T>, Matrix<T>), Error> {
        let n = self.cols;
        assert!(self.rows == n, "Matrix must be square for LUP decomposition.");
        let mut l = Matrix::<T>::zeros(n, n);
        let mut u = self.clone();
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
        Ok((l, u, p))
    }
}

#[cfg(test)]
mod tests {
    use matrix::Matrix;

    #[test]
    #[should_panic]
    fn test_non_square_lup_decomp() {
        let a: Matrix<f64> = Matrix::ones(2, 3);

        let _ = a.lup_decomp();
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
            let (l, u, p) = x.lup_decomp().unwrap();
            assert_matrix_eq!(x, p.transpose() * l * u, comp = float);
        }

        {
            let x = matrix![2.0, 0.0, 0.0;
                            5.0, 0.0, 0.0;
                            3.0, 1.0, 3.0];
            let (l, u, p) = x.lup_decomp().unwrap();
            assert_matrix_eq!(x, p.transpose() * l * u, comp = float);
        }

        {
            let x = matrix![1.0, 2.0, 3.0;
                            3.0, 2.0, 1.0;
                            4.0, 4.0, 4.0];
            let (l, u, p) = x.lup_decomp().unwrap();
            assert_matrix_eq!(x, p.transpose() * l * u, comp = float);
        }
    }
}
