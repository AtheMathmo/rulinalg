use matrix::{Matrix, BaseMatrix, BaseMatrixMut};
use error::{Error, ErrorKind};
use utils;

use std::any::Any;

use libnum::Float;

impl<T> Matrix<T> where T: Any + Float
{

    fn produce_pivot(&self) -> Matrix<T> {
        let n = self.rows();

        let mut p = Matrix::<T>::identity(n);

        // Compute the permutation matrix
        for i in 0..n {
            // Find the max value in each column
            let mut curr_max_idx = i;
            let mut curr_max = self[[i, i]];

            for j in i+1..n {
                if self[[j,i]] > curr_max {
                    curr_max = self[[j,i]];
                    curr_max_idx = j;
                }
            }

            if curr_max_idx != i {
                p.swap_rows(i, curr_max_idx);
            }
        }

        p
    }

    /// Computes L, U, and P for LUP decomposition.
    ///
    /// Returns L,U, and P respectively.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(3,3, vec![1.0,2.0,0.0,
    ///                               0.0,3.0,4.0,
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
    /// - Matrix cannot be LUP decomposed.
    pub fn lup_decomp(&self) -> Result<(Matrix<T>, Matrix<T>, Matrix<T>), Error> {
        let n = self.cols;
        assert!(self.rows == n, "Matrix must be square for LUP decomposition.");

        let mut l = Matrix::<T>::zeros(n, n);
        let mut u = Matrix::<T>::zeros(n, n);
        let p = self.produce_pivot();
        let a_2 = &p * self;

        for i in 0..n {
            l[[i, i]] = T::one();

            for j in 0..i+1 {
                let mut s1 = T::zero();

                for k in 0..j {
                    s1 = s1 + l[[j, k]] * u[[k, i]];
                }

                u[[j ,i]] = a_2[[j,i]] - s1;
            }

            for j in i..n {
                let mut s2 = T::zero();

                for k in 0..i {
                    s2 = s2 + l[[j, k]] * u[[k, i]];
                }

                let denom = u[[i,i]];

                if denom.abs() < T::epsilon() {
                    return Err(Error::new(ErrorKind::DivByZero,
                        "Singular matrix found in LUP decomposition. \
                        A value in the diagonal of U == 0.0."));
                }
                
                l[[j, i]] = (a_2[[j,i]] - s2) / denom;
            }

        }

        Ok((l,u,p))
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
    fn test_lup_decomp() {
        use error::ErrorKind;
        let a: Matrix<f64> = matrix!(
            1., 2., 3., 4.;
            0., 0., 0., 0.;
            0., 0., 0., 0.;
            0., 0., 0., 0.
        );

        match a.lup_decomp() {
            Err(e) => assert!(*e.kind() == ErrorKind::DivByZero),
            Ok(_) => panic!()
        }
    }

    #[test]
    fn test_basic_pivot() {
        let a = matrix![5f64,4.,3.,2.,1.;
                        4.,3.,2.,1.,5.;
                        3.,2.,1.,5.,4.;
                        2.,1.,5.,4.,3.;
                        1.,5.,4.,3.,2.];
        let p = a.produce_pivot();

        let true_p = matrix![1f64,0.,0.,0.,0.;
                            0.,0.,0.,0.,1.;
                            0.,0.,0.,1.,0.;
                            0.,0.,1.,0.,0.;
                            0.,1.,0.,0.,0.];

        assert!(p.data().iter()
                    .zip(true_p.data().iter())
                    .all(|(&x,&y)| (x-y).abs() == 0.0));
    }
}
