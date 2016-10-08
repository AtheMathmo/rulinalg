use matrix::{Matrix, BaseMatrix};
use error::{Error, ErrorKind};
use utils;

use std::any::Any;
use std::ops::{Mul, Add, Div, Sub, Neg};

use libnum::{One, Zero};

impl<T> Matrix<T> where T: Any + Copy + One + Zero + Neg<Output=T> +
                           Add<T, Output=T> + Mul<T, Output=T> +
                           Sub<T, Output=T> + Div<T, Output=T> +
                           PartialOrd
{

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

        let mt = self.transpose();

        let mut p = Matrix::<T>::identity(n);

        // Compute the permutation matrix
        for i in 0..n {
            let (row,_) = utils::argmax(&mt.data[i*(n+1)..(i+1)*n]);

            if row != 0 {
                for j in 0..n {
                    p.data.swap(i*n + j, row*n+j)
                }
            }
        }

        let a_2 = &p * self;

        for i in 0..n {
            l.data[i*(n+1)] = T::one();

            for j in 0..i+1 {
                let mut s1 = T::zero();

                for k in 0..j {
                    s1 = s1 + l.data[j*n + k] * u.data[k*n + i];
                }

                u.data[j*n + i] = a_2[[j,i]] - s1;
            }

            for j in i..n {
                let mut s2 = T::zero();

                for k in 0..i {
                    s2 = s2 + l.data[j*n + k] * u.data[k*n + i];
                }

                let denom = u[[i,i]];

                if denom == T::zero() {
                    return Err(Error::new(ErrorKind::DivByZero,
                        "Singular matrix found in LUP decomposition. \
                        A value in the diagonal of U == 0.0."));
                }
                l.data[j*n + i] = (a_2[[j,i]] - s2) / denom;
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
        let a = Matrix::new(2, 3, vec![1.0; 6]);

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
}