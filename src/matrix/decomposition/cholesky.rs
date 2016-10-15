use matrix::{Matrix, BaseMatrix};
use error::{Error, ErrorKind};

use std::any::Any;

use libnum::Float;

impl<T> Matrix<T>
    where T: Any + Float
{
    /// Cholesky decomposition
    ///
    /// Returns the cholesky decomposition of a positive definite matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, BaseMatrix};
    /// let m = Matrix::new(3,3, vec![1.0,0.5,0.5,0.5,1.0,0.5,0.5,0.5,1.0]);
    /// let l: Matrix<f64> = m.cholesky().unwrap();
    /// let exp: Matrix<f64> = Matrix::new(3, 3, vec![1.0, 0.0, 0.0, 0.5, 0.8660254, 0.0, 0.5, 0.28867513, 0.81649658]);
    /// assert_eq!(l.cols(), exp.cols());
    /// assert_eq!(l.rows(), exp.rows());
    /// for (a, b) in l.iter().zip(exp.iter()) {
    ///     assert!((a - b).abs() < 1.0e-6);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    ///
    /// # Failures
    ///
    /// - Matrix is not positive definite.
    pub fn cholesky(&self) -> Result<Matrix<T>, Error> {
        assert!(self.rows == self.cols,
                "Matrix must be square for Cholesky decomposition.");

        let mut new_data = Vec::<T>::with_capacity(self.rows() * self.cols());

        for i in 0..self.rows() {

            for j in 0..self.cols() {

                if j > i {
                    new_data.push(T::zero());
                    continue;
                }

                let mut sum = T::zero();
                for k in 0..j {
                    sum = sum + (new_data[i * self.cols() + k] * new_data[j * self.cols() + k]);
                }

                if j == i {
                    new_data.push((self[[i, i]] - sum).sqrt());
                } else {
                    let p = (self[[i, j]] - sum) / new_data[j * self.cols + j];

                    if !p.is_finite() {
                        return Err(Error::new(ErrorKind::DecompFailure,
                                              "Matrix is not positive definite."));
                    } else {

                    }
                    new_data.push(p);
                }
            }
        }

        Ok(Matrix {
            rows: self.rows(),
            cols: self.cols(),
            data: new_data,
        })
    }
}

#[cfg(test)]
mod tests {
    use matrix::Matrix;

    #[test]
    #[should_panic]
    fn test_non_square_cholesky() {
        let a = Matrix::new(2, 3, vec![1.0; 6]);

        let _ = a.cholesky();
    }
}
