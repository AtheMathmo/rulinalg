use matrix::{Matrix, MatrixSlice, BaseMatrix};
use error::{Error, ErrorKind};

use std::any::Any;

use libnum::Float;

impl<T> Matrix<T>
    where T: Any + Float
{
    /// Compute the QR decomposition of the matrix.
    ///
    /// Returns the tuple (Q,R).
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let m = Matrix::new(3,3, vec![1.0,0.5,0.5,0.5,1.0,0.5,0.5,0.5,1.0]);
    ///
    /// let (q, r) = m.qr_decomp().unwrap();
    /// ```
    ///
    /// # Failures
    ///
    /// - Cannot compute the QR decomposition.
    pub fn qr_decomp(self) -> Result<(Matrix<T>, Matrix<T>), Error> {
        let m = self.rows();
        let n = self.cols();

        let mut q = Matrix::<T>::identity(m);
        let mut r = self;

        for i in 0..(n - ((m == n) as usize)) {
            let holder_transform: Result<Matrix<T>, Error>;
            {
                let lower_slice = MatrixSlice::from_matrix(&r, [i, i], m - i, 1);
                holder_transform =
                    Matrix::make_householder(&lower_slice.iter().cloned().collect::<Vec<_>>());
            }

            if !holder_transform.is_ok() {
                return Err(Error::new(ErrorKind::DecompFailure,
                                      "Cannot compute QR decomposition."));
            } else {
                let mut holder_data = holder_transform.unwrap().into_vec();

                // This bit is inefficient
                // using for now as we'll swap to lapack eventually.
                let mut h_full_data = Vec::with_capacity(m * m);

                for j in 0..m {
                    let mut row_data: Vec<T>;
                    if j < i {
                        row_data = vec![T::zero(); m];
                        row_data[j] = T::one();
                        h_full_data.extend(row_data);
                    } else {
                        row_data = vec![T::zero(); i];
                        h_full_data.extend(row_data);
                        h_full_data.extend(holder_data.drain(..m - i));
                    }
                }

                let h = Matrix::new(m, m, h_full_data);

                q = q * &h;
                r = h * &r;
            }
        }

        Ok((q, r))
    }
}