//! Operation for Invertible `Matrix`


use std::any::Any;
use std::ops::Mul;

use matrix::{Matrix, MatrixSlice, MatrixSliceMut};
use matrix::slice::BaseSlice;
use matrix::decomposition::Decomposition;
use matrix::{back_substitution, forward_substitution, parity};
use vector::Vector;
use error::{Error, ErrorKind};

use libnum::Float;


/// Invertible Matrix
pub trait Invertible<T>: Decomposition<T> {
    /// Solves the equation `Ax = y`.
    ///
    /// Requires a Vector `y` as input.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::vector::Vector;
    /// use rulinalg::matrix::invertible::Invertible;
    ///
    /// let a = Matrix::new(2,2, vec![2.0,3.0,1.0,2.0]);
    /// let y = Vector::new(vec![13.0,8.0]);
    ///
    /// let x = a.solve(y).unwrap();
    ///
    /// assert_eq!(*x.data(), vec![2.0, 3.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix column count and vector size are different.
    /// - The matrix is not square.
    ///
    /// # Failures
    ///
    /// - The matrix cannot be decomposed into an LUP form to solve.
    /// - There is no valid solution as the matrix is singular.
    fn solve(&self, y: Vector<T>) -> Result<Vector<T>, Error>
        where T: Any + Float,
              for <'a> &'a Matrix<T>: Mul<&'a Self, Output=Matrix<T>>,
    {
        let (l, u, p) = try!(self.lup_decomp());

        let b = try!(forward_substitution(&l, p * y));
        back_substitution(&u, b)
    }

    /// Computes the inverse of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::invertible::Invertible;
    ///
    /// let a = Matrix::new(2,2, vec![2.,3.,1.,2.]);
    /// let inv = a.inverse().expect("This matrix should have an inverse!");
    ///
    /// let I = a * inv;
    ///
    /// assert_eq!(*I.data(), vec![1.0,0.0,0.0,1.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    ///
    /// # Failures
    ///
    /// - The matrix could not be LUP decomposed.
    /// - The matrix has zero determinant.
    fn inverse(&self) -> Result<Matrix<T>, Error>
        where T: Any + Float,
              for <'a> &'a Matrix<T>: Mul<&'a Self, Output=Matrix<T>>,
              for <'a> &'a Matrix<T>: Mul<Vector<T>, Output=Vector<T>>
    {
        assert!(self.rows() == self.cols(), "Matrix is not square.");

        let mut inv_t_data = Vec::<T>::new();
        let (l, u, p) = try!(self.lup_decomp().map_err(|_| {
            Error::new(ErrorKind::DecompFailure,
                       "Could not compute LUP factorization for inverse.")
        }));

        let mut d = T::one();

        unsafe {
            for i in 0..l.cols {
                d = d * *l.get_unchecked([i, i]);
                d = d * *u.get_unchecked([i, i]);
            }
        }

        if d == T::zero() {
            return Err(Error::new(ErrorKind::DecompFailure,
                                  "Matrix is singular and cannot be inverted."));
        }

        for i in 0..self.rows() {
            let mut id_col = vec![T::zero(); self.cols()];
            id_col[i] = T::one();

            let b = forward_substitution(&l, &p * Vector::new(id_col))
                .expect("Matrix is singular AND has non-zero determinant!?");
            inv_t_data.append(&mut back_substitution(&u, b)
                .expect("Matrix is singular AND has non-zero determinant!?")
                .into_vec());

        }

        Ok(Matrix::new(self.rows(), self.cols(), inv_t_data).transpose())
    }

    /// Computes the determinant of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::invertible::Invertible;
    ///
    /// let a = Matrix::new(3,3, vec![1.0,2.0,0.0,
    ///                               0.0,3.0,4.0,
    ///                               5.0, 1.0, 2.0]);
    ///
    /// let det = a.det();
    ///
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    fn det(&self) -> T
        where T: Any + Float,
              for <'a> &'a Matrix<T>: Mul<&'a Self, Output=Matrix<T>>,
    {
        assert!(self.rows() == self.cols(), "Matrix is not square.");

        let n = self.cols();

        if self.is_diag() {
            let mut d = T::one();

            unsafe {
                for i in 0..n {
                    d = d * *self.get_unchecked([i, i]);
                }
            }

            return d;
        }

        if n == 2 {
            unsafe {
                (*self.get_unchecked([0, 0]) * *self.get_unchecked([1, 1])) - 
                    (*self.get_unchecked([0, 1]) * *self.get_unchecked([1, 0]))
            }
        } else if n == 3 {
            unsafe {
                (*self.get_unchecked([0, 0]) * *self.get_unchecked([1, 1]) * *self.get_unchecked([2, 2])) +
                (*self.get_unchecked([0, 1]) * *self.get_unchecked([1, 2]) * *self.get_unchecked([2, 0])) +
                (*self.get_unchecked([0, 2]) * *self.get_unchecked([1, 0]) * *self.get_unchecked([2, 1])) -
                (*self.get_unchecked([0, 0]) * *self.get_unchecked([1, 2]) * *self.get_unchecked([2, 1])) -
                (*self.get_unchecked([0, 1]) * *self.get_unchecked([1, 0]) * *self.get_unchecked([2, 2])) -
                (*self.get_unchecked([0, 2]) * *self.get_unchecked([1, 1]) * *self.get_unchecked([2, 0]))
            }
        } else {
            let (l, u, p) = self.lup_decomp().expect("Could not compute LUP decomposition.");

            let mut d = T::one();

            unsafe {
                for i in 0..l.cols {
                    d = d * *l.get_unchecked([i, i]);
                    d = d * *u.get_unchecked([i, i]);
                }
            }

            let sgn = parity(&p);

            sgn * d
        }
    }
}

impl<T> Invertible<T> for Matrix<T> {}
impl<'a, T> Invertible<T> for MatrixSlice<'a, T> {}
impl<'a, T> Invertible<T> for MatrixSliceMut<'a, T> {}


#[cfg(test)]
mod tests {

    use matrix::invertible::Invertible;
    use matrix::Matrix;
    use vector::Vector;
    use libnum::abs;

    #[test]
    fn matrix_det() {
        let a = Matrix::new(2, 2, vec![2., 3., 1., 2.]);
        let b = a.det();

        assert_eq!(b, 1.);

        let c = Matrix::new(3, 3, vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let d = c.det();

        assert_eq!(d, 0.);

        let e = Matrix::<f64>::new(5,
                                   5,
                                   vec![1., 2., 3., 4., 5., 3., 0., 4., 5., 6., 2., 1., 2., 3.,
                                        4., 0., 0., 0., 6., 5., 0., 0., 0., 5., 6.]);

        let f = e.det();

        println!("det is {0}", f);
        let error = abs(f - 99.);
        assert!(error < 1e-10);
    }

    #[test]
    fn matrix_solve() {
        let a = Matrix::new(2, 2, vec![2., 3., 1., 2.]);

        let y = Vector::new(vec![8., 5.]);

        let x = a.solve(y).unwrap();

        assert_eq!(x.size(), 2);

        assert_eq!(x[0], 1.);
        assert_eq!(x[1], 2.);
    }


}
