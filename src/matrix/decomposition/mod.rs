//! Matrix Decompositions
//!
//! References:
//! 1. [On Matrix Balancing and EigenVector computation]
//! (http://arxiv.org/pdf/1401.5766v1.pdf), James, Langou and Lowery
//!
//! 2. [The QR algorithm for eigen decomposition]
//! (http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter4.pdf)
//!
//! 3. [Computation of the SVD]
//! (http://www.cs.utexas.edu/users/inderjit/public_papers/HLA_SVD.pdf)

mod qr;
mod cholesky;
mod bidiagonal;
mod svd;
mod hessenberg;
mod lu;
mod eigen;

use std::any::Any;

use matrix::{Matrix, BaseMatrix};
use norm::Euclidean;
use vector::Vector;
use utils;
use error::{Error, ErrorKind};

use libnum::Float;

impl<T> Matrix<T>
    where T: Any + Float
{
    /// Compute the cos and sin values for the givens rotation.
    ///
    /// Returns a tuple (c, s).
    fn givens_rot(a: T, b: T) -> (T, T) {
        let r = a.hypot(b);

        (a / r, -b / r)
    }

    fn make_householder(column: &[T]) -> Result<Matrix<T>, Error> {
        let size = column.len();

        if size == 0 {
            return Err(Error::new(ErrorKind::InvalidArg,
                                  "Column for householder transform cannot be empty."));
        }

        let denom = column[0] + column[0].signum() * utils::dot(column, column).sqrt();

        if denom == T::zero() {
            return Err(Error::new(ErrorKind::DecompFailure,
                                  "Cannot produce househoulder transform from column as first \
                                   entry is 0."));
        }

        let mut v = column.into_iter().map(|&x| x / denom).collect::<Vec<T>>();
        // Ensure first element is fixed to 1.
        v[0] = T::one();
        let v = Vector::new(v);
        let v_norm_sq = v.dot(&v);

        let v_vert = Matrix::new(size, 1, v.data().clone());
        let v_hor = Matrix::new(1, size, v.into_vec());
        Ok(Matrix::<T>::identity(size) - (v_vert * v_hor) * ((T::one() + T::one()) / v_norm_sq))
    }

    fn make_householder_vec(column: &[T]) -> Result<Matrix<T>, Error> {
        let size = column.len();

        if size == 0 {
            return Err(Error::new(ErrorKind::InvalidArg,
                                  "Column for householder transform cannot be empty."));
        }

        let denom = column[0] + column[0].signum() * utils::dot(column, column).sqrt();

        if denom == T::zero() {
            return Err(Error::new(ErrorKind::DecompFailure,
                                  "Cannot produce househoulder transform from column as first \
                                   entry is 0."));
        }

        let mut v = column.into_iter().map(|&x| x / denom).collect::<Vec<T>>();
        // Ensure first element is fixed to 1.
        v[0] = T::one();
        let v = Matrix::new(size, 1, v);

        Ok(&v / v.norm(Euclidean))
    }
}
