//! The matrix module.
//!
//! Currently contains all code
//! relating to the matrix linear algebra struct.
//!
//! Most of the logic for manipulating matrices is generically implemented
//! via `BaseMatrix` and `BaseMatrixMut` trait.

use std;
use std::any::Any;
use std::marker::PhantomData;
use libnum::Float;

use error::{Error, ErrorKind};
use utils;
use vector::Vector;

mod decomposition;
mod impl_ops;
mod impl_mat;
mod mat_mul;
mod iter;
mod deref;
mod slice;
mod base;
mod permutation_matrix;
mod impl_permutation_mul;

pub use self::base::{BaseMatrix, BaseMatrixMut};
pub use self::permutation_matrix::{PermutationMatrix, Parity};

/// Matrix dimensions
#[derive(Debug, Clone, Copy)]
pub enum Axes {
    /// The row axis.
    Row,
    /// The column axis.
    Col,
}

/// The `Matrix` struct.
///
/// Can be instantiated with any type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

/// A `MatrixSlice`
///
/// This struct provides a slice into a matrix.
///
/// The struct contains the upper left point of the slice
/// and the width and height of the slice.
#[derive(Debug, Clone, Copy)]
pub struct MatrixSlice<'a, T: 'a> {
    ptr: *const T,
    rows: usize,
    cols: usize,
    row_stride: usize,
    marker: PhantomData<&'a T>,
}

/// A mutable `MatrixSliceMut`
///
/// This struct provides a mutable slice into a matrix.
///
/// The struct contains the upper left point of the slice
/// and the width and height of the slice.
#[derive(Debug)]
pub struct MatrixSliceMut<'a, T: 'a> {
    ptr: *mut T,
    rows: usize,
    cols: usize,
    row_stride: usize,
    marker: PhantomData<&'a mut T>,
}

/// Row of a matrix.
///
/// This struct points to a slice making up
/// a row in a matrix. You can deref this
/// struct to retrieve a `MatrixSlice` of
/// the row.
///
/// # Example
///
/// ```
/// # #[macro_use] extern crate rulinalg; fn main() {
/// use rulinalg::matrix::BaseMatrix;
///
/// let mat = matrix![1.0, 2.0;
///                   3.0, 4.0];
///
/// let row = mat.row(1);
/// assert_eq!((*row + 2.0).sum(), 11.0);
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Row<'a, T: 'a> {
    row: MatrixSlice<'a, T>,
}

/// Mutable row of a matrix.
///
/// This struct points to a mutable slice
/// making up a row in a matrix. You can deref
/// this struct to retrieve a `MatrixSlice`
/// of the row.
///
/// # Example
///
/// ```
/// # #[macro_use] extern crate rulinalg; fn main() {
/// use rulinalg::matrix::BaseMatrixMut;
///
/// let mut mat = matrix![1.0, 2.0;
///                       3.0, 4.0];
///
/// {
///     let mut row = mat.row_mut(1);
///     *row += 2.0;
/// }
/// let expected = matrix![1.0, 2.0;
///                        5.0, 6.0];
/// assert_matrix_eq!(mat, expected);
/// # }
/// ```
#[derive(Debug)]
pub struct RowMut<'a, T: 'a> {
    row: MatrixSliceMut<'a, T>,
}


// MAYBE WE SHOULD MOVE SOME OF THIS STUFF OUT
//

impl<'a, T: 'a> Row<'a, T> {
    /// Returns the row as a slice.
    pub fn raw_slice(&self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.row.as_ptr(), self.row.cols()) }
    }
}

impl<'a, T: 'a> RowMut<'a, T> {
    /// Returns the row as a slice.
    pub fn raw_slice(&self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.row.as_ptr(), self.row.cols()) }
    }

    /// Returns the row as a slice.
    pub fn raw_slice_mut(&mut self) -> &'a mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.row.as_mut_ptr(), self.row.cols()) }
    }
}

/// Row iterator.
#[derive(Debug)]
pub struct Rows<'a, T: 'a> {
    slice_start: *const T,
    row_pos: usize,
    slice_rows: usize,
    slice_cols: usize,
    row_stride: isize,
    _marker: PhantomData<&'a T>,
}

/// Mutable row iterator.
#[derive(Debug)]
pub struct RowsMut<'a, T: 'a> {
    slice_start: *mut T,
    row_pos: usize,
    slice_rows: usize,
    slice_cols: usize,
    row_stride: isize,
    _marker: PhantomData<&'a mut T>,
}

/// Column of a matrix.
///
/// This struct points to a `MatrixSlice`
/// making up a column in a matrix.
/// You can deref this struct to retrieve
/// the raw column `MatrixSlice`.
///
/// # Example
///
/// ```
/// # #[macro_use] extern crate rulinalg; fn main() {
/// use rulinalg::matrix::BaseMatrix;
///
/// let mat = matrix![1.0, 2.0;
///                   3.0, 4.0];
///
/// let col = mat.col(1);
/// assert_eq!((*col + 2.0).sum(), 10.0);
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Column<'a, T: 'a> {
    col: MatrixSlice<'a, T>,
}

/// Mutable column of a matrix.
///
/// This struct points to a `MatrixSliceMut`
/// making up a column in a matrix.
/// You can deref this struct to retrieve
/// the raw column `MatrixSliceMut`.
///
/// # Example
///
/// ```
/// # #[macro_use] extern crate rulinalg; fn main() {
/// use rulinalg::matrix::BaseMatrixMut;
///
/// let mut mat = matrix![1.0, 2.0;
///                   3.0, 4.0];
/// {
///     let mut column = mat.col_mut(1);
///     *column += 2.0;
/// }
/// let expected = matrix![1.0, 4.0;
///                        3.0, 6.0];
/// assert_matrix_eq!(mat, expected);
/// # }
/// ```
#[derive(Debug)]
pub struct ColumnMut<'a, T: 'a> {
    col: MatrixSliceMut<'a, T>,
}

/// Diagonal offset (used by Diagonal iterator).
#[derive(Debug, PartialEq)]
pub enum DiagOffset {
    /// The main diagonal of the matrix.
    Main,
    /// An offset above the main diagonal.
    Above(usize),
    /// An offset below the main diagonal.
    Below(usize),
}

/// An iterator over the diagonal elements of a matrix.
#[derive(Debug)]
pub struct Diagonal<'a, T: 'a, M: 'a + BaseMatrix<T>> {
    matrix: &'a M,
    diag_pos: usize,
    diag_end: usize,
    _marker: PhantomData<&'a T>,
}

/// An iterator over the mutable diagonal elements of a matrix.
#[derive(Debug)]
pub struct DiagonalMut<'a, T: 'a, M: 'a + BaseMatrixMut<T>> {
    matrix: &'a mut M,
    diag_pos: usize,
    diag_end: usize,
    _marker: PhantomData<&'a mut T>,
}

/// Iterator for matrix.
///
/// Iterates over the underlying slice data
/// in row-major order.
#[derive(Debug)]
pub struct SliceIter<'a, T: 'a> {
    slice_start: *const T,
    row_pos: usize,
    col_pos: usize,
    slice_rows: usize,
    slice_cols: usize,
    row_stride: usize,
    _marker: PhantomData<&'a T>,
}

/// Iterator for mutable matrix.
///
/// Iterates over the underlying slice data
/// in row-major order.
#[derive(Debug)]
pub struct SliceIterMut<'a, T: 'a> {
    slice_start: *mut T,
    row_pos: usize,
    col_pos: usize,
    slice_rows: usize,
    slice_cols: usize,
    row_stride: usize,
    _marker: PhantomData<&'a mut T>,
}

/// Solves the system Ux = y by back substitution.
///
/// Here U is an upper triangular matrix and y a vector
/// which is dimensionally compatible with U.
fn back_substitution<T, M>(u: &M, y: Vector<T>) -> Result<Vector<T>, Error>
    where T: Any + Float,
          M: BaseMatrix<T>
{
    assert!(u.rows() == u.cols(), "Matrix U must be square.");
    assert!(y.size() == u.rows(),
        "Matrix and RHS vector must be dimensionally compatible.");
    let mut x = y;

    let n = u.rows();
    for i in (0 .. n).rev() {
        let row = u.row(i);

        // TODO: Remove unsafe once `get` is available in `BaseMatrix`
        let divisor = unsafe { u.get_unchecked([i, i]).clone() };
        if divisor.abs() < T::epsilon() {
            return Err(Error::new(ErrorKind::DivByZero,
                "Lower triangular matrix is singular to working precision."));
        }

        // We have
        // u[i, i] x[i] = b[i] - sum_j { u[i, j] * x[j] }
        // where j = i + 1, ..., (n - 1)
        //
        // Note that the right-hand side sum term can be rewritten as
        // u[i, (i + 1) .. n] * x[(i + 1) .. n]
        // where * denotes the dot product.
        // This is handy, because we have a very efficient
        // dot(., .) implementation!
        let dot = {
            let row_part = &row.raw_slice()[(i + 1) .. n];
            let x_part = &x.data()[(i + 1) .. n];
            utils::dot(row_part, x_part)
        };

        x[i] = (x[i] - dot) / divisor;
    }

    Ok(x)
}

/// Solves the system Lx = y by forward substitution.
///
/// Here, L is a square, lower triangular matrix and y
/// is a vector which is dimensionally compatible with L.
fn forward_substitution<T, M>(l: &M, y: Vector<T>) -> Result<Vector<T>, Error>
    where T: Any + Float,
          M: BaseMatrix<T>
{
    assert!(l.rows() == l.cols(), "Matrix L must be square.");
    assert!(y.size() == l.rows(),
        "Matrix and RHS vector must be dimensionally compatible.");
    let mut x = y;

    for (i, row) in l.row_iter().enumerate() {
        // TODO: Remove unsafe once `get` is available in `BaseMatrix`
        let divisor = unsafe { l.get_unchecked([i, i]).clone() };
        if divisor.abs() < T::epsilon() {
            return Err(Error::new(ErrorKind::DivByZero,
                "Lower triangular matrix is singular to working precision."));
        }

        // We have
        // l[i, i] x[i] = b[i] - sum_j { l[i, j] * x[j] }
        // where j = 0, ..., i - 1
        //
        // Note that the right-hand side sum term can be rewritten as
        // l[i, 0 .. i] * x[0 .. i]
        // where * denotes the dot product.
        // This is handy, because we have a very efficient
        // dot(., .) implementation!
        let dot = {
            let row_part = &row.raw_slice()[0 .. i];
            let x_part = &x.data()[0 .. i];
            utils::dot(row_part, x_part)
        };

        x[i] = (x[i] - dot) / divisor;
    }
    Ok(x)
}

/// Computes the parity of a permutation matrix.
fn parity<T, M>(m: &M) -> T
    where T: Any + Float,
          M: BaseMatrix<T>
{
    let mut visited = vec![false; m.rows()];
    let mut sgn = T::one();

    for k in 0..m.rows() {
        if !visited[k] {
            let mut next = k;
            let mut len = 0;

            while !visited[next] {
                len += 1;
                visited[next] = true;
                unsafe {
                    next = utils::find(&m.row_unchecked(next)
                                           .raw_slice(),
                                       T::one());
                }
            }

            if len % 2 == 0 {
                sgn = -sgn;
            }
        }
    }
    sgn
}
