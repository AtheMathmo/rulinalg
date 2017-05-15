//! The Compressed sparse matrix module
//!
//! Used as a common interface for Csc and Csr matrices implementations.

mod compressed;
pub mod csc_matrix;
pub mod csr_matrix;

use std::marker::PhantomData;

use sparse_matrix::{Triplet, SparseMatrix};

/// Contract for compressed matrices implementation
pub trait CompressedMatrix<T>: SparseMatrix<T> {
    /// Returns a non-mutable reference to the underlying data.
    fn data(&self) -> &[T];

    /// Constructs matrix with given coordinates (rows, cols and values).
    ///
    /// Requires slice of coordinates.
    fn from_triplets<R>(rows: usize, cols: usize, triplets: &[R]) -> Self where R: Triplet<T>;

    /// Returns indices
    fn indices(&self) -> &[usize];

    /// Consumes the Matrix and returns the Vec of data.
    fn into_vec(self) -> Vec<T>;

    /// Iterates linearly over the matrix returning its data and respective indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::{CompressedMatrix, CscMatrix};
    ///
    /// let a = CscMatrix::new(3,3, vec![1, 2, 3], vec![0, 0, 1], vec![0, 2, 3, 3]);
    ///
    /// // Iterates linearly over CSC matrix, i.e, over columns.
    /// // Prints [0,0], [1,2] - [1], [3], [], []
    /// for (col_rows_indices, col_data) in a.iter() {
    ///     println!("{:?}, {:?}", col_rows_indices, col_data);
    /// }
    ///
    /// ```
    /// use rulinalg::sparse_matrix::{CompressedMatrix, CsrMatrix};
    ///
    /// let a = CsrMatrix::new(3,3, vec![1, 2, 3], vec![0, 0, 1], vec![0, 2, 3, 3]);
    ///
    /// // Iterates over CSR matrix, i.e, over rows
    /// // Prints [0, 0], [1, 2] - [1], [3] - [], []
    /// for (row_cols_indices, row_data) in a.iter() {
    ///     println!("{:?}, {:?}", row_cols_indices, row_data);
    /// }
    /// ```
    fn iter(&self) -> CompressedIter<T>;

    /// Iterates over of the matrix returning its mutable data and respective column indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::{CompressedMatrix, CscMatrix};
    ///
    /// let mut a = CscMatrix::new(3,3, vec![1, 2, 3], vec![0, 0, 1], vec![0, 2, 3, 3]);
    ///
    /// // Iterates over CSC matrix, i.e, over columns.
    /// // Prints [0, 0], [1, 2] - [1], [3] - [], []
    /// for (_, col_data) in a.iter_mut() {
    /// 		for data in col_data {
    ///     	*data = *data * 2;
    /// 		}
    /// }
    ///
    /// ```
    /// use rulinalg::sparse_matrix::{CompressedMatrix, CsrMatrix};
    ///
    /// let mut a = CsrMatrix::new(3,3, vec![1, 2, 3], vec![0, 0, 1], vec![0, 2, 3, 3]);
    ///
    /// // Iterates linearly over CSR matrix, i.e, over rows
    /// // Prints [0,0], [1,2] - [1], [3] - [], []
    /// for (_, row_data) in a.iter_mut() {
    /// 		for data in row_data {
    ///     	*data = *data * 2;
    /// 		}
    /// }
    /// ```
    fn iter_mut(&mut self) -> CompressedIterMut<T>;

    /// Returns a mutable slice of the underlying data.
    fn mut_data(&mut self) -> &mut [T];

    /// Construct a new matrix based only in rows and cols lengh
    fn new(rows: usize, cols: usize, data: Vec<T>, indices: Vec<usize>, ptrs: Vec<usize>) -> Self;

    /// Returns pointers (offsets)
    fn ptrs(&self) -> &[usize];
}

/// The `Compressed` struct.
///
/// Can be instantiated with any type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Compressed<T> {
    cols: usize,
    data: Vec<T>,
    indices: Vec<usize>,
    ptrs: Vec<usize>,
    rows: usize,
}

/// Compressed matrix linear iterator
#[derive(Debug)]
pub struct CompressedIter<'a, T: 'a> {
    _marker: PhantomData<&'a T>,
    current_pos: usize,
    data: *const T,
    indices: &'a [usize],
    positions: usize,
    ptrs: &'a [usize],
}

/// Compressed matrix mutable linear iterator
#[derive(Debug)]
pub struct CompressedIterMut<'a, T: 'a> {
    _marker: PhantomData<&'a mut T>,
    current_pos: usize,
    data: *mut T,
    indices: &'a [usize],
    positions: usize,
    ptrs: &'a [usize],
}
