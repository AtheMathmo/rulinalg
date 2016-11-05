//! The Compressed sparse matrix module
//!
//! Used as a common interface for Csc and Csr matrices implementations.

pub mod compressed;
pub mod csc_matrix;
pub mod csr_matrix;

pub use self::compressed::{Compressed, CompressedLinear, CompressedLinearMut};
use sparse_matrix::{Triplet, SparseMatrix};

/// Contract for compressed matrices implementation
pub trait CompressedMatrix<T>: SparseMatrix<T> {
    /// Constructs matrix with given coordinates (rows, cols and values).
    ///
    /// Requires slice of coordinates.
    fn from_triplets<R>(rows: usize, cols: usize, triplets: &[R]) -> Self where R: Triplet<T>;
    /// Construct a new matrix based only in rows and cols lengh
    fn new(rows: usize, cols: usize, data: Vec<T>, indices: Vec<usize>, ptrs: Vec<usize>) -> Self;

    /// Returns indices
    fn indices(&self) -> &[usize];
    /// Returns pointers (offsets)
    fn ptrs(&self) -> &[usize];

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
    /// for (col_rows_indices, col_data) in a.iter_linear() {
    ///     println!("{:?}, {:?}", col_rows_indices, col_data);
    /// }
    ///
    /// ```
    /// use rulinalg::sparse_matrix::{CompressedMatrix, CsrMatrix};
    ///
    /// let a = CsrMatrix::new(3,3, vec![1, 2, 3], vec![0, 0, 1], vec![0, 2, 3, 3]);
    ///
    /// // Iterates linearly over CSR matrix, i.e, over rows
    /// // Prints [0, 0], [1, 2] - [1], [3] - [], []
    /// for (row_cols_indices, row_data) in a.iter_linear() {
    ///     println!("{:?}, {:?}", row_cols_indices, row_data);
    /// }
    /// ```
    fn iter_linear(&self) -> CompressedLinear<T>;
    /// Iterates linearly over of the matrix returning its mutable data and respective column indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::{CompressedMatrix, CscMatrix};
    ///
    /// let mut a = CscMatrix::new(3,3, vec![1, 2, 3], vec![0, 0, 1], vec![0, 2, 3, 3]);
    ///
    /// // Iterates linearly over CSC matrix, i.e, over columns.
    /// // Prints [0, 0], [1, 2] - [1], [3] - [], []
    /// for (_, col_data) in a.iter_linear_mut() {
	///		for data in col_data {
    ///     	*data = *data * 2;
    ///		}
    /// }
    ///
    /// ```
    /// use rulinalg::sparse_matrix::{CompressedMatrix, CsrMatrix};
    ///
    /// let mut a = CsrMatrix::new(3,3, vec![1, 2, 3], vec![0, 0, 1], vec![0, 2, 3, 3]);
    ///
    /// // Iterates linearly over CSR matrix, i.e, over rows
    /// // Prints [0,0], [1,2] - [1], [3] - [], []
    /// for (_, row_data) in a.iter_linear_mut() {
	///		for data in row_data {
    ///     	*data = *data * 2;
    ///		}
    /// }
    /// ```
    fn iter_linear_mut(&mut self) -> CompressedLinearMut<T>;
}
