//! The Compressed sparse matrix module
//!
//! Used as a common interface for Csc and Csr matrices implementations

mod compressed_matrix_utils;
pub mod csc_matrix;
pub mod csr_matrix;

use sparse_matrix::SparseMatrix;

/// Contract for compressed matrices implementation
pub trait CompressedMatrix<T>: SparseMatrix<T> {
    /// Construct a new matrix based only in rows and cols lengh
    fn new(rows: usize,
           cols: usize,
           indices: Vec<usize>,
           ptrs: Vec<usize>,
           values: Vec<T>)
           -> Self;

    /// Returns indices
    fn get_indices(&self) -> &Vec<usize>;
    /// Returns pointers (offsets)
    fn get_ptrs(&self) -> &Vec<usize>;
    /// Returns values
    fn get_values(&self) -> &Vec<T>;
    /// Sets indices
    fn set_indices(&mut self, indices: &Vec<usize>);
    /// Sets pointers (offsets)
    fn set_ptrs(&mut self, ptrs: &Vec<usize>);
    /// Sets values
    fn set_values(&mut self, values: &Vec<T>);
}
