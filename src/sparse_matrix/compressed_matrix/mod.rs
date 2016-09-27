//! The Compressed sparse matrix module
//!
//! Used as a common interface for Csc and Csr matrices implementations

mod compressed_matrix_utils;
pub mod csc_matrix;
pub mod csr_matrix;

use sparse_matrix::SparseMatrix;
use sparse_matrix::triplet::Triplet;

/// Contract for compressed matrices implementation
pub trait CompressedMatrix<T>: SparseMatrix<T> {
    /// Constructs matrix with given coordinates (rows, cols and values).
    ///
    /// Requires slice of coordinates.
    fn from_triplets<R>(rows: usize, cols: usize, triplets: &[R]) -> Self where R: Triplet<T>;
    /// Construct a new matrix based only in rows and cols lengh
    fn new(rows: usize,
           cols: usize,
           indices: Vec<usize>,
           ptrs: Vec<usize>,
           values: Vec<T>)
           -> Self;

    /// Returns indices
    fn indices(&self) -> &[usize];
    /// Returns pointers (offsets)
    fn ptrs(&self) -> &[usize];
    /// Returns values
    fn values(&self) -> &[T];
}
