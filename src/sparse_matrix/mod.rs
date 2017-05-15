//! The Sparse matrix module
//!
//! Used as a common interface for all sparse matrices implementations.
//!
//! References:
//! 1. [Performance comparison of storage formats for sparse matrices]
//! (http://facta.junis.ni.ac.rs/mai/mai24/fumi-24_39_51.pdf), Ivan P. Stanimirović and
//! Milan B. Tasić

pub mod compressed_matrix;
pub mod triplet;

pub use self::compressed_matrix::{CompressedMatrix, CompressedIter, CompressedIterMut};
pub use self::compressed_matrix::csc_matrix::CscMatrix;
pub use self::compressed_matrix::csr_matrix::CsrMatrix;
pub use self::triplet::Triplet;

/// Contract for sparse matrices implementation
pub trait SparseMatrix<T> {
    /// Returns number of cols
    fn cols(&self) -> usize;

    /// Returns the specified data
    fn get(&self, row: usize, col: usize) -> T;

    /// Constructs matrix with given diagonal.
    ///
    /// Requires slice of diagonal elements.
    fn from_diag(diag: &[T]) -> Self;

    /// Constructs the identity matrix.
    ///
    /// Requires the size of the matrix.
    fn identity(size: usize) -> Self;

    /// Returns number of non zero elements
    fn nnz(&self) -> usize;

    /// Returns number of rows
    fn rows(&self) -> usize;

    /// Tranposes the given matrix
    fn transpose(&mut self);
}
