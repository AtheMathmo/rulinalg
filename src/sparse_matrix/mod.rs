//! The Sparse matrix module
//!
//! Used as a common interface for all sparse matrices implementations
//!
//! References:
//! 1. [Performance comparison of storage formats for sparse matrices]
//! (http://facta.junis.ni.ac.rs/mai/mai24/fumi-24_39_51.pdf), Ivan P. Stanimirović and Milan B. Tasić

pub mod compressed_matrix;
pub mod triplet;

use sparse_matrix::triplet::Triplet;
use sparse_matrix::compressed_matrix::csc_matrix::CscMatrix;
use sparse_matrix::compressed_matrix::csr_matrix::CsrMatrix;

/// Contract for sparse matrices implementation
pub trait SparseMatrix<T> {
    /// Constructs matrix with given diagonal.
    ///
    /// Requires slice of diagonal elements.
    fn from_diag(diag: &[T]) -> Self;
    /// Constructs matrix with given coordinates (rows, cols and values).
    ///
    /// Requires slice of coordinates.
    fn from_triplets<R>(triplets: &[R]) -> Self where R: Triplet<T>;
    /// Constructs the identity matrix.
    ///
    /// Requires the size of the matrix.
    fn identity(size: usize) -> Self;

    /// Returns number of rows
    fn rows(&self) -> usize;
    /// Returns number of cols
    fn cols(&self) -> usize;
    /// Returns number of non zero elements
    fn nnz(&self) -> usize;

    /// Tranposes the given matrix
    fn transpose(&self) -> Self;

    /// Creates a new CscMatrix
    fn to_csc(&self) -> CscMatrix<T>;
    /// Creates a new CsrMatrix
    fn to_csr(&self) -> CsrMatrix<T>;
}
