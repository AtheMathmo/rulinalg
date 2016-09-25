//! The Sparse matrix module
//!
//! Used as a common interface for all sparse matrices implementations

pub mod compressed_matrix;
pub mod coo_matrix;

use sparse_matrix::compressed_matrix::csc_matrix::CscMatrix;
use sparse_matrix::compressed_matrix::csr_matrix::CsrMatrix;
use sparse_matrix::coo_matrix::CooMatrix;

/// Contract for sparse matrices implementation
pub trait SparseMatrix<T> {
    /// Constructs matrix with given diagonal.
    ///
    /// Requires slice of diagonal elements.
    fn from_diag(diag: &[T]) -> Self;
    /// Constructs the identity matrix.
    ///
    /// Requires the size of the matrix.
    fn identity(size: usize) -> Self;

    /// Returns number of rows
    fn get_rows(&self) -> usize;
    /// Returns number of cols
    fn get_cols(&self) -> usize;
    /// Returns number of non zero elements
    fn get_nnz(&self) -> usize;

    /// Tranposes the given matrix
    fn transpose(&self) -> Self;

    /// Creates a new CooMatrix
    fn to_coo(&self) -> CooMatrix<T>;
    /// Creates a new CscMatrix
    fn to_csc(&self) -> CscMatrix<T>;
    /// Creates a new CsrMatrix
    fn to_csr(&self) -> CsrMatrix<T>;
}
