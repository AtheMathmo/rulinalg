//! The CSC (Compressed Sparse Column) matrix module.
//!
//! ---
//!
//! Creating new CSC matrices:
//!
//! ```
//! use rulinalg::sparse_matrix::{CompressedMatrix, CscMatrix};
//!
//! // A new matrix with 3 rows and 2 columns.
//! let _ = CscMatrix::new(3, 2, vec![1, 2, 3], vec![0, 1, 0], vec![0, 2, 3]);
//! ```
//!
//! The matrices are stored in column-major order. This means in the example above the top
//! row will be [1,2,0].
//!
//! Performing arithmetic operations:
//!
//! ```
//! use rulinalg::sparse_matrix::{CompressedMatrix, CscMatrix};
//!
//! // A new matrix with 3 rows and 2 columns.
//! let a = CscMatrix::new(5, 5, vec![2, 2, 2, 2, 2, 2], vec![0, 0, 1, 1, 4, 4], vec![0, 2, 4, 4, 6, 6]);
//!
//! // Produces a 5x5 CSC matrix with some sevens.
//! let _ = a + 5;
//! ```

pub mod csc_matrix_arithm;

use libnum::{One, Zero};

use sparse_matrix::compressed_matrix::{Compressed, CompressedLinear, CompressedLinearMut};
use sparse_matrix::{CompressedMatrix, CsrMatrix, SparseMatrix, Triplet};

/// The `CscMatrix` struct.
///
/// Can be instantiated with any type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CscMatrix<T> {
    compressed: Compressed<T>,
}

impl<T: Copy + One + Zero> CscMatrix<T> {
    /// Creates a new CSR matrix based on CSC matrix underlying data.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::{CompressedMatrix, CscMatrix};
    ///
    /// let csc_mat = CscMatrix::new(4, 4, vec![1, 2, 3], vec![0, 0, 0], vec![0, 1, 2, 3, 3]).to_csr();
    /// ```
    pub fn to_csr(&self) -> CsrMatrix<T> {
        CsrMatrix::new(self.compressed.rows(),
                       self.compressed.cols(),
                       self.compressed.data().to_vec(),
                       self.compressed.indices().to_vec(),
                       self.compressed.ptrs().to_vec())
    }
}

impl<T: Copy + One + Zero> CompressedMatrix<T> for CscMatrix<T> {
    fn from_triplets<R>(rows: usize, cols: usize, triplets: &[R]) -> CscMatrix<T>
        where R: Triplet<T>
    {
        CscMatrix { compressed: Compressed::from_triplets(rows, cols, triplets) }
    }
    fn new(rows: usize,
           cols: usize,
           data: Vec<T>,
           indices: Vec<usize>,
           ptrs: Vec<usize>)
           -> CscMatrix<T> {
        CscMatrix { compressed: Compressed::new(cols, rows, data, indices, ptrs) }
    }

    fn indices(&self) -> &[usize] {
        self.compressed.indices()
    }
    fn ptrs(&self) -> &[usize] {
        self.compressed.ptrs()
    }

    fn iter_linear(&self) -> CompressedLinear<T> {
        self.compressed.iter_linear()
    }
    fn iter_linear_mut(&mut self) -> CompressedLinearMut<T> {
        self.compressed.iter_linear_mut()
    }
}

impl<T: Copy + One + Zero> SparseMatrix<T> for CscMatrix<T> {
    fn from_diag(diag: &[T]) -> CscMatrix<T> {
        CscMatrix { compressed: Compressed::from_diag(diag) }
    }
    fn identity(size: usize) -> CscMatrix<T> {
        CscMatrix { compressed: Compressed::identity(size) }
    }

    fn cols(&self) -> usize {
        self.compressed.rows()
    }
    fn nnz(&self) -> usize {
        self.compressed.nnz()
    }
    fn rows(&self) -> usize {
        self.compressed.cols()
    }

    fn get(&self, row: usize, col: usize) -> T {
        self.compressed.get(col, row)
    }
    fn transpose(&mut self) {
        self.compressed.transpose()
    }

    fn data(&self) -> &[T] {
        self.compressed.data()
    }
    fn mut_data(&mut self) -> &mut [T] {
        self.compressed.mut_data()
    }
    fn into_vec(self) -> Vec<T> {
        self.compressed.into_vec()
    }
}

#[cfg(test)]
mod tests {
    use sparse_matrix::{CompressedMatrix, CscMatrix, CsrMatrix, SparseMatrix};

    #[test]
    fn test_equality() {
        let a = CscMatrix::new(3, 3, vec![1, 2, 3, 4], vec![0, 1, 1, 2], vec![0, 1, 2, 2]);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn test_non_equality() {
        let a = CscMatrix::new(3, 3, vec![1, 2, 3, 4], vec![0, 1, 1, 2], vec![0, 1, 2, 2]);
        let b = CscMatrix::new(3, 3, vec![1, 2], vec![2, 2], vec![0, 1, 2, 2]);
        assert!(a != b);
    }

    #[test]
    #[should_panic]
    fn test_new_mat_bad_data() {
        let _ = CscMatrix::new(3, 3, vec![1, 2, 3], vec![0, 1, 2, 3], vec![0, 1, 2]);
    }
    #[test]
    fn test_from_diag() {
        let a = CscMatrix::new(3, 3, vec![1, 2, 3], vec![0, 1, 2], vec![0, 1, 2, 3]);
        let b = CscMatrix::from_diag(&[1, 2, 3]);
        assert_eq!(a, b);
    }
    #[test]
    fn test_from_triplets() {
        let a = CscMatrix::from_triplets(3, 3, &[(0, 0, 1), (1, 1, 2), (2, 2, 3)]);
        let b = CscMatrix::new(3, 3, vec![1, 2, 3], vec![0, 1, 2], vec![0, 1, 2, 3]);
        assert_eq!(a, b);
    }
    #[test]
    fn test_identity() {
        let a = CscMatrix::new(3, 3, vec![1, 1, 1], vec![0, 1, 2], vec![0, 1, 2, 3]);
        let b = CscMatrix::identity(3);
        assert_eq!(a, b);
    }

    #[test]
    fn test_transpose() {
        let mut a = CscMatrix::new(3, 3, vec![1, 2, 3], vec![0, 1, 2], vec![0, 1, 3, 3]);
        a.transpose();
        let b = CscMatrix::new(3, 3, vec![1, 2, 3], vec![0, 1, 1], vec![0, 1, 2, 3]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_to_csr() {
        let a = CscMatrix::new(3, 3, vec![1, 2, 3], vec![0, 1, 2], vec![0, 1, 3, 3]).to_csr();
        let b = CsrMatrix::new(3, 3, vec![1, 2, 3], vec![0, 1, 2], vec![0, 1, 3, 3]);
        assert_eq!(a, b);
    }
}
