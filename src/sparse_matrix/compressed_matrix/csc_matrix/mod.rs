//! The CSC sparse matrix module.

mod csc_matrix_arithm;

use libnum::{One, Zero};

use sparse_matrix::compressed_matrix::compressed_matrix_utils::*;
use sparse_matrix::compressed_matrix::CompressedMatrix;
use sparse_matrix::compressed_matrix::csr_matrix::CsrMatrix;
use sparse_matrix::coordinate::Coordinate;
use sparse_matrix::SparseMatrix;

/// The `CscMatrix` struct.
///
/// Can be instantiated with any type.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct CscMatrix<T> {
    rows: usize,
    cols: usize,
    indices: Vec<usize>,
    nnz: usize,
    ptrs: Vec<usize>,
    values: Vec<T>,
}

impl<T: Copy + One + Zero> CompressedMatrix<T> for CscMatrix<T> {
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::SparseMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::CompressedMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::csc_matrix::CscMatrix;
    ///
    /// let csc_mat = CscMatrix::new(4, 4, vec![0, 1, 2], vec![0, 1, 2, 3], vec![1, 2, 3]);
    ///
    /// assert_eq!(csc_mat.rows(), 4);
    /// assert_eq!(csc_mat.cols(), 4);
    /// ```
    ///
    /// # Panics
    ///
    /// Indices length does not match values length
    fn new(rows: usize,
           cols: usize,
           indices: Vec<usize>,
           ptrs: Vec<usize>,
           values: Vec<T>)
           -> CscMatrix<T> {
        assert!(indices.len() == values.len(),
                "Column indices must be equal row values");

        let nnz = indices.len();

        CscMatrix {
            rows: rows,
            cols: cols,
            indices: indices,
            nnz: nnz,
            ptrs: ptrs,
            values: values,
        }
    }

    fn indices(&self) -> &[usize] {
        &self.indices.as_slice()
    }
    fn ptrs(&self) -> &[usize] {
        &self.ptrs.as_slice()
    }
    fn values(&self) -> &[T] {
        &self.values.as_slice()
    }
}

impl<T: Copy + One + Zero> SparseMatrix<T> for CscMatrix<T> {
	fn from_coordinates<C>(coords: &[C]) -> CscMatrix<T> where C: Coordinate<T> {
    	unimplemented!();
    }
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::SparseMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::CompressedMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::csc_matrix::CscMatrix;
    ///
    /// let csc_mat = CscMatrix::from_diag(&[1.0,2.0,3.0,4.0]);
    /// ```
    fn from_diag(diag: &[T]) -> CscMatrix<T> {
        let size = diag.len();

        CscMatrix {
            rows: size,
            cols: size,
            indices: (0..size).collect::<Vec<usize>>(),
            nnz: size,
            ptrs: (0..(size + 1)).collect::<Vec<usize>>(),
            values: diag.to_vec(),
        }
    }
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::SparseMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::CompressedMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::csc_matrix::CscMatrix;
    ///
    /// let I = CscMatrix::<f64>::identity(4);
    /// ```
    fn identity(size: usize) -> CscMatrix<T> {
        CscMatrix {
            rows: size,
            cols: size,
            indices: (0..size).collect::<Vec<usize>>(),
            nnz: size,
            ptrs: (0..(size + 1)).collect::<Vec<usize>>(),
            values: vec![T::one(); size],
        }
    }

    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn nnz(&self) -> usize {
        self.nnz
    }

    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::SparseMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::CompressedMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::csc_matrix::CscMatrix;
    ///
    /// let _ = CscMatrix::new(4, 4, vec![0, 0, 0], vec![0, 1, 2, 3, 3], vec![1, 2, 3]).transpose();
    /// ```
    ///
    /// Complexity: O(rows + max(cols, nnz))
    fn transpose(&self) -> CscMatrix<T>
        where T: Copy
    {
    	let (indices, ptrs, values) = transpose(self, self.rows + 1);

        CscMatrix {
            rows: self.rows,
            cols: self.cols,
            indices: indices,
            nnz: self.nnz,
            ptrs: ptrs,
            values: values,
        }
    }

    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::SparseMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::CompressedMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::csc_matrix::CscMatrix;
    ///
    /// let csc_mat = CscMatrix::new(4, 4, vec![0, 0, 0], vec![0, 1, 2, 3, 3], vec![1, 2, 3]).to_csc();
    /// ```
    fn to_csc(&self) -> CscMatrix<T> {
        self.clone()
    }
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::SparseMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::CompressedMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::csc_matrix::CscMatrix;
    ///
    /// let csr_mat = CscMatrix::new(4, 4, vec![0, 0, 0], vec![0, 1, 2, 3, 3], vec![1, 2, 3]).to_csr();
    /// ```
    fn to_csr(&self) -> CsrMatrix<T> {
        CsrMatrix::new(self.rows,
                       self.cols,
                       self.indices.clone(),
                       self.ptrs.clone(),
                       self.values.clone())
    }
}

impl<T: Clone> Clone for CscMatrix<T> {
    fn clone(&self) -> CscMatrix<T> {
        CscMatrix {
            rows: self.rows,
            cols: self.cols,
            indices: self.indices.clone(),
            nnz: self.nnz,
            ptrs: self.ptrs.clone(),
            values: self.values.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use sparse_matrix::compressed_matrix::CompressedMatrix;
    use sparse_matrix::compressed_matrix::csc_matrix::CscMatrix;
    use sparse_matrix::compressed_matrix::csr_matrix::CsrMatrix;
    use sparse_matrix::SparseMatrix;

    #[test]
    fn test_equality() {
        let a = CscMatrix::new(3, 3, vec![0, 1, 1, 2], vec![0, 1, 2, 2], vec![1, 2, 3, 4]);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    #[should_panic]
    fn test_new_mat_bad_data() {
        let _ = CscMatrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1, 2, 3]);
    }
    #[test]
    fn test_from_diag() {
        let a = CscMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 2, 3], vec![1, 2, 3]);
        let b = CscMatrix::from_diag(&[1, 2, 3]);
        assert_eq!(a, b);
    }
    #[test]
    fn test_identity() {
        let a = CscMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 2, 3], vec![1, 1, 1]);
        let b = CscMatrix::identity(3);
        assert_eq!(a, b);
    }

    #[test]
    fn test_transpose() {
        let a = CscMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 3, 3], vec![1, 2, 3]).transpose();
        let b = CscMatrix::new(3, 3, vec![0, 1, 1], vec![0, 1, 2, 3], vec![1, 2, 3]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_to_csc() {
        let a = CscMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 3, 3], vec![1, 2, 3]);
        let b = a.to_csc();
        assert_eq!(a, b);
    }
    #[test]
    fn test_to_csr() {
        let a = CscMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 3, 3], vec![1, 2, 3]).to_csr();
        let b = CsrMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 3, 3], vec![1, 2, 3]);
        assert_eq!(a, b);
    }
}
