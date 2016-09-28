//! The CSR sparse matrix module.

mod csr_matrix_arithm;

use std::mem::swap;

use libnum::{One, Zero};

use sparse_matrix::{CompressedMatrix, CscMatrix, MatrixCoordinate, SparseMatrix, Triplet};
use sparse_matrix::compressed_matrix::compressed_matrix_utils::*;

/// The `CsrMatrix` struct.
///
/// Can be instantiated with any type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CsrMatrix<T> {
    rows: usize,
    cols: usize,
    indices: Vec<usize>,
    ptrs: Vec<usize>,
    values: Vec<T>,
}

impl<T: Copy + One + Zero> CsrMatrix<T> {
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::{CompressedMatrix, CsrMatrix};
    ///
    /// let csc_mat = CsrMatrix::new(4, 4, vec![0, 0, 0], vec![0, 1, 2, 3, 3], vec![1, 2, 3]).to_csc();
    /// ```
    pub fn to_csc(&self) -> CscMatrix<T> {
        CscMatrix::new(self.rows,
                       self.cols,
                       self.indices.clone(),
                       self.ptrs.clone(),
                       self.values.clone())
    }

	/// Transposes the given CSR matrix returning a new CSC matrix, which leads to a free transformation.
	///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::{CompressedMatrix, CsrMatrix, SparseMatrix};
    ///
    /// let _ = CsrMatrix::new(4, 4, vec![0, 0, 0], vec![0, 1, 2, 3, 3], vec![1, 2, 3]).transpose();
    /// ```
    ///
    /// Complexity: O(1)
    pub fn transpose_csc(&self) -> CscMatrix<T>
        where T: Copy
    {
        CscMatrix::new(self.rows,
                       self.cols,
                       self.indices.clone(),
                       self.ptrs.clone(),
                       self.values.clone())
    }
}

impl<T: Copy + One + Zero> CompressedMatrix<T> for CsrMatrix<T> {
    fn from_triplets<R>(rows: usize, cols: usize, triplets: &[R]) -> CsrMatrix<T> where R: Triplet<T> {
    	let (indices, ptrs, values) = from_triplets(rows, MatrixCoordinate::ROW, MatrixCoordinate::COL, triplets);

        CsrMatrix {
            rows: rows,
            cols: cols,
            indices: indices,
            ptrs: ptrs,
            values: values,
        }
    }
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::{CompressedMatrix, CsrMatrix, SparseMatrix};
    ///
    /// let csr_mat = CsrMatrix::new(4, 4, vec![0, 1, 2], vec![0, 1, 2, 3], vec![1, 2, 3]);
    ///
    /// assert_eq!(csr_mat.rows(), 4);
    /// assert_eq!(csr_mat.cols(), 4);
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
               -> CsrMatrix<T> {
        assert!(indices.len() == values.len(),
                "Column indices must be equal row values");

        CsrMatrix {
            rows: rows,
            cols: cols,
            indices: indices,
            ptrs: ptrs,
            values: values,
        }
    }

	fn indices(&self) -> &[usize]{
		&self.indices.as_slice()
	}
	fn ptrs(&self) -> &[usize] {
		&self.ptrs.as_slice()
	}
	fn values(&self) -> &[T] {
		&self.values.as_slice()
	}
}

impl<T: Copy + One + Zero> SparseMatrix<T> for CsrMatrix<T> {
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::{CompressedMatrix, CsrMatrix, SparseMatrix};
    ///
    /// let csr_mat = CsrMatrix::from_diag(&[1.0,2.0,3.0,4.0]);
    /// ```
    fn from_diag(diag: &[T]) -> CsrMatrix<T> {
        let size = diag.len();

        CsrMatrix {
            rows: size,
            cols: size,
            indices: (0..size).collect::<Vec<usize>>(),
            ptrs: (0..(size + 1)).collect::<Vec<usize>>(),
            values: diag.to_vec(),
        }
    }
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::{CompressedMatrix, CsrMatrix, SparseMatrix};
    ///
    /// let I = CsrMatrix::<f64>::identity(4);
    /// ```
    fn identity(size: usize) -> CsrMatrix<T> {
        CsrMatrix {
            rows: size,
            cols: size,
            indices: (0..size).collect::<Vec<usize>>(),
            ptrs: (0..(size + 1)).collect::<Vec<usize>>(),
            values: vec![T::one(); size]
        }
    }

	fn rows(&self) -> usize {
		self.rows
	}
	fn cols(&self) -> usize {
		self.cols
	}
	fn nnz(&self) -> usize {
		self.values.len()
	}

    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::{CompressedMatrix, CsrMatrix, SparseMatrix};
    ///
    /// let _ = CsrMatrix::new(4, 4, vec![0, 0, 0], vec![0, 1, 2, 3, 3], vec![1, 2, 3]).transpose();
    /// ```
    ///
    /// Complexity: O(cols + max(rows, nnz))
    fn transpose(&mut self)
    {
    	swap(&mut self.rows, &mut self.cols);

    	let (indices, ptrs, values) = transpose(self, self.cols + 1);

		self.indices = indices;
		self.ptrs = ptrs;
		self.values = values;
    }
}

#[cfg(test)]
mod tests {
    use sparse_matrix::{CompressedMatrix, CscMatrix, CsrMatrix, SparseMatrix};

    #[test]
    fn test_equality() {
        let a = CsrMatrix::new(3, 3, vec![0, 1, 1, 2], vec![0, 1, 2, 2], vec![1, 2, 3, 4]);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn test_non_equality() {
        let a = CsrMatrix::new(3, 3, vec![0, 1, 1, 2], vec![0, 1, 2, 2], vec![1, 2, 3, 4]);
        let b = CsrMatrix::new(3, 3, vec![2, 2], vec![0, 1, 2, 2], vec![1, 2]);
        assert_ne!(a, b);
    }

    #[test]
    #[should_panic]
    fn test_new_mat_bad_data() {
        let _ = CsrMatrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1, 2, 3]);
    }
    #[test]
    fn test_from_diag() {
        let a = CsrMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 2, 3], vec![1, 2, 3]);
        let b = CsrMatrix::from_diag(&[1, 2, 3]);
        assert_eq!(a, b);
    }
    #[test]
    fn test_from_triplets() {
        let a = CsrMatrix::from_triplets(3, 3, &[(0, 0, 1), (1, 1, 2), (2, 2, 3)]);
        let b = CsrMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 2, 3], vec![1, 2, 3]);
        assert_eq!(a, b);
    }
    #[test]
    fn test_identity() {
        let a = CsrMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 2, 3], vec![1, 1, 1]);
        let b = CsrMatrix::identity(3);
        assert_eq!(a, b);
    }

    #[test]
    fn test_transpose() {
        let mut a = CsrMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 3, 3], vec![1, 2, 3]);
        a.transpose();
        let b = CsrMatrix::new(3, 3, vec![0, 1, 1], vec![0, 1, 2, 3], vec![1, 2, 3]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_to_csc() {
        let a = CsrMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 3, 3], vec![1, 2, 3]).to_csc();
        let b = CscMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 3, 3], vec![1, 2, 3]);
        assert_eq!(a, b);
    }
}