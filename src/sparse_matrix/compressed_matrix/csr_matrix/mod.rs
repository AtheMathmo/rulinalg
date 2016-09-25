//! The CSR sparse matrix module.

mod csr_matrix_arithm;

use libnum::{One, Zero};

use sparse_matrix::compressed_matrix::compressed_matrix_utils::*;
use sparse_matrix::compressed_matrix::CompressedMatrix;
use sparse_matrix::compressed_matrix::csc_matrix::CscMatrix;
use sparse_matrix::coordinate::Coordinate;
use sparse_matrix::SparseMatrix;

/// The `CsrMatrix` struct.
///
/// Can be instantiated with any type.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct CsrMatrix<T> {
    rows: usize,
    cols: usize,
    indices: Vec<usize>,
    nnz: usize,
    ptrs: Vec<usize>,
    values: Vec<T>,
}

impl<T: Copy + One + Zero> CompressedMatrix<T> for CsrMatrix<T> {
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::SparseMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::CompressedMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::csr_matrix::CsrMatrix;
    ///
    /// let csr_mat = CsrMatrix::new(4, 4, vec![0, 1, 2], vec![0, 1, 2, 3], vec![1, 2, 3]);
    ///
    /// assert_eq!(csr_mat.get_rows(), 4);
    /// assert_eq!(csr_mat.get_cols(), 4);
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

        let nnz = indices.len();

        CsrMatrix {
            rows: rows,
            cols: cols,
            indices: indices,
            nnz: nnz,
            ptrs: ptrs,
            values: values,
        }
    }

	fn get_indices(&self) -> &Vec<usize> {
		&self.indices
	}
	fn get_ptrs(&self) -> &Vec<usize> {
		&self.ptrs
	}
	fn get_values(&self) -> &Vec<T> {
		&self.values
	}
    fn set_indices(&mut self, indices: &Vec<usize>) {
    	if indices.len() == self.nnz {
	    	self.indices = indices.clone();
    	}
    }
    fn set_ptrs(&mut self, ptrs: &Vec<usize>) {
    	if ptrs.len() == (self.rows + 1) {
	    	self.ptrs = ptrs.clone();
    	}
    }
    fn set_values(&mut self, values: &Vec<T>) {
    	if values.len() == self.nnz {
	    	self.values = values.clone();
    	}
    }
}

impl<T: Copy + One + Zero> SparseMatrix<T> for CsrMatrix<T> {
    fn from_coordinates<C>(coords: &[C]) -> CsrMatrix<T> where C: Coordinate<T> {
    	unimplemented!();
    }
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::SparseMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::CompressedMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::csr_matrix::CsrMatrix;
    ///
    /// let csr_mat = CsrMatrix::from_diag(&[1.0,2.0,3.0,4.0]);
    /// ```
    fn from_diag(diag: &[T]) -> CsrMatrix<T> {
        let size = diag.len();

        CsrMatrix {
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
    /// use rulinalg::sparse_matrix::compressed_matrix::csr_matrix::CsrMatrix;
    ///
    /// let I = CsrMatrix::<f64>::identity(4);
    /// ```
    fn identity(size: usize) -> CsrMatrix<T> {
        CsrMatrix {
            rows: size,
            cols: size,
            indices: (0..size).collect::<Vec<usize>>(),
            nnz: size,
            ptrs: (0..(size + 1)).collect::<Vec<usize>>(),
            values: vec![T::one(); size]
        }
    }

	fn get_rows(&self) -> usize {
		self.rows
	}
	fn get_cols(&self) -> usize {
		self.cols
	}
	fn get_nnz(&self) -> usize {
		self.nnz
	}

    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::SparseMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::CompressedMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::csr_matrix::CsrMatrix;
    ///
    /// let _ = CsrMatrix::new(4, 4, vec![0, 0, 0], vec![0, 1, 2, 3, 3], vec![1, 2, 3]).transpose();
    /// ```
    ///
    /// Complexity: O(cols + max(rows, nnz))
    fn transpose(&self) -> CsrMatrix<T>
        where T: Copy
    {
	    let mut csr_matrix = self.clone();
    	transpose(&mut csr_matrix, self.cols + 1);
        csr_matrix
    }

    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::SparseMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::CompressedMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::csr_matrix::CsrMatrix;
    ///
    /// let csc_mat = CsrMatrix::new(4, 4, vec![0, 0, 0], vec![0, 1, 2, 3, 3], vec![1, 2, 3]).to_csc();
    /// ```
    fn to_csc(&self) -> CscMatrix<T> {
    	CscMatrix::new(self.rows, self.cols, self.indices.clone(), self.ptrs.clone(), self.values.clone())
    }
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::SparseMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::CompressedMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::csr_matrix::CsrMatrix;
    ///
    /// let csr_mat = CsrMatrix::new(4, 4, vec![0, 0, 0], vec![0, 1, 2, 3, 3], vec![1, 2, 3]).to_csr();
    /// ```
    fn to_csr(&self) -> CsrMatrix<T> {
    	self.clone()
    }
}

impl<T: Clone> Clone for CsrMatrix<T> {
    fn clone(&self) -> CsrMatrix<T> {
        CsrMatrix {
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
        let a = CsrMatrix::new(3, 3, vec![0, 1, 1, 2], vec![0, 1, 2, 2], vec![1, 2, 3, 4]);
        let b = a.clone();
        assert_eq!(a, b);
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
    fn test_identity() {
        let a = CsrMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 2, 3], vec![1, 1, 1]);
        let b = CsrMatrix::identity(3);
        assert_eq!(a, b);
    }

    #[test]
    fn test_transpose() {
        let a = CsrMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 3, 3], vec![1, 2, 3]).transpose();
        let b = CsrMatrix::new(3, 3, vec![0, 1, 1], vec![0, 1, 2, 3], vec![1, 2, 3]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_to_csc() {
        let a = CsrMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 3, 3], vec![1, 2, 3]).to_csc();
        let b = CscMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 3, 3], vec![1, 2, 3]);
        assert_eq!(a, b);
    }
    #[test]
    fn test_to_csr() {
        let a = CsrMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 3, 3], vec![1, 2, 3]);
        let b = a.to_csr();
        assert_eq!(a, b);
    }
}