//! The CSC sparse matrix module.

mod csc_matrix_arithm;

use libnum::{One, Zero};

use sparse_matrix::compressed_matrix::compressed_matrix_utils::*;
use sparse_matrix::compressed_matrix::CompressedMatrix;
use sparse_matrix::compressed_matrix::csr_matrix::CsrMatrix;
use sparse_matrix::coo_matrix::CooMatrix;
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
    /// assert_eq!(csc_mat.get_rows(), 4);
    /// assert_eq!(csc_mat.get_cols(), 4);
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
        if ptrs.len() == (self.cols + 1) {
            self.ptrs = ptrs.clone();
        }
    }
    fn set_values(&mut self, values: &Vec<T>) {
        if values.len() == self.nnz {
            self.values = values.clone();
        }
    }
}

impl<T: Copy + One + Zero> SparseMatrix<T> for CscMatrix<T> {
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
    /// use rulinalg::sparse_matrix::compressed_matrix::csc_matrix::CscMatrix;
    ///
    /// let _ = CscMatrix::new(4, 4, vec![0, 0, 0], vec![0, 1, 2, 3, 3], vec![1, 2, 3]).transpose();
    /// ```
    ///
    /// Complexity: O(rows + max(cols, nnz))
    fn transpose(&self) -> CscMatrix<T>
        where T: Copy
    {
        let mut csc_matrix = self.clone();
        transpose(&mut csc_matrix, self.rows + 1);
        csc_matrix
    }

    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::SparseMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::CompressedMatrix;
    /// use rulinalg::sparse_matrix::compressed_matrix::csc_matrix::CscMatrix;
    ///
    /// let coo_mat = CscMatrix::new(4, 4, vec![0, 0, 0], vec![0, 1, 2, 3, 3], vec![1, 2, 3]).to_coo();
    /// ```
    fn to_coo(&self) -> CooMatrix<T> {
        let cols_indices = get_expansed_ptrs_indices(&self.ptrs, self.nnz);

        CooMatrix::new(self.rows,
                       self.cols,
                       self.indices.clone(),
                       cols_indices,
                       self.values.clone())
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
    use sparse_matrix::coo_matrix::CooMatrix;
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
    fn test_to_coo() {
        let a = CscMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 3, 3], vec![1, 2, 3]).to_coo();
        let b = CooMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 1], vec![1, 2, 3]);
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
