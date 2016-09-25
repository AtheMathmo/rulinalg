//! The COO sparse matrix module.

use libnum::{One, Zero};

use sparse_matrix::compressed_matrix::CompressedMatrix;
use sparse_matrix::compressed_matrix::csc_matrix::CscMatrix;
use sparse_matrix::compressed_matrix::csr_matrix::CsrMatrix;
use sparse_matrix::SparseMatrix;

/// The `CsrMatrix` struct.
///
/// Can be instantiated with any type.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct CooMatrix<T> {
    rows: usize,
    cols: usize,
    rows_indices: Vec<usize>,
    cols_indices: Vec<usize>,
    nnz: usize,
    values: Vec<T>,
}

impl<T> CooMatrix<T> {
    /// Constructor for CooMatrix struct.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::SparseMatrix;
    /// use rulinalg::sparse_matrix::coo_matrix::CooMatrix;
    ///
    /// let coo_mat = CooMatrix::new(4, 4, vec![0, 1, 2], vec![0, 1, 2], vec![1, 2, 3]);
    ///
    /// assert_eq!(coo_mat.get_rows(), 4);
    /// assert_eq!(coo_mat.get_cols(), 4);
    /// ```
    ///
    /// # Panics
    ///
    /// Rows indices, columns indices and values indices does not have the same length
    pub fn new(rows: usize,
               cols: usize,
               rows_indices: Vec<usize>,
               cols_indices: Vec<usize>,
               values: Vec<T>)
               -> CooMatrix<T> {
        assert!(rows_indices.len() == cols_indices.len() && cols_indices.len() == values.len(),
                "Indices, pointers and values must have the same length");

        let nnz = rows_indices.len();

        CooMatrix {
            rows: rows,
            cols: cols,
            rows_indices: rows_indices,
            cols_indices: cols_indices,
            nnz: nnz,
            values: values,
        }
    }
}

impl<T: Copy + One + Zero> SparseMatrix<T> for CooMatrix<T> {
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::SparseMatrix;
    /// use rulinalg::sparse_matrix::coo_matrix::CooMatrix;
    ///
    /// let coo_mat = CooMatrix::from_diag(&[1.0,2.0,3.0,4.0]);
    /// ```
    fn from_diag(diag: &[T]) -> CooMatrix<T> {
        let size = diag.len();

        CooMatrix {
            rows: size,
            cols: size,
            rows_indices: (0..size).collect::<Vec<usize>>(),
            cols_indices: (0..size).collect::<Vec<usize>>(),
            nnz: size,
            values: diag.to_vec(),
        }
    }
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::SparseMatrix;
    /// use rulinalg::sparse_matrix::coo_matrix::CooMatrix;
    ///
    /// let I = CooMatrix::<f64>::identity(4);
    /// ```
    fn identity(size: usize) -> CooMatrix<T> {
        CooMatrix {
            rows: size,
            cols: size,
            rows_indices: (0..size).collect::<Vec<usize>>(),
            cols_indices: (0..size).collect::<Vec<usize>>(),
            nnz: size,
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
    /// use rulinalg::sparse_matrix::coo_matrix::CooMatrix;
    ///
    /// let coo_mat_t = CooMatrix::new(4, 4, vec![0, 0, 0], vec![0, 1, 2], vec![1, 2, 3]).transpose();
    /// ```
    ///
    /// Complexity: O(1)
    fn transpose(&self) -> CooMatrix<T>
        where T: Copy
    {
        CooMatrix {
            rows: self.rows,
            cols: self.cols,
            rows_indices: self.cols_indices.clone(),
            cols_indices: self.rows_indices.clone(),
            nnz: self.nnz,
            values: self.values.clone(),
        }
    }

    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::SparseMatrix;
    /// use rulinalg::sparse_matrix::coo_matrix::CooMatrix;
    ///
    /// let coo_mat = CooMatrix::new(4, 4, vec![0, 0, 0], vec![0, 1, 2], vec![1, 2, 3]).to_coo();
    /// ```
    fn to_coo(&self) -> CooMatrix<T> {
        self.clone()
    }
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::SparseMatrix;
    /// use rulinalg::sparse_matrix::coo_matrix::CooMatrix;
    ///
    /// let coo_mat = CooMatrix::new(4, 4, vec![0, 0, 0], vec![0, 1, 2], vec![1, 2, 3]).to_csc();
    /// ```
    fn to_csc(&self) -> CscMatrix<T> {
        let mut indices = vec![0; self.nnz];
        let mut last_ptr = 0;
        let mut ptrs = vec![0; self.cols + 1];
        let mut sum_ptrs = 0;
        let mut values = vec![T::zero(); self.nnz];

        for row_idcs_idx in 0..self.nnz {
            ptrs[self.cols_indices[row_idcs_idx]] += 1;
        }

        for ptr_idx in 0..self.cols {
            let tmp_ptr = ptrs[ptr_idx];
            ptrs[ptr_idx] = sum_ptrs;
            sum_ptrs += tmp_ptr;
        }

        ptrs[self.rows] = self.nnz;

        for row_idcs_idx in 0..self.nnz {
            let ptr_idx = self.cols_indices[row_idcs_idx];
            let dest_idx = ptrs[ptr_idx];

            indices[dest_idx] = self.rows_indices[row_idcs_idx];
            values[dest_idx] = self.values[row_idcs_idx];

            ptrs[ptr_idx] += 1;
        }

        for ptr_idx in 0..self.nnz {
            let tmp_ptr = ptrs[ptr_idx];
            ptrs[ptr_idx] = last_ptr;
            last_ptr = tmp_ptr;
        }

        CscMatrix::new(self.rows, self.cols, indices, ptrs, values)
    }
    /// # Examples
    ///
    /// ```
    /// use rulinalg::sparse_matrix::SparseMatrix;
    /// use rulinalg::sparse_matrix::coo_matrix::CooMatrix;
    ///
    /// let coo_mat = CooMatrix::new(4, 4, vec![0, 0, 0], vec![0, 1, 2], vec![1, 2, 3]).to_csr();
    /// ```
    fn to_csr(&self) -> CsrMatrix<T> {
        let mut indices = vec![0; self.nnz];
        let mut last_ptr = 0;
        let mut ptrs = vec![0; self.rows + 1];
        let mut sum_ptrs = 0;
        let mut values = vec![T::zero(); self.nnz];

        for row_idcs_idx in 0..self.nnz {
            ptrs[self.rows_indices[row_idcs_idx]] += 1;
        }

        for ptr_idx in 0..self.rows {
            let tmp_ptr = ptrs[ptr_idx];
            ptrs[ptr_idx] = sum_ptrs;
            sum_ptrs += tmp_ptr;
        }

        ptrs[self.rows] = self.nnz;

        for row_idcs_idx in 0..self.nnz {
            let ptr_idx = self.rows_indices[row_idcs_idx];
            let dest_idx = ptrs[ptr_idx];

            indices[dest_idx] = self.cols_indices[row_idcs_idx];
            values[dest_idx] = self.values[row_idcs_idx];

            ptrs[ptr_idx] += 1;
        }

        for ptr_idx in 0..self.nnz {
            let tmp_ptr = ptrs[ptr_idx];
            ptrs[ptr_idx] = last_ptr;
            last_ptr = tmp_ptr;
        }

        CsrMatrix::new(self.rows, self.cols, indices, ptrs, values)
    }
}

impl<T: Clone> Clone for CooMatrix<T> {
    fn clone(&self) -> CooMatrix<T> {
        CooMatrix {
            rows: self.rows,
            cols: self.cols,
            rows_indices: self.rows_indices.clone(),
            cols_indices: self.cols_indices.clone(),
            nnz: self.nnz,
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
        let a = CooMatrix::new(3, 3, vec![0, 1, 1, 2], vec![0, 1, 2, 2], vec![1, 2, 3, 4]);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    #[should_panic]
    fn test_new_mat_bad_data() {
        let _ = CooMatrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1, 2, 3]);
    }
    #[test]
    fn test_from_diag() {
        let a = CooMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 2], vec![1, 2, 3]);
        let b = CooMatrix::from_diag(&[1, 2, 3]);
        assert_eq!(a, b);
    }
    #[test]
    fn test_identity() {
        let a = CooMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 2], vec![1, 1, 1]);
        let b = CooMatrix::identity(3);
        assert_eq!(a, b);
    }

    #[test]
    fn test_transpose() {
        let a = CooMatrix::new(3, 3, vec![0, 1, 1, 2], vec![0, 1, 2, 2], vec![1, 2, 3, 4])
            .transpose();
        let b = CooMatrix::new(3, 3, vec![0, 1, 2, 2], vec![0, 1, 1, 2], vec![1, 2, 3, 4]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_to_coo() {
        let a = CooMatrix::new(3, 3, vec![0, 1, 1, 2], vec![0, 1, 2, 2], vec![1, 2, 3, 4]);
        let b = a.to_coo();
        assert_eq!(a, b);
    }
    #[test]
    fn test_to_csc() {
        let a = CooMatrix::new(3, 3, vec![0, 1, 1, 2], vec![0, 1, 2, 2], vec![1, 2, 3, 4]).to_csc();
        let b = CscMatrix::new(3, 3, vec![0, 1, 1, 2], vec![0, 1, 2, 4], vec![1, 2, 3, 4]);
        assert_eq!(a, b);
    }
    #[test]
    fn test_to_csr() {
        let a = CooMatrix::new(3, 3, vec![0, 1, 1, 2], vec![0, 1, 2, 2], vec![1, 2, 3, 4]).to_csr();
        let b = CsrMatrix::new(3, 3, vec![0, 1, 2, 2], vec![0, 1, 3, 4], vec![1, 2, 3, 4]);
        assert_eq!(a, b);
    }
}
