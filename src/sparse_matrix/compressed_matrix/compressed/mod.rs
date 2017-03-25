//! The Compressed sparse matrix module (Row-major order).
//!
//! Since CSC and CSR matrices shares much of the same source code, this module is intended
//! as a central destination for logical operations.
//!
//! How the delegation of responsibility works:
//! CSR matrix: Delegates all methods with no modifications.
//! CSC matrix: Delegates all methods swaping rows for columns.

mod compressed_iter;
mod compressed_utils;

use std::marker::PhantomData;
use std::mem;

use libnum::{One, Zero};

use self::compressed_utils::*;
use sparse_matrix::{CompressedIter, CompressedIterMut, CompressedMatrix, SparseMatrix};
use sparse_matrix::compressed_matrix::Compressed;

impl<T: Copy + One + Zero> CompressedMatrix<T> for Compressed<T> {
    fn data(&self) -> &[T] {
        &self.data
    }

    fn from_triplets(rows: usize, cols: usize, triplets: &[(usize, usize, T)]) -> Compressed<T> {
        let summed_triplets = sum_triplets(triplets);

        let nnz = summed_triplets.len();

        let mut data = vec![T::zero(); nnz];
        let mut indices = vec![0; nnz];
        let mut last_ptr = 0;
        let mut ptrs = vec![0; rows + 1];
        let mut sum_ptrs = 0;

        // Fill ptrs with each row nnz
        for idcs_idx in 0..nnz {
            ptrs[summed_triplets[idcs_idx].0] += 1;
        }

        // Incremental addition of rows nnz
        for ptr_idx in 0..rows {
            let tmp_ptr = ptrs[ptr_idx];
            ptrs[ptr_idx] = sum_ptrs;
            sum_ptrs += tmp_ptr;
        }

        ptrs[rows] = nnz;

        // Fill indices, data and adjust ptrs
        for idcs_idx in 0..nnz {
            let ptr_idx = summed_triplets[idcs_idx].0;
            let dest_idx = ptrs[ptr_idx];

            data[dest_idx] = summed_triplets[idcs_idx].2;
            indices[dest_idx] = summed_triplets[idcs_idx].1;

            ptrs[ptr_idx] += 1;
        }

        // Correct ptrs order
        for ptr_idx in 0..nnz {
            let tmp_ptr = ptrs[ptr_idx];
            ptrs[ptr_idx] = last_ptr;
            last_ptr = tmp_ptr;
        }

        Compressed {
            cols: cols,
            data: data,
            indices: indices,
            ptrs: ptrs,
            rows: rows,
        }
    }


    fn indices(&self) -> &[usize] {
        &self.indices
    }

    fn into_vec(self) -> Vec<T> {
        self.data
    }

    fn iter(&self) -> CompressedIter<T> {
        CompressedIter {
            _marker: PhantomData::<&T>,
            current_pos: 0,
            data: self.data.as_ptr(),
            indices: &self.indices,
            positions: self.rows,
            ptrs: &self.ptrs,
        }
    }

    fn iter_mut(&mut self) -> CompressedIterMut<T> {
        CompressedIterMut {
            _marker: PhantomData::<&mut T>,
            current_pos: 0,
            data: self.data.as_mut_ptr(),
            indices: &self.indices,
            positions: self.rows,
            ptrs: &self.ptrs,
        }
    }

    fn mut_data(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn new(rows: usize,
           cols: usize,
           data: Vec<T>,
           indices: Vec<usize>,
           ptrs: Vec<usize>)
           -> Compressed<T> {
        assert!(ptrs.len() == rows + 1, "Invalid pointers length");
        assert!(data.len() == indices.len(),
                "Data length must be equal indices length");

        Compressed {
            cols: cols,
            data: data,
            indices: indices,
            ptrs: ptrs,
            rows: rows,
        }
    }

    fn ptrs(&self) -> &[usize] {
        &self.ptrs
    }
}

impl<T: Copy + One + Zero> SparseMatrix<T> for Compressed<T> {
    fn cols(&self) -> usize {
        self.cols
    }

    fn identity(size: usize) -> Compressed<T> {
        Compressed {
            cols: size,
            data: vec![T::one(); size],
            indices: (0..size).collect::<Vec<usize>>(),
            ptrs: (0..(size + 1)).collect::<Vec<usize>>(),
            rows: size,
        }
    }

    fn from_diag(diag: &[T]) -> Compressed<T> {
        let size = diag.len();

        Compressed {
            cols: size,
            data: diag.to_vec(),
            indices: (0..size).collect::<Vec<usize>>(),
            ptrs: (0..(size + 1)).collect::<Vec<usize>>(),
            rows: size,
        }
    }

    fn get(&self, row: usize, col: usize) -> T {
        assert!(row < self.rows() && col < self.cols());
        
        self.indices[self.ptrs[row]..self.ptrs[row + 1]]
            .binary_search(&col)
            .map(|index| self.data[self.ptrs[row] + index])
            .unwrap_or(T::zero())
    }

    fn nnz(&self) -> usize {
        self.data.len()
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn transpose(&mut self) {
        mem::swap(&mut self.rows, &mut self.cols);

        let nnz = self.nnz();
        let mut counter = vec![0; nnz];
        let mut indices = vec![0; nnz];
        let mut ptrs = vec![0; self.ptrs.len()];
        let mut data = vec![T::zero(); nnz];

        for idx in &self.indices {
            counter[*idx] += 1;
        }

        ptrs[0] = 0;

        for idx in 0..self.rows {
            ptrs[idx + 1] = ptrs[idx] + counter[idx];
        }

        expand_ptrs_rev(&self.ptrs, |ptr_idx, dest_idx| {
            let old_col_idx = self.indices[dest_idx];

            counter[old_col_idx] -= 1;

            let nnz_idx = ptrs[old_col_idx] + counter[old_col_idx];

            indices[nnz_idx] = ptr_idx;
            data[nnz_idx] = self.data[dest_idx];
        });

        self.data = data;
        self.indices = indices;
        self.ptrs = ptrs;
    }
}
