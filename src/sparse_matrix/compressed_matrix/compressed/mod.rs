//! The Compressed sparse matrix module (Row-major order).
//!
//! Since CSC and CSR matrices shares much of the same source code, this module is intended
//! as a central destination for logical operations.
//!
//! How the delegation of responsibility works:
//! CSR matrix: Delegates all methods with no modifications.
//! CSC matrix: Delegates all methods swaping rows for columns.

mod compressed_utils;
mod compressed_iter_linear;

use std::marker::PhantomData;
use std::mem;

use libnum::{One, Zero};

use self::compressed_utils::*;
use sparse_matrix::{Triplet, CompressedMatrix, SparseMatrix};

/// The `Compressed` struct.
///
/// Can be instantiated with any type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Compressed<T> {
    cols: usize,
    data: Vec<T>,
    indices: Vec<usize>,
    ptrs: Vec<usize>,
    rows: usize,
}

/// Compressed matrix linear iterator
#[derive(Debug)]
pub struct CompressedLinear<'a, T: 'a> {
    _marker: PhantomData<&'a T>,
    current_pos: usize,
    data: *const T,
    indices: &'a [usize],
    positions: usize,
    ptrs: &'a [usize],
}

/// Compressed matrix mutable linear iterator
#[derive(Debug)]
pub struct CompressedLinearMut<'a, T: 'a> {
    _marker: PhantomData<&'a mut T>,
    current_pos: usize,
    data: *mut T,
    indices: &'a [usize],
    positions: usize,
    ptrs: &'a [usize],
}

impl<T: Copy + One + Zero> CompressedMatrix<T> for Compressed<T> {
    fn from_triplets<R>(rows: usize, cols: usize, triplets: &[R]) -> Compressed<T>
        where R: Triplet<T>
    {
        let nnz = triplets.len();

        let mut data = vec![T::zero(); nnz];
        let mut indices = vec![0; nnz];
        let mut last_ptr = 0;
        let mut ptrs = vec![0; rows + 1];
        let mut sum_ptrs = 0;

        for idcs_idx in 0..nnz {
            ptrs[triplets[idcs_idx].row()] += 1;
        }

        for ptr_idx in 0..rows {
            let tmp_ptr = ptrs[ptr_idx];
            ptrs[ptr_idx] = sum_ptrs;
            sum_ptrs += tmp_ptr;
        }

        ptrs[rows] = nnz;

        for idcs_idx in 0..nnz {
            let ptr_idx = triplets[idcs_idx].row();
            let dest_idx = ptrs[ptr_idx];

            data[dest_idx] = triplets[idcs_idx].value();
            indices[dest_idx] = triplets[idcs_idx].col();

            ptrs[ptr_idx] += 1;
        }

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

    fn indices(&self) -> &[usize] {
        &self.indices
    }
    fn ptrs(&self) -> &[usize] {
        &self.ptrs
    }

    fn iter_linear(&self) -> CompressedLinear<T> {
        CompressedLinear {
            _marker: PhantomData::<&T>,
            current_pos: 0,
            data: self.data.as_ptr(),
            indices: &self.indices,
            positions: self.rows,
            ptrs: &self.ptrs,
        }
    }
    fn iter_linear_mut(&mut self) -> CompressedLinearMut<T> {
        CompressedLinearMut {
            _marker: PhantomData::<&mut T>,
            current_pos: 0,
            data: self.data.as_mut_ptr(),
            indices: &self.indices,
            positions: self.rows,
            ptrs: &self.ptrs,
        }
    }
}

impl<T: Copy + One + Zero> SparseMatrix<T> for Compressed<T> {
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
    fn identity(size: usize) -> Compressed<T> {
        Compressed {
            cols: size,
            data: vec![T::one(); size],
            indices: (0..size).collect::<Vec<usize>>(),
            ptrs: (0..(size + 1)).collect::<Vec<usize>>(),
            rows: size,
        }
    }

    fn cols(&self) -> usize {
        self.cols
    }
    fn nnz(&self) -> usize {
        self.data.len()
    }
    fn rows(&self) -> usize {
        self.rows
    }

    fn get(&self, row: usize, col: usize) -> T {
        assert!(row < self.rows() && col < self.cols());

        let mut result_data = T::zero();

        for idx in self.ptrs[row]..self.ptrs[row + 1] {
            if self.indices[idx] == col {
                result_data = self.data[idx];
                break;
            }
        }

        return result_data;
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
        expand_ptrs(&self.ptrs, |ptr_idx, dest_idx| {
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

    fn data(&self) -> &[T] {
        &self.data
    }
    fn mut_data(&mut self) -> &mut [T] {
        &mut self.data
    }
    fn into_vec(self) -> Vec<T> {
        self.data
    }
}
