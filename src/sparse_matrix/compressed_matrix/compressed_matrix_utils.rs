use libnum::{One, Zero};

use sparse_matrix::compressed_matrix::CompressedMatrix;

fn expand_ptrs_indices_values<C>(ptrs: &[usize], mut closure: C)
    where C: FnMut(usize, usize)
{
    for idx in (0..(ptrs.len() - 1)).rev() {
        let greater_ptr = ptrs[idx + 1];
        let lesser_ptr = ptrs[idx];
        for value in (lesser_ptr..greater_ptr).rev() {
            closure(idx, value)
        }
    }
}

pub fn transpose<M, T>(compressed_matrix: &M, ptrs_size: usize) -> (Vec<usize>, Vec<usize>, Vec<T>)
    where M: CompressedMatrix<T>,
          T: Copy + One + Zero
{
    let indices = compressed_matrix.indices();
    let nnz = compressed_matrix.nnz();
    let ptrs = compressed_matrix.ptrs();
    let reduced_ptrs_size = ptrs_size - 1;
    let values = compressed_matrix.values();
    let mut counter = vec![0; reduced_ptrs_size];
    let mut new_indices = vec![0; nnz];
    let mut new_ptrs = vec![0; ptrs_size];
    let mut new_values = vec![T::zero(); nnz];

    for idx in indices {
        counter[*idx] += 1;
    }

    new_ptrs[0] = 0;

    for idx in 0..reduced_ptrs_size {
        new_ptrs[idx + 1] = new_ptrs[idx] + counter[idx];
    }

    expand_ptrs_indices_values(&ptrs, |ptr_idx, dest_idx| {
        let old_col_idx = indices[dest_idx];

        counter[old_col_idx] -= 1;

        let nnz_idx = new_ptrs[old_col_idx] + counter[old_col_idx];

        new_indices[nnz_idx] = ptr_idx;
        new_values[nnz_idx] = values[dest_idx];
    });

    (new_indices, new_ptrs, new_values)
}