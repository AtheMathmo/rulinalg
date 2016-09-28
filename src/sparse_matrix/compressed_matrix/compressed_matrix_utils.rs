use libnum::{One, Zero};

use sparse_matrix::{CompressedMatrix, MatrixCoordinate, Triplet};

fn expand_ptrs_indices_values<C>(ptrs: &[usize], mut closure: C)
    where C: FnMut(usize, usize)
{
    for (idx, ptr) in ptrs.windows(2).enumerate().rev() {
        for value in (ptr[0]..ptr[1]).rev() {
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

pub fn from_triplets<T, R>(coo_len: usize,
                           row_coo: MatrixCoordinate,
                           col_coo: MatrixCoordinate,
                           triplets: &[R])
                           -> (Vec<usize>, Vec<usize>, Vec<T>)
    where T: Copy + One + Zero,
          R: Triplet<T>
{
    let nnz = triplets.len();

    let mut indices = vec![0; nnz];
    let mut last_ptr = 0;
    let mut ptrs = vec![0; coo_len + 1];
    let mut sum_ptrs = 0;
    let mut values = vec![T::zero(); nnz];

    for idcs_idx in 0..nnz {
        ptrs[triplets[idcs_idx].from_coordinate(row_coo)] += 1;
    }

    for ptr_idx in 0..coo_len {
        let tmp_ptr = ptrs[ptr_idx];
        ptrs[ptr_idx] = sum_ptrs;
        sum_ptrs += tmp_ptr;
    }

    ptrs[coo_len] = nnz;

    for idcs_idx in 0..nnz {
        let ptr_idx = triplets[idcs_idx].from_coordinate(row_coo);
        let dest_idx = ptrs[ptr_idx];

        indices[dest_idx] = triplets[idcs_idx].from_coordinate(col_coo);
        values[dest_idx] = triplets[idcs_idx].value();

        ptrs[ptr_idx] += 1;
    }

    for ptr_idx in 0..nnz {
        let tmp_ptr = ptrs[ptr_idx];
        ptrs[ptr_idx] = last_ptr;
        last_ptr = tmp_ptr;
    }

    (indices, ptrs, values)
}
