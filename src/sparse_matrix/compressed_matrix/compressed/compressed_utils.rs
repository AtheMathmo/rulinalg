pub fn expand_ptrs<C>(ptrs: &[usize], mut closure: C)
    where C: FnMut(usize, usize)
{
    for (idx, range) in ptrs.windows(2).enumerate().rev() {
        for nnz_idx in (range[0]..range[1]).rev() {
            closure(idx, nnz_idx)
        }
    }
}
