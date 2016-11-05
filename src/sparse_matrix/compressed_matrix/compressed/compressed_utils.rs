/// Given a slice of pointers, returns a closure that indicates its overlapped "index" and current
/// expanded "value".
/// If, for example, this function was used with the pointers of a `CSCMatrix`, the "index" would
/// represent a row index and the "value" a index used to locate data and column index.
///
/// # Ilustrations
///
/// 0 , 3 , 6 , 8 , 8
/// |___|___|___|___|
///   |   |   |   |
///   0   1   2   3
///   |   |   |
///   |   |   |-> (2, 6) - (2, 7)
///   |   |-> (1, 3) - (1, 4) - (1, 5)
///   |-> (0, 0) - (0, 1) - (0, 2)
///
/// All values are reverted, which gives us the following sequence: (2, 7) - (2, 6) - (1, 5) -
/// (1, 4) - (1, 3) - (0, 2) - (0, 1) - (0, 0).
pub fn expand_ptrs_rev<C>(ptrs: &[usize], mut closure: C)
    where C: FnMut(usize, usize)
{
    for (idx, range) in ptrs.windows(2).enumerate().rev() {
        for nnz_idx in (range[0]..range[1]).rev() {
            closure(idx, nnz_idx)
        }
    }
}
