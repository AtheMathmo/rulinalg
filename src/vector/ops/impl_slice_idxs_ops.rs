use std::ops::{Index, IndexMut};

use vector::{VectorSlice, VectorSliceMut, BaseVector, BaseVectorMut};

impl<'a, T> IndexMut<usize> for VectorSliceMut<'a, T> {
    fn index_mut(&mut self, idx: usize) -> &mut T {
        assert!(idx < self.size);
        unsafe { self.get_unchecked_mut(idx) }
    }
}

macro_rules! impl_index_slice (
    ($slice_type:ident, $doc:expr) => (
#[doc=$doc]
impl<'a, T> Index<usize> for $slice_type<'a, T> {
    type Output = T;
    fn index(&self, idx: usize) -> &T {
        assert!(idx < self.size);
        unsafe { &*(self.get_unchecked(idx)) }
    }
}
    );
);
impl_index_slice!(VectorSlice, "vector slice.");
impl_index_slice!(VectorSliceMut, "mutable vector slice.");
