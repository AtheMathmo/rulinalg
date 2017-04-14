use std::ops::{Index, IndexMut};

use vector::{VectorSlice, VectorSliceMut, BaseVector, BaseVectorMut};

impl<'a, T> IndexMut<usize> for VectorSliceMut<'a, T> {
    fn index_mut(&mut self, idx: usize) -> &mut T {
        assert!(idx < self.size);
        unsafe { self.get_unchecked_mut(idx) }
    }
}

macro_rules! impl_index (
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
impl_index!(VectorSlice, "vector slice.");
impl_index!(VectorSliceMut, "mutable vector slice.");

#[cfg(test)]
mod tests {
    use vector::{BaseVector, Vector, VectorSliceMut};

    #[test]
    fn vector_slice_index_mut() {
        let our_vec = vec![1., 2., 3., 4.];
        let mut our_vector = Vector::new(our_vec.clone());
        let mut our_slice = VectorSliceMut::from_vector(&mut our_vector, 0, 4);

        for i in 0..4 {
            our_slice[i] += 1.;
        }

        assert_eq!(our_slice.data(), &[2., 3., 4., 5.]);
    }
}