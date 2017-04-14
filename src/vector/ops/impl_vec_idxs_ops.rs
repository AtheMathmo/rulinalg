use std::ops::{Index, IndexMut};

use vector::{BaseVector, BaseVectorMut, Vector};

impl<T> Index<usize> for Vector<T> {
    type Output = T;

    fn index(&self, idx: usize) -> &T {
        assert!(idx < self.size);
        unsafe { self.get_unchecked(idx) }
    }
}

impl<T> IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, idx: usize) -> &mut T {
        assert!(idx < self.size);
        unsafe { self.get_unchecked_mut(idx) }
    }
}

#[cfg(test)]
mod tests {
    use vector::Vector;
    
    #[test]
    fn vector_index_mut() {
        let our_vec = vec![1., 2., 3., 4.];
        let mut our_vector = Vector::new(our_vec.clone());

        for i in 0..4 {
            our_vector[i] += 1.;
        }

        assert_eq!(our_vector, vector![2., 3., 4., 5.]);
    }
}