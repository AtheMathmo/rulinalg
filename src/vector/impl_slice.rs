use std::marker::PhantomData;

use super::{Vector, VectorSlice, VectorSliceMut};

impl<'a, T> VectorSlice<'a, T> {
    /// Produces a `VectorSlice` from a `Vector`
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::{BaseVector, VectorSlice};
    ///
    /// let vector = vector![1, 2, 3, 4, 5];
    /// let slice = VectorSlice::from_vector(&vector, 0, 2);
    /// assert_eq!(slice.data(), &[1, 2]);
    /// # }
    /// ```
    pub fn from_vector(vector: &'a Vector<T>, start: usize, size: usize) -> Self {
        assert!(start + size <= vector.size,
                "View dimension exceeds vector dimension.");

        unsafe {
            VectorSlice {
                marker: PhantomData::<&'a T>,
                ptr: vector.data.get_unchecked(start) as *const T,
                size: size,
            }
        }
    }

    /// Creates a `VectorSlice` from raw parts.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::{BaseVector, VectorSlice};
    ///
    /// let a = vector![1, 2, 3, 4, 5];
    ///
    /// unsafe {
    ///     // Create a vector slice with size 3
    ///     let b = VectorSlice::from_raw_parts(a.as_ptr(), 3);
    /// }
    /// # }
    /// ```
    ///
    /// # Safety
    ///
    /// The pointer must be followed by a contiguous slice of data larger than `size`.
    /// If not then other operations will produce undefined behaviour.
    pub unsafe fn from_raw_parts(ptr: *const T, size: usize) -> Self {
        VectorSlice {
            marker: PhantomData::<&'a T>,
            ptr: ptr,
            size: size,
        }
    }
}

impl<'a, T> VectorSliceMut<'a, T> {
    /// Produces a `VectorSliceMut` from a mutable `Vector`
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::{BaseVectorMut, VectorSliceMut};
    ///
    /// let mut vector = vector![1, 2, 3, 4, 5];
    /// let mut slice = VectorSliceMut::from_vector(&mut vector, 1, 3);
    /// assert_eq!(slice.data_mut(), &mut [2, 3, 4]);
    /// # }
    /// ```
    pub fn from_vector(vector: &'a mut Vector<T>, start: usize, size: usize) -> Self {
        assert!(start + size <= vector.size,
                "View dimension exceeds vector dimension.");

        VectorSliceMut {
            marker: PhantomData::<&'a mut T>,
            ptr: unsafe { vector.data.get_unchecked_mut(start) as *mut T },
            size: size,
        }
    }

    /// Creates a `VectorSliceMut` from raw parts.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::{BaseVectorMut, VectorSliceMut};
    ///
    /// let mut a = vector![1, 2, 3, 4, 5];
    ///
    /// unsafe {
    ///     // Create a vector slice with size 3
    ///     let b = VectorSliceMut::from_raw_parts(a.as_mut_ptr(), 3);
    /// }
    /// # }
    /// ```
    ///
    /// # Safety
    ///
    /// The pointer must be followed by a contiguous slice of data larger than `size`.
    /// If not then other operations will produce undefined behaviour.
    pub unsafe fn from_raw_parts(ptr: *mut T, size: usize) -> Self {
        VectorSliceMut {
            marker: PhantomData::<&'a mut T>,
            ptr: ptr,
            size: size,
        }
    }
}

#[cfg(test)]
mod tests {
    use vector::{BaseVector, Vector, VectorSlice, VectorSliceMut};

    #[test]
    #[should_panic]
    fn make_slice_bad_dim() {
        let a = Vector::ones(3) * 2.0;
        let _ = VectorSlice::from_vector(&a, 1, 3);
    }

    #[test]
    fn make_slice() {
        let a = Vector::ones(3) * 2.0;
        let b = VectorSlice::from_vector(&a, 1, 2);

        assert_eq!(b.size(), 2);
    }

    #[test]
    fn make_slice_mut() {
        let mut a = Vector::ones(3) * 2.0;
        {
            let mut b = VectorSliceMut::from_vector(&mut a, 1, 2);
            assert_eq!(b.size(), 2);
            b += 2.0;
        }
        println!("{:?}", a);
        let exp = vector![2.0, 4.0, 4.0];
        assert_vector_eq!(a, exp);
    }
}
