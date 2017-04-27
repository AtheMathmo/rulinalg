//! Traits for vector operations.
//!
//! These traits defines operations for structs representing vectors.
//!
//! Implementations are provided for
//! - `Vector`: an owned vector
//! - `VectorSlice`: a borrowed immutable block of `Vector`
//! - `VectorSliceMut`: a borrowed mutable block of `Vector`
//!
//! ```
//! # #[macro_use] extern crate rulinalg; fn main() {
//! use rulinalg::vector::BaseVector;
//!
//! let a = vector![1, 2, 3, 4, 5];
//! let b = vector![6, 7];
//!
//! // Manually create our slice - [4, 5].
//! let vector_slice = a.sub_slice(3, 2);
//!
//! // We can perform arithmetic with mixing owned and borrowed versions
//! let _new_vector = &vector_slice + &b;
//! # }
//! ```

use std::ops::{Add, Mul, Div};
use std::slice;

use libnum::Zero;
use norm::{VectorNorm, VectorMetric};
use utils;

use super::{Vector, VectorSlice, VectorSliceMut};

/// Trait for immutable vector structs.
pub trait BaseVector<T>: Sized {
    /// Left index of the vector.
    fn as_ptr(&self) -> *const T;

    /// Returns a `VectorSlice` over the whole vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::BaseVector;
    ///
    /// let a = vector![1, 2, 3, 4, 5];
    /// let b = a.as_slice();
    /// # }
    /// ```
    fn as_slice(&self) -> VectorSlice<T> {
        unsafe { VectorSlice::from_raw_parts(self.as_ptr(), self.size()) }
    }

    /// Horizontally concatenates two vectors. With self on the left.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::BaseVector;
    ///
    /// let a = vector![1, 2, 3];
    /// let b = vector![4, 5];
    ///
    /// let c = &a.concat(&b);
    /// assert_eq!(c.size(), a.size() + b.size());
    /// assert_eq!(c[3], 4);
    /// # }
    /// ```
    fn concat<S>(&self, vector: &S) -> Vector<T>
        where T: Copy,
              S: BaseVector<T>
    {
        let mut new_data = Vec::with_capacity(self.size() + vector.size());

        new_data.extend_from_slice(self.data());
        new_data.extend_from_slice(vector.data());

        Vector::new(new_data)
    }

    /// Returns a slice representing the underlying data of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::{BaseVector, VectorSlice};
    ///
    /// let a = vector![1, 2, 3, 4, 5];
    ///
    /// assert_eq!(a.data(), &[1, 2, 3, 4, 5]);
    ///
    /// let b = VectorSlice::from_vector(&a, 1, 2);
    ///
    /// assert_eq!(b.data(), &[2, 3]);
    /// # }
    /// ```
    fn data(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.size()) }
    }

    /// The elementwise division of two vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::{BaseVector, Vector};
    ///
    /// let a = vector![1, 2, 3, 4, 5];
    /// let b = vector![1, 2, 3, 4, 5];
    ///
    /// let c = &a.elediv(&b);
    /// assert_eq!(c, &vector![1; 5]);
    /// # }
    /// ```
    fn elediv(&self, vector: &Self) -> Vector<T>
        where T: Copy + Div<T, Output = T>
    {
        assert!(self.size() == vector.size(), "Vector size are not equal.");
        Vector::new(utils::ele_div(&self.data(), &vector.data()))
    }

    /// The elementwise product of two vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::{BaseVector, Vector};
    ///
    /// let a = vector![1, 2, 3, 4, 5];
    /// let b = vector![1, 2, 3, 4, 5];
    ///
    /// let c = &a.elemul(&b);
    /// assert_eq!(c, &vector![1, 4, 9, 16, 25]);
    /// # }
    /// ```
    fn elemul(&self, vector: &Self) -> Vector<T>
        where T: Copy + Mul<T, Output = T>
    {
        assert_eq!(self.size(), vector.size());
        Vector::new(utils::ele_mul(&self.data(), &vector.data()))
    }

    /// Get a reference to an element in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::BaseVector;
    ///
    /// let a = vector![1, 2, 3, 4, 5];
    ///
    /// assert_eq!(a.get(5), None);
    /// assert_eq!(a.get(6), None);
    ///
    /// assert_eq!(*a.get(0).unwrap(), 1)
    /// # }
    /// ```
    fn get(&self, index: usize) -> Option<&T> {
        self.data().get(index)
    }

    /// Returns true if the vector contais no elements.
    fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Returns an iterator over the vector data.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::BaseVector;
    ///
    /// let a = vector![1, 2, 3, 4, 5];
    /// let b = a.sub_slice(1, 2);
    ///
    /// let c = b.iter().map(|v| *v).collect::<Vec<usize>>();
    /// assert_eq!(c, vec![2, 3]);
    /// # }
    /// ```
    fn iter<'a>(&self) -> slice::Iter<T>
        where T: 'a
    {
        self.data().iter()
    }

    /// Compute metric distance between two vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::{BaseVector, Vector};
    /// use rulinalg::norm::Euclidean;
    ///
    /// let a = vector![3.0, 4.0];
    /// let b = vector![0.0, 8.0];
    ///
    /// // Compute the square root of the sum of
    /// // elementwise squared-differences
    /// let c = a.metric(&b, Euclidean);
    /// assert_eq!(c, 5.0);
    /// # }
    /// ```
    fn metric<'a, 'b, B, M>(&'a self, vector: &'b B, m: M) -> T
        where B: 'b + BaseVector<T>,
              M: VectorMetric<'a, 'b, T, Self, B>
    {
        m.metric(self, vector)
    }

    /// Compute vector norm for vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::{BaseVector, Vector};
    /// use rulinalg::norm::Euclidean;
    ///
    /// let a = vector![3.0, 4.0];
    /// let c = a.norm(Euclidean);
    ///
    /// assert_eq!(c, 5.0);
    /// # }
    /// ```
    fn norm<N>(&self, norm: N) -> T
        where N: VectorNorm<T, Self>
    {
        norm.norm(self)
    }

    /// Returns a new vector based on desired indexes.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::BaseVector;
    ///
    /// let a = vector![1, 2, 3, 4, 5];
    /// let b = &a.select(&[0, 3]);
    ///
    /// assert_eq!(b.size(), 2);
    /// assert_eq!(b.data(), &[1, 4]);
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if indexes exceed the vector size.
    fn select(&self, indexes: &[usize]) -> Vector<T>
        where T: Copy
    {
        let mut values_vector = Vec::with_capacity(indexes.len());

        for index in indexes {
            assert!(*index < self.size(), "Index is greater than size of vector");

            unsafe {
                values_vector.push(*self.get_unchecked(*index));
            }
        }

        Vector::new(values_vector)
    }

    /// Size of the vector.
    fn size(&self) -> usize;

    /// Split the vector at the specified index returning two `VectorxSlice`s.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::BaseVector;
    ///
    /// let a = vector![1, 2, 3, 4, 5];
    /// let (b, c) = a.split_at(1);
    /// # }
    /// ```
    fn split_at(&self, index: usize) -> (VectorSlice<T>, VectorSlice<T>) {
        assert!(index < self.size());

        let slice_1: VectorSlice<T>;
        let slice_2: VectorSlice<T>;

        unsafe {
            slice_1 = VectorSlice::from_raw_parts(self.as_ptr(), index);
            slice_2 = VectorSlice::from_raw_parts(self.as_ptr().offset(index as isize),
                                                  self.size());
        }

        (slice_1, slice_2)
    }

    /// Produce a `VectorSlice` from an existing vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::{BaseVector, VectorSlice};
    ///
    /// let a = vector![1, 2, 3, 4, 5];
    /// let slice = VectorSlice::from_vector(&a, 1, 3);
    /// let new_slice = slice.sub_slice(0, 1);
    /// # }
    /// ```
    fn sub_slice<'a>(&self, start: usize, size: usize) -> VectorSlice<'a, T>
        where T: 'a
    {
        assert!(start + size <= self.size(),
                "View dimension exceeds vector dimensions.");

        unsafe { VectorSlice::from_raw_parts(self.as_ptr().offset(start as isize), size) }
    }

    /// The sum of the vector.
    ///
    /// Returns the sum of all elements in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// let a = vector![1, 2, 3, 4, 5];
    ///
    /// let c = a.sum();
    /// assert_eq!(c, 15);
    /// # }
    /// ```
    fn sum(&self) -> T
        where T: Copy + Zero + Add<T, Output = T>
    {
        utils::unrolled_sum(&self.data())
    }

    /// Get a reference to an element in the vector without bounds checking.
    unsafe fn get_unchecked(&self, index: usize) -> &T {
        self.data().get_unchecked(index)
    }
}

/// Trait for mutable vector structs.
pub trait BaseVectorMut<T>: BaseVector<T> {
    /// Left index of the vector.
    fn as_mut_ptr(&mut self) -> *mut T;

    /// Returns a `VectorSliceMut` over the whole vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::BaseVectorMut;
    ///
    /// let mut a = vector![1, 2, 3, 4, 5];
    /// let b = a.as_mut_slice();
    /// # }
    /// ```
    fn as_mut_slice(&mut self) -> VectorSliceMut<T> {
        unsafe { VectorSliceMut::from_raw_parts(self.as_mut_ptr(), self.size()) }
    }

    /// Returns a mutable slice representing the underlying data of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::{BaseVectorMut, VectorSliceMut};
    ///
    /// let mut a = vector![1, 2, 3, 4, 5];
    ///
    /// {
    ///     let a_data_mut = a.data_mut();
    ///     a_data_mut[0] = 0;
    ///     assert_eq!(a_data_mut, &[0, 2, 3, 4, 5]);
    /// }
    ///
    /// let mut b = VectorSliceMut::from_vector(&mut a, 1, 2);
    ///
    /// let b_data_mut = b.data_mut();
    /// b_data_mut[1] = 10;
    /// assert_eq!(b_data_mut, &[2, 10]);
    /// # }
    /// ```
    fn data_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.size()) }
    }

    /// Get a mutable reference to an element in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::{BaseVector, BaseVectorMut};
    ///
    /// let mut a = vector![1, 2, 3, 4, 5];
    ///
    /// assert_eq!(a.get(5), None);
    /// assert_eq!(a.get(6), None);
    ///
    /// assert_eq!(*a.get_mut(0).unwrap(), 1);
    /// *a.get_mut(0).unwrap() = 2;
    /// assert_eq!(*a.get_mut(0).unwrap(), 2);
    /// # }
    /// ```
    fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data_mut().get_mut(index)
    }

    /// Returns an iterator over the vector data.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::BaseVector;
    ///
    /// let a = vector![1, 2, 3, 4, 5];
    /// let b = a.sub_slice(1, 2);
    ///
    /// let c = b.iter().map(|v| *v).collect::<Vec<usize>>();
    /// assert_eq!(c, vec![2, 3]);
    /// # }
    /// ```
    fn iter_mut<'a>(&mut self) -> slice::IterMut<T>
        where T: 'a
    {
        self.data_mut().iter_mut()
    }

    /// Split the vector at the specified index returning two `VectorxSliceMut`s.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::BaseVectorMut;
    ///
    /// let mut a = vector![1, 2, 3, 4, 5];
    /// let (b, c) = a.split_at_mut(1);
    /// # }
    /// ```
    fn split_at_mut(&mut self, index: usize) -> (VectorSliceMut<T>, VectorSliceMut<T>) {
        assert!(index < self.size());

        let slice_1: VectorSliceMut<T>;
        let slice_2: VectorSliceMut<T>;

        unsafe {
            slice_1 = VectorSliceMut::from_raw_parts(self.as_mut_ptr(), index);
            slice_2 = VectorSliceMut::from_raw_parts(self.as_mut_ptr().offset(index as isize),
                                                     self.size());
        }

        (slice_1, slice_2)
    }

    /// Produce a `VectorSliceMut` from an existing vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::{BaseVectorMut, VectorSliceMut};
    ///
    /// let mut a = vector![1, 2, 3, 4, 5];
    /// let mut slice = VectorSliceMut::from_vector(&mut a, 1, 3);
    /// let new_slice = slice.sub_slice_mut(0, 1);
    /// # }
    /// ```
    fn sub_slice_mut<'a>(&mut self, start: usize, size: usize) -> VectorSliceMut<'a, T>
        where T: 'a
    {
        assert!(start + size <= self.size(),
                "View dimension exceeds vector dimensions.");

        unsafe { VectorSliceMut::from_raw_parts(self.as_mut_ptr().offset(start as isize), size) }
    }

    /// Get a mutable reference to an element in the vector without bounds checking.
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        self.data_mut().get_unchecked_mut(index)
    }
}

impl<T> BaseVector<T> for Vector<T> {
    fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl<'a, T> BaseVector<T> for VectorSlice<'a, T> {
    fn as_ptr(&self) -> *const T {
        self.ptr
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl<'a, T> BaseVector<T> for VectorSliceMut<'a, T> {
    fn as_ptr(&self) -> *const T {
        self.ptr
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl<T> BaseVectorMut<T> for Vector<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
}

impl<'a, T> BaseVectorMut<T> for VectorSliceMut<'a, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}
