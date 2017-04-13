//! The vector module.
//!
//! Currently contains all code
//! relating to the vector linear algebra struct.

mod base;
mod impl_iter;
mod impl_slice;
mod impl_vec;
mod ops;

pub use self::base::{BaseVector, BaseVectorMut};
use std::marker::PhantomData;

/// The Vector struct.
///
/// Can be instantiated with any type.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Vector<T> {
    size: usize,
    data: Vec<T>,
}

/// A `VectorSlice`
///
/// This struct provides a slice into a vector.
///
/// The struct contains the left point of the slice
/// and the width of the slice.
#[derive(Debug, Clone, Copy)]
pub struct VectorSlice<'a, T: 'a> {
    marker: PhantomData<&'a T>,
    ptr: *const T,
    size: usize,
}

/// A mutable `MatrixSliceMut`
///
/// This struct provides a mutable slice into a vector.
///
/// The struct contains the left point of the slice
/// and the width of the slice.
#[derive(Debug)]
pub struct VectorSliceMut<'a, T: 'a> {
    marker: PhantomData<&'a mut T>,
    ptr: *mut T,
    size: usize,
}
