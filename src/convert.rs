//! The convert module.
//!
//! Contains implementations of `std::convert::From`
//! for Matrix and Vector types.

use std::convert::From;

use libnum::Zero;

use super::matrix::{DiagOffset, Matrix, MatrixSlice, MatrixSliceMut, BaseMatrix};
use super::vector::Vector;

impl<T> From<Vec<T>> for Vector<T> {
    fn from(vec: Vec<T>) -> Self {
        Vector::new(vec)
    }
}

impl<'a, T> From<&'a [T]> for Vector<T> where T: Clone {
    fn from(slice: &'a [T]) -> Self {
        Vector::new(slice.to_owned())
    }
}

impl<T> From<Vector<T>> for Matrix<T> {
    fn from(vector: Vector<T>) -> Self {
        Matrix::new(vector.size(), 1, vector.into_vec())
    }
}

macro_rules! impl_matrix_from {
    ($slice_type:ident) => {
        impl<'a, T: Copy> From<$slice_type<'a, T>> for Matrix<T> {
            fn from(slice: $slice_type<'a, T>) -> Self {
                slice.iter_rows().collect::<Matrix<T>>()
            }
        }
    }
}

impl_matrix_from!(MatrixSlice);
impl_matrix_from!(MatrixSliceMut);


macro_rules! impl_diag_offset_from {
    ($unsigned_type:ty, $signed_type:ty) => {
impl From<$unsigned_type> for DiagOffset {
    fn from(int: $unsigned_type) -> Self {
        if int.is_zero() {
            DiagOffset::Main
        } else {
            DiagOffset::Above(int as usize)
        }
    }
}

impl From<$signed_type> for DiagOffset {
    fn from(int: $signed_type) -> Self {
        if int.is_zero() {
            DiagOffset::Main
        } else if int.is_positive() {
            DiagOffset::Above(int as usize)
        } else {
            DiagOffset::Below((-int) as usize)
        }
    }
}
    }
}

impl_diag_offset_from!(u8, i8);
impl_diag_offset_from!(u16, i16);
impl_diag_offset_from!(u32, i32);
impl_diag_offset_from!(u64, i64);
impl_diag_offset_from!(usize, isize);

#[cfg(test)]
mod tests {
    use matrix::{DiagOffset, Matrix, MatrixSlice, MatrixSliceMut, BaseMatrix};
    use vector::Vector;

    #[test]
    fn inner_product_as_matrix_multiplication() {
        let u: Vector<f32> = Vector::new(vec![1., 2., 3.]);
        let v: Vector<f32> = Vector::new(vec![3., 4., 5.]);
        let dot_product = u.dot(&v);

        let um: Matrix<f32> = u.into();
        let vm: Matrix<f32> = v.into();
        let matrix_product = um.transpose() * vm;

        assert_eq!(dot_product, matrix_product.data()[0]);
    }

    #[test]
    fn matrix_from_slice() {
        let mut a = Matrix::new(3, 3, vec![2.0; 9]);

        {
            let b = MatrixSlice::from_matrix(&a, [1, 1], 2, 2);
            let c = Matrix::from(b);
            assert_eq!(c.rows(), 2);
            assert_eq!(c.cols(), 2);
        }

        let d = MatrixSliceMut::from_matrix(&mut a, [1, 1], 2, 2);
        let e = Matrix::from(d);
        assert_eq!(e.rows(), 2);
        assert_eq!(e.cols(), 2);
    }

    #[test]
    fn diag_offset_from_int() {
        let a: DiagOffset = 3.into();
        assert_eq!(a, DiagOffset::Above(3));
        let a: DiagOffset = (-3).into();
        assert_eq!(a, DiagOffset::Below(3));
        let a: DiagOffset = 0.into();
        assert_eq!(a, DiagOffset::Main);
    }

}
