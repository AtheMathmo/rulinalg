use std::ops::{Neg, Not};

use vector::{Vector, VectorSlice, VectorSliceMut, BaseVector};

macro_rules! impl_unary_op (
    ($trt:ident, $op:ident, $doc:expr, $slice:ident) => (
/// Gets
#[doc=$doc]
/// of vector slice.
impl<'a, T> $trt for $slice<'a, T>
    where T: Copy + $trt<Output = T> {
    type Output = Vector<T>;
    fn $op(self) -> Vector<T> {
        (&self).$op()
    }
}

/// Gets
#[doc=$doc]
/// of vector slice.
impl<'a, 'b, T> $trt for &'a $slice<'b, T>
    where T: Copy + $trt<Output = T> {
    type Output = Vector<T>;
    fn $op(self) -> Vector<T> {
        let new_data = self.iter().map(|v| v.$op()).collect::<Vec<_>>();
        Vector::new(new_data)
    }
}
    );
);
impl_unary_op!(Neg, neg, "negative", VectorSlice);
impl_unary_op!(Neg, neg, "negative", VectorSliceMut);
impl_unary_op!(Not, not, "not", VectorSlice);
impl_unary_op!(Not, not, "not", VectorSliceMut);