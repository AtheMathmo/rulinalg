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

#[cfg(test)]
mod tests {
    use vector::{BaseVector, VectorSlice};

    #[test]
    fn vector_slice_neg_f32() {
        let a = vector![1., 2., 3., 4., 5., 6.];
        let b = VectorSlice::from_vector(&a, 0, 6);
        let exp = &[-1., -2., -3., -4., -5., -6.];

        assert_eq!((-&b).data(), exp);
        assert_eq!((-b).data(), exp);
    }

    #[test]
    fn vector_slice_neg_int() {
        let a = vector![1, 2, 3, 4, 5, 6];
        let b = VectorSlice::from_vector(&a, 0, 6);
        let exp = &[-1, -2, -3, -4, -5, -6];

        assert_eq!((-&b).data(), exp);
        assert_eq!((-b).data(), exp);
    }

    #[test]
    fn vector_slice_not_int() {
        let a = vector![1, 2, 3, 4, 5, 6];
        let b = VectorSlice::from_vector(&a, 0, 6);
        let exp = &[!1, !2, !3, !4, !5, !6];

        assert_eq!((!&b).data(), exp);
        assert_eq!((!b).data(), exp);
    }

    #[test]
    fn vector_slice_not_bool() {
        let a = vector![false, true, false];
        let b = VectorSlice::from_vector(&a, 0, 3);
        let exp = &[true, false, true];

        assert_eq!((!&b).data(), exp);
        assert_eq!((!b).data(), exp);
    }
}