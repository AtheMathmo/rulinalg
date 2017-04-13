use std::ops::{Neg, Not};

use vector::Vector;

macro_rules! impl_unary_op (
    ($trt:ident, $op:ident, $sym:tt, $doc:expr) => (
/// Gets
#[doc=$doc]
/// of vector.
impl<T> $trt for Vector<T>
    where T: Copy + $trt<Output = T> {
    type Output = Vector<T>;

    fn $op(mut self) -> Vector<T> {
        for val in &mut self.data {
            *val = $sym *val;
        }
        self
    }
}

/// Gets
#[doc=$doc]
/// of vector.
impl<'a, T> $trt for &'a Vector<T>
    where T: Copy + $trt<Output = T> {
    type Output = Vector<T>;

    fn $op(self) -> Vector<T> {
        let new_data = self.data.iter().map(|v| $sym *v).collect::<Vec<_>>();
        Vector::new(new_data)
    }
}
    );
);
impl_unary_op!(Neg, neg, -, "negative");
impl_unary_op!(Not, not, !, "not");

#[cfg(test)]
mod tests {
    #[test]
    fn vector_neg_f32() {
        let a = vector![1., 2., 3., 4., 5., 6.];
        let exp = vector![-1., -2., -3., -4., -5., -6.];

        assert_eq!(-&a, exp);
        assert_eq!(-a, exp);
    }

    #[test]
    fn vector_neg_int() {
        let a = vector![1, 2, 3, 4, 5, 6];
        let exp = vector![-1, -2, -3, -4, -5, -6];

        assert_eq!(-&a, exp);
        assert_eq!(-a, exp);
    }

    #[test]
    fn vector_not_int() {
        let a = vector![1, 2, 3, 4, 5, 6];
        let exp = vector![!1, !2, !3, !4, !5, !6];

        assert_eq!(!&a, exp);
        assert_eq!(!a, exp);
    }

    #[test]
    fn vector_not_bool() {
        let a = vector![true, false, true];
        let exp = vector![false, true, false];

        assert_eq!(!&a, exp);
        assert_eq!(!a, exp);
    }
}