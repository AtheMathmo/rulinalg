use std::ops::{Mul, Add, Div, Sub, Rem,
               MulAssign, AddAssign, DivAssign, SubAssign, RemAssign,
               BitAnd, BitOr, BitXor, BitAndAssign, BitOrAssign, BitXorAssign};
               
use vector::Vector;
use utils;

macro_rules! impl_assign_op (
    ($assign_trt:ident, $trt:ident, $op:ident, $op_assign:ident, $doc:expr) => (
/// Performs elementwise
#[doc=$doc]
/// assignment between two vectors.
impl<T> $assign_trt<Vector<T>> for Vector<T>
    where T: Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: Vector<T>) {
        utils::in_place_vec_bin_op(&mut self.data, &_rhs.data, |x, &y| {*x = (*x).$op(y) });
    }
}

/// Performs elementwise
#[doc=$doc]
/// assignment between two vectors.
impl<'a, T> $assign_trt<&'a Vector<T>> for Vector<T>
    where T: Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: &Vector<T>) {
        utils::in_place_vec_bin_op(&mut self.data, &_rhs.data, |x, &y| {*x = (*x).$op(y) });
    }
}
    );
);
impl_assign_op!(AddAssign, Add, add, add_assign, "addition");
impl_assign_op!(SubAssign, Sub, sub, sub_assign, "subtraction");
impl_assign_op!(RemAssign, Rem, rem, rem_assign, "remainder");
impl_assign_op!(BitAndAssign, BitAnd, bitand, bitand_assign, "bitwise-and");
impl_assign_op!(BitOrAssign, BitOr, bitor, bitor_assign, "bitwise-or");
impl_assign_op!(BitXorAssign, BitXor, bitxor, bitxor_assign, "bitwise-xor");

macro_rules! impl_assign_op_scalar (
    ($assign_trt:ident, $trt:ident, $op:ident, $op_assign:ident, $doc:expr) => (
/// Performs
#[doc=$doc]
/// assignment between a vector and a scalar.
impl<T> $assign_trt<T> for Vector<T>
    where T: Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: T) {
        for x in &mut self.data {
            *x = (*x).$op(_rhs)
        }
    }
}

/// Performs
#[doc=$doc]
/// assignment between a vector and a scalar.
impl<'a, T> $assign_trt<&'a T> for Vector<T>
    where T: Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: &T) {
        for x in &mut self.data {
            *x = (*x).$op(*_rhs)
        }
    }
}
    );
);
impl_assign_op_scalar!(AddAssign, Add, add, add_assign, "addition");
impl_assign_op_scalar!(SubAssign, Sub, sub, sub_assign, "subtraction");
impl_assign_op_scalar!(DivAssign, Div, div, div_assign, "division");
impl_assign_op_scalar!(MulAssign, Mul, mul, mul_assign, "multiplication");
impl_assign_op_scalar!(RemAssign, Rem, rem, rem_assign, "reminder");
impl_assign_op_scalar!(BitAndAssign, BitAnd, bitand, bitand_assign, "bitwise-and");
impl_assign_op_scalar!(BitOrAssign, BitOr, bitor, bitor_assign, "bitwise-or");
impl_assign_op_scalar!(BitXorAssign, BitXor, bitxor, bitxor_assign, "bitwise-xor");

#[cfg(test)]
mod tests {
    use vector::Vector;

    /*********************************
    *                                *
    *     Arithmetic Assignments     *
    *                                *
    *********************************/

    #[test]
    fn vector_add_assign_int_broadcast() {
        let mut a = (0..9).collect::<Vector<_>>();

        let exp = (2..11).collect::<Vector<_>>();

        a += &2;
        assert_eq!(a, exp);

        let mut a = (0..9).collect::<Vector<_>>();

        a += 2;
        assert_eq!(a, exp);
    }

    #[test]
    fn vector_add_assign_int_elemwise() {
        let mut a = (0..9).collect::<Vector<_>>();
        let b = (0..9).collect::<Vector<_>>();

        let exp = (0..9).map(|x| 2 * x).collect::<Vector<_>>();

        a += &b;
        assert_eq!(a, exp);

        let mut a = (0..9).collect::<Vector<_>>();

        a += b;
        assert_eq!(a, exp);
    }

    #[test]
    fn vector_sub_assign_int_broadcast() {
        let mut a = (0..9).collect::<Vector<_>>();

        let exp = (-2..7).collect::<Vector<_>>();

        a -= &2;
        assert_eq!(a, exp);

        let mut a = (0..9).collect::<Vector<i32>>();
        a -= 2;
        assert_eq!(a, exp);
    }

    #[test]
    fn vector_sub_assign_int_elemwise() {
        let mut a = vector![1, 2, 3, 4, 5];
        let b = vector![2, 2, 2, 3, 3];

        let exp = vector![-1, 0, 1, 1, 2];

        a -= &b;
        assert_eq!(a, exp);

        let mut a = vector![1, 2, 3, 4, 5];

        a -= b;
        assert_eq!(a, exp);
    }

    #[test]
    fn vector_div_assign_f32_broadcast() {
        let a_data = vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let exp = vector![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];

        let mut a = Vector::new(a_data.clone());

        a /= &2f32;
        assert_eq!(a, exp);

        let mut a = Vector::new(a_data.clone());
        a /= 2f32;
        assert_eq!(a, exp);
    }

    #[test]
    fn vector_mul_assign_f32_broadcast() {
        let a_data = vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let exp = vector![2f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
        let mut a = Vector::new(a_data.clone());

        a *= &2f32;
        assert_eq!(a, exp);

        let mut a = Vector::new(a_data.clone());
        a *= 2f32;
        assert_eq!(a, exp);
    }

    #[test]
    fn vector_rem_assign_int_broadcast() {
        let mut a = vector![1, 2, 3];

        let exp = vector![1, 2, 0];

        a %= &3;
        assert_eq!(a, exp);

        let mut a = vector![1, 2, 3];
        a %= 3;
        assert_eq!(a, exp);
    }

    #[test]
    fn vector_rem_assign_int_elemwise() {
        let mut a = vector![1, 2, 3, 4, 5];
        let b = vector![2, 2, 2, 3, 3];

        let exp = vector![1, 0, 1, 1, 2];

        a %= &b;
        assert_eq!(a, exp);

        let mut a = vector![1, 2, 3, 4, 5];

        a %= b;
        assert_eq!(a, exp);
    }

    /******************************
    *                             *
    *     Bitwise Assignments     *
    *                             *
    ******************************/

    #[test]
    fn vector_bitand_assign_int_broadcast() {
        let mut a = vector![1, 2, 3, 4, 5];
        let b = 2;

        let exp = vector![1 & 2, 2 & 2, 3 & 2, 4 & 2, 5 & 2];

        a &= &b;
        assert_eq!(a, exp);

        let mut a = vector![1, 2, 3, 4, 5];

        a &= b;
        assert_eq!(a, exp);
    }

    #[test]
    fn vector_bitand_assign_bool_broadcast() {
        let mut a = vector![true, true, false, false];
        let b = true;

        let exp = vector![true, true, false, false];

        a &= &b;
        assert_eq!(a, exp);

        let mut a = vector![true, true, false, false];

        a &= b;
        assert_eq!(a, exp);
    }

    #[test]
    fn vector_bitand_assign_int_elemwise() {
        let mut a = vector![1, 2, 3, 4, 5];
        let b = vector![2, 2, 2, 3, 3];

        let exp = vector![1 & 2, 2 & 2, 3 & 2, 4 & 3, 5 & 3];

        a &= &b;
        assert_eq!(a, exp);

        let mut a = vector![1, 2, 3, 4, 5];

        a &= b;
        assert_eq!(a, exp);
    }

    #[test]
    fn vector_bitand_assign_bool_elemwise() {
        let mut a = vector![true, true, false, false];
        let b = vector![true, false, true, false];

        let exp = vector![true, false, false, false];

        a &= &b;
        assert_eq!(a, exp);

        let mut a = vector![true, true, false, false];

        a &= b;
        assert_eq!(a, exp);
    }

    #[test]
    fn vector_bitor_assign_int_broadcast() {
        let mut a = vector![1, 2, 3, 4, 5];
        let b = 2;

        let exp = vector![1 | 2, 2 | 2, 3 | 2, 4 | 2, 5 | 2];

        a |= &b;
        assert_eq!(a, exp);

        let mut a = vector![1, 2, 3, 4, 5];

        a |= b;
        assert_eq!(a, exp);
    }

    #[test]
    fn vector_bitor_assign_bool_broadcast() {
        let mut a = vector![true, true, false, false];
        let b = true;

        let exp = vector![true, true, true, true];

        a |= &b;
        assert_eq!(a, exp);

        let mut a = vector![true, true, false, false];

        a |= b;
        assert_eq!(a, exp);
    }

    #[test]
    fn vector_bitor_assign_int_elemwise() {
        let mut a = vector![1, 2, 3, 4, 5];
        let b = vector![2, 2, 2, 3, 3];

        let exp = vector![1 | 2, 2 | 2, 3 | 2, 4 | 3, 5 | 3];

        a |= &b;
        assert_eq!(a, exp);

        let mut a = vector![1, 2, 3, 4, 5];

        a |= b;
        assert_eq!(a, exp);
    }

    #[test]
    fn vector_bitor_assign_bool_elemwise() {
        let mut a = vector![true, true, false, false];
        let b = vector![true, false, true, false];

        let exp = vector![true, true, true, false];

        a |= &b;
        assert_eq!(a, exp);

        let mut a = vector![true, true, false, false];

        a |= b;
        assert_eq!(a, exp);
    }

    #[test]
    fn vector_bitxor_assign_int_broadcast() {
        let mut a = vector![1, 2, 3, 4, 5];
        let b = 2;

        let exp = vector![1 ^ 2, 2 ^ 2, 3 ^ 2, 4 ^ 2, 5 ^ 2];

        a ^= &b;
        assert_eq!(a, exp);

        let mut a = vector![1, 2, 3, 4, 5];

        a ^= b;
        assert_eq!(a, exp);
    }

    #[test]
    fn vector_bitxor_assign_bool_broadcast() {
        let mut a = vector![true, true, false, false];
        let b = true;

        let exp = vector![false, false, true, true];

        a ^= &b;
        assert_eq!(a, exp);

        let mut a = vector![true, true, false, false];

        a ^= b;
        assert_eq!(a, exp);
    }

    #[test]
    fn vector_bitxor_assign_int_elemwise() {
        let mut a = vector![1, 2, 3, 4, 5];
        let b = vector![2, 2, 2, 3, 3];

        let exp = vector![1 ^ 2, 2 ^ 2, 3 ^ 2, 4 ^ 3, 5 ^ 3];

        a ^= &b;
        assert_eq!(a, exp);

        let mut a = vector![1, 2, 3, 4, 5];

        a ^= b;
        assert_eq!(a, exp);
    }

    #[test]
    fn vector_bitxor_assign_bool_elemwise() {
        let mut a = vector![true, true, false, false];
        let b = vector![true, false, true, false];

        let exp = vector![false, true, true, false];

        a ^= &b;
        assert_eq!(a, exp);

        let mut a = vector![true, true, false, false];

        a ^= b;
        assert_eq!(a, exp);
    }
}
