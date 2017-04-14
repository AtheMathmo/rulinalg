use std::ops::{Mul, Add, Div, Sub, Rem, MulAssign, AddAssign, DivAssign, SubAssign, RemAssign,
               BitAnd, BitOr, BitXor, BitAndAssign, BitOrAssign, BitXorAssign};

use vector::{VectorSlice, VectorSliceMut, BaseVector, BaseVectorMut};
use utils;

macro_rules! impl_assign_op (
    ($assign_trt:ident, $trt:ident, $op:ident, $op_assign:ident, $doc:expr, $slice:ident) => (
/// Performs
#[doc=$doc]
/// assignment between two vectors
impl<'a, 'b, T> $assign_trt<$slice<'a, T>> for VectorSliceMut<'b, T>
    where T: Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: $slice<T>) {
        utils::in_place_vec_bin_op(self.data_mut(), _rhs.data(), |x, &y| {*x = (*x).$op(y) });
    }
}

/// Performs
#[doc=$doc]
/// assignment between two vectors
impl<'a, 'b, 'c, T> $assign_trt<&'a $slice<'b, T>> for VectorSliceMut<'c, T>
    where T : Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: &$slice<T>) {
        utils::in_place_vec_bin_op(self.data_mut(), _rhs.data(), |x, &y| {*x = (*x).$op(y) });
    }
}
    );
);
impl_assign_op!(AddAssign, Add, add, add_assign, "addition", VectorSlice);
impl_assign_op!(AddAssign, Add, add, add_assign, "addition", VectorSliceMut);
impl_assign_op!(BitAndAssign,
                BitAnd,
                bitand,
                bitand_assign,
                "bitwise-and",
                VectorSlice);
impl_assign_op!(BitAndAssign,
                BitAnd,
                bitand,
                bitand_assign,
                "bitwise-and",
                VectorSliceMut);
impl_assign_op!(BitOrAssign,
                BitOr,
                bitor,
                bitor_assign,
                "bitwise-or",
                VectorSlice);
impl_assign_op!(BitOrAssign,
                BitOr,
                bitor,
                bitor_assign,
                "bitwise-or",
                VectorSliceMut);
impl_assign_op!(BitXorAssign,
                BitXor,
                bitxor,
                bitxor_assign,
                "bitwise-xor",
                VectorSlice);
impl_assign_op!(BitXorAssign,
                BitXor,
                bitxor,
                bitxor_assign,
                "bitwise-xor",
                VectorSliceMut);
impl_assign_op!(RemAssign, Rem, rem, rem_assign, "remainder", VectorSlice);
impl_assign_op!(RemAssign, Rem, rem, rem_assign, "remainder", VectorSliceMut);
impl_assign_op!(SubAssign, Sub, sub, sub_assign, "subtraction", VectorSlice);
impl_assign_op!(SubAssign,
                Sub,
                sub,
                sub_assign,
                "subtraction",
                VectorSliceMut);

macro_rules! impl_assign_op_scalar (
    ($assign_trt:ident, $trt:ident, $op:ident, $op_assign:ident, $doc:expr) => (
/// Performs
#[doc=$doc]
/// assignment between a mutable vector slice and a scalar.
impl<'a, T> $assign_trt<T> for VectorSliceMut<'a, T>
    where T: Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: T) {
        for x in self.iter_mut() {
            *x = (*x).$op(_rhs)
        }
    }
}

/// Performs
#[doc=$doc]
/// assignment between a mutable vector slice and a scalar.
impl<'a, 'b, T> $assign_trt<&'a T> for VectorSliceMut<'b, T>
    where T : Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: &T) {
        for x in self.iter_mut() {
            *x = (*x).$op(*_rhs)
        }
    }
}
    );
);
impl_assign_op_scalar!(AddAssign, Add, add, add_assign, "addition");
impl_assign_op_scalar!(BitAndAssign, BitAnd, bitand, bitand_assign, "bitwise-and");
impl_assign_op_scalar!(BitOrAssign, BitOr, bitor, bitor_assign, "bitwise-or");
impl_assign_op_scalar!(BitXorAssign, BitXor, bitxor, bitxor_assign, "bitwise-xor");
impl_assign_op_scalar!(DivAssign, Div, div, div_assign, "division");
impl_assign_op_scalar!(MulAssign, Mul, mul, mul_assign, "multiplication");
impl_assign_op_scalar!(RemAssign, Rem, rem, rem_assign, "reminder");
impl_assign_op_scalar!(SubAssign, Sub, sub, sub_assign, "subtraction");

#[cfg(test)]
mod tests {
    use vector::{BaseVector, VectorSlice, VectorSliceMut};

    /*********************************
     *                                *
     *     Arithmetic Assignments     *
     *                                *
     *********************************/

    #[test]
    fn vector_slice_add_assign_int_broadcast() {
        let mut a = vector![1, 2, 3];
        let mut b = VectorSliceMut::from_vector(&mut a, 0, 3);

        b += &2;
        assert_eq!(b.data(), &[3, 4, 5]);

        b += 2;
        assert_eq!(b.data(), &[5, 6, 7]);
    }

    #[test]
    fn vector_slice_add_assign_int_elemwise() {
        let mut a = vector![1, 2, 3];
        let mut b = vector![4, 5, 6];
        let mut c = VectorSliceMut::from_vector(&mut a, 0, 3);

        let d = vector![7, 8, 9];
        let e = VectorSlice::from_vector(&d, 0, 3);
        let f = VectorSliceMut::from_vector(&mut b, 0, 3);

        c += &e;
        assert_eq!(c.data(), &[8, 10, 12]);
        c += &f;
        assert_eq!(c.data(), &[12, 15, 18]);

        c += e;
        assert_eq!(c.data(), &[19, 23, 27]);
        c += f;
        assert_eq!(c.data(), &[23, 28, 33]);
    }

    #[test]
    fn vector_slice_sub_assign_int_broadcast() {
        let mut a = vector![1, 2, 3];
        let mut b = VectorSliceMut::from_vector(&mut a, 0, 3);

        b -= &2;
        assert_eq!(b.data(), &[-1, 0, 1]);

        b -= 2;
        assert_eq!(b.data(), &[-3, -2, -1]);
    }

    #[test]
    fn vector_slice_sub_assign_int_elemwise() {
        let mut a = vector![1, 2, 3];
        let mut b = vector![4, 5, 6];
        let mut c = VectorSliceMut::from_vector(&mut a, 0, 3);

        let d = vector![7, 8, 9];
        let e = VectorSlice::from_vector(&d, 0, 3);
        let f = VectorSliceMut::from_vector(&mut b, 0, 3);

        c -= &e;
        assert_eq!(c.data(), &[-6, -6, -6]);
        c -= &f;
        assert_eq!(c.data(), &[-10, -11, -12]);

        c -= e;
        assert_eq!(c.data(), &[-17, -19, -21]);
        c -= f;
        assert_eq!(c.data(), &[-21, -24, -27]);
    }

    #[test]
    fn vector_slice_div_assign_f32_broadcast() {
        let mut a = vector![1.0, 2.0, 3.0];
        let mut b = VectorSliceMut::from_vector(&mut a, 0, 3);

        b /= &2f32;
        assert_eq!(b.data(), &[0.5, 1.0, 1.5]);

        b /= 2f32;
        assert_eq!(b.data(), &[0.25, 0.5, 0.75]);
    }

    #[test]
    fn vector_slice_mul_assign_f32_broadcast() {
        let mut a = vector![1.0, 2.0, 3.0];
        let mut b = VectorSliceMut::from_vector(&mut a, 0, 3);

        b *= &2f32;
        assert_eq!(b.data(), &[2.0, 4.0, 6.0]);

        b *= 2f32;
        assert_eq!(b.data(), &[4.0, 8.0, 12.0]);
    }

    #[test]
    fn vector_slice_rem_assign_int_broadcast() {
        let mut a = vector![1, 2, 3];
        let mut b = VectorSliceMut::from_vector(&mut a, 0, 3);

        b %= &2;
        assert_eq!(b.data(), &[1, 0, 1]);

        b %= 2;
        assert_eq!(b.data(), &[1, 0, 1]);
    }

    #[test]
    fn vector_slice_rem_assign_int_elemwise() {
        let mut a = vector![1, 2, 3];
        let mut b = vector![4, 5, 6];
        let mut c = VectorSliceMut::from_vector(&mut a, 0, 3);

        let d = vector![2, 2, 2];
        let e = VectorSlice::from_vector(&d, 0, 3);
        let f = VectorSliceMut::from_vector(&mut b, 0, 3);

        c %= &e;
        assert_eq!(c.data(), &[1, 0, 1]);
        c %= &f;
        assert_eq!(c.data(), &[1, 0, 1]);

        c %= e;
        assert_eq!(c.data(), &[1, 0, 1]);
        c %= f;
        assert_eq!(c.data(), &[1, 0, 1]);
    }

    /******************************
     *                             *
     *     Bitwise Assignments     *
     *                             *
     ******************************/

    #[test]
    fn vector_slice_bitand_assign_int_broadcast() {
        let mut a = vector![1, 2, 3];
        let mut b = VectorSliceMut::from_vector(&mut a, 0, 3);

        b &= &2;
        assert_eq!(b.data(), &[0, 2, 2]);

        b &= 1;
        assert_eq!(b.data(), &[0, 0, 0]);
    }
    
    #[test]
    fn vector_slice_bitor_assign_int_broadcast() {
        let mut a = vector![1, 2, 3];
        let mut b = VectorSliceMut::from_vector(&mut a, 0, 3);

        b |= &2;
        assert_eq!(b.data(), &[3, 2, 3]);

        b |= 1;
        assert_eq!(b.data(), &[3, 3, 3]);
    }

    #[test]
    fn vector_slice_bitxor_assign_int_broadcast() {
        let mut a = vector![1, 2, 3];
        let mut b = VectorSliceMut::from_vector(&mut a, 0, 3);

        b ^= &2;
        assert_eq!(b.data(), &[3, 0, 1]);

        b ^= 1;
        assert_eq!(b.data(), &[2, 1, 0]);
    }
}