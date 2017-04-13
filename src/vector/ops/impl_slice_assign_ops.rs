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
impl_assign_op!(BitAndAssign, BitAnd, bitand, bitand_assign, "bitwise-and", VectorSlice);
impl_assign_op!(BitAndAssign, BitAnd, bitand, bitand_assign, "bitwise-and", VectorSliceMut);
impl_assign_op!(BitOrAssign, BitOr, bitor, bitor_assign, "bitwise-or", VectorSlice);
impl_assign_op!(BitOrAssign, BitOr, bitor, bitor_assign, "bitwise-or", VectorSliceMut);
impl_assign_op!(BitXorAssign, BitXor, bitxor, bitxor_assign, "bitwise-xor", VectorSlice);
impl_assign_op!(BitXorAssign, BitXor, bitxor, bitxor_assign, "bitwise-xor", VectorSliceMut);
impl_assign_op!(RemAssign, Rem, rem, rem_assign, "remainder", VectorSlice);
impl_assign_op!(RemAssign, Rem, rem, rem_assign, "remainder", VectorSliceMut);
impl_assign_op!(SubAssign, Sub, sub, sub_assign, "subtraction", VectorSlice);
impl_assign_op!(SubAssign, Sub, sub, sub_assign, "subtraction", VectorSliceMut);

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