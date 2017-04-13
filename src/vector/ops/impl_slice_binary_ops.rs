use std::ops::{Mul, Add, Div, Sub, Rem, BitAnd, BitOr, BitXor};

use vector::{Vector, VectorSlice, VectorSliceMut, BaseVector};
use utils;

macro_rules! impl_bin_op (
    ($trt:ident, $op:ident, $doc:expr, $slice_1:ident, $slice_2:ident) => (
/// Performs
#[doc=$doc]
/// between the slices.
impl<'a, 'b, T> $trt<$slice_1<'a, T>> for $slice_2<'b, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Vector<T>;
    fn $op(self, s: $slice_1<T>) -> Vector<T> {
        (&self).$op(s)
    }
}

/// Performs
#[doc=$doc]
/// between the slices.
impl<'a, 'b, 'c, T> $trt<&'a $slice_1<'b, T>> for $slice_2<'c, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Vector<T>;
    fn $op(self, s: &$slice_1<T>) -> Vector<T> {
        (&self).$op(s)
    }
}

/// Performs
#[doc=$doc]
/// between the slices.
impl<'a, 'b, 'c, T> $trt<$slice_1<'a, T>> for &'b $slice_2<'c, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Vector<T>;
    fn $op(self, s: $slice_1<T>) -> Vector<T> {
        self.$op(&s)
    }
}

/// Performs
#[doc=$doc]
/// between the slices.
impl<'a, 'b, 'c, 'd, T> $trt<&'a $slice_1<'b, T>> for &'c $slice_2<'d, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Vector<T>;
    fn $op(self, f: &$slice_1<T>) -> Vector<T> {
        assert!(self.size == f.size);
        let new_data = utils::vec_bin_op(self.data(), f.data(), |x, y| x.$op(y));
        Vector {
            size: self.size,
            data: new_data,
        }
    }
}
    );
);
impl_bin_op!(Add, add, "addition", VectorSlice, VectorSlice);
impl_bin_op!(Add, add, "addition", VectorSlice, VectorSliceMut);
impl_bin_op!(Add, add, "addition", VectorSliceMut, VectorSlice);
impl_bin_op!(Add, add, "addition", VectorSliceMut, VectorSliceMut);
impl_bin_op!(BitAnd, bitand, "bitwise-and", VectorSlice, VectorSlice);
impl_bin_op!(BitAnd, bitand, "bitwise-and", VectorSlice, VectorSliceMut);
impl_bin_op!(BitAnd, bitand, "bitwise-and", VectorSliceMut, VectorSlice);
impl_bin_op!(BitAnd,
             bitand,
             "bitwise-and",
             VectorSliceMut,
             VectorSliceMut);
impl_bin_op!(BitOr, bitor, "bitwise-or", VectorSlice, VectorSlice);
impl_bin_op!(BitOr, bitor, "bitwise-or", VectorSlice, VectorSliceMut);
impl_bin_op!(BitOr, bitor, "bitwise-or", VectorSliceMut, VectorSlice);
impl_bin_op!(BitOr, bitor, "bitwise-or", VectorSliceMut, VectorSliceMut);
impl_bin_op!(BitXor, bitxor, "bitwise-xor", VectorSlice, VectorSlice);
impl_bin_op!(BitXor, bitxor, "bitwise-xor", VectorSlice, VectorSliceMut);
impl_bin_op!(BitXor, bitxor, "bitwise-xor", VectorSliceMut, VectorSlice);
impl_bin_op!(BitXor,
             bitxor,
             "bitwise-xor",
             VectorSliceMut,
             VectorSliceMut);
impl_bin_op!(Rem, rem, "remainder", VectorSlice, VectorSlice);
impl_bin_op!(Rem, rem, "remainder", VectorSlice, VectorSliceMut);
impl_bin_op!(Rem, rem, "remainder", VectorSliceMut, VectorSlice);
impl_bin_op!(Rem, rem, "remainder", VectorSliceMut, VectorSliceMut);
impl_bin_op!(Sub, sub, "subtraction", VectorSlice, VectorSlice);
impl_bin_op!(Sub, sub, "subtraction", VectorSlice, VectorSliceMut);
impl_bin_op!(Sub, sub, "subtraction", VectorSliceMut, VectorSlice);
impl_bin_op!(Sub, sub, "subtraction", VectorSliceMut, VectorSliceMut);

macro_rules! impl_bin_op_scalar (
    ($trt:ident, $op:ident, $doc:expr, $slice:ident) => (
/// Scalar
#[doc=$doc]
/// with vector slice.
impl<'a, T> $trt<T> for $slice<'a, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Vector<T>;
    fn $op(self, f: T) -> Vector<T> {
        (&self).$op(f)
    }
}

/// Scalar
#[doc=$doc]
/// with vector slice.
impl<'a, 'b, T> $trt<&'a T> for $slice<'b, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Vector<T>;
    fn $op(self, f: &T) -> Vector<T> {
        (&self).$op(f)
    }
}

/// Scalar
#[doc=$doc]
/// with vector slice.
impl<'a, 'b, T> $trt<T> for &'a $slice<'b, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Vector<T>;
    fn $op(self, f: T) -> Vector<T> {
        (&self).$op(&f)
    }
}

/// Scalar
#[doc=$doc]
/// with vector slice.
impl<'a, 'b, 'c, T> $trt<&'a T> for &'b $slice<'c, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Vector<T>;
    fn $op(self, f: &T) -> Vector<T> {
        let new_data = self.iter().map(|v| v.$op(*f)).collect();
        Vector { size: self.size, data: new_data }
    }
}
    );
);
impl_bin_op_scalar!(Add, add, "addition", VectorSlice);
impl_bin_op_scalar!(Add, add, "addition", VectorSliceMut);
impl_bin_op_scalar!(BitAnd, bitand, "bitwise-and", VectorSlice);
impl_bin_op_scalar!(BitAnd, bitand, "bitwise-and", VectorSliceMut);
impl_bin_op_scalar!(BitOr, bitor, "bitwise-or", VectorSlice);
impl_bin_op_scalar!(BitOr, bitor, "bitwise-or", VectorSliceMut);
impl_bin_op_scalar!(BitXor, bitxor, "bitwise-xor", VectorSlice);
impl_bin_op_scalar!(BitXor, bitxor, "bitwise-xor", VectorSliceMut);
impl_bin_op_scalar!(Div, div, "division", VectorSlice);
impl_bin_op_scalar!(Div, div, "division", VectorSliceMut);
impl_bin_op_scalar!(Mul, mul, "multiplication", VectorSlice);
impl_bin_op_scalar!(Mul, mul, "multiplication", VectorSliceMut);
impl_bin_op_scalar!(Rem, rem, "remainder", VectorSlice);
impl_bin_op_scalar!(Rem, rem, "remainder", VectorSliceMut);
impl_bin_op_scalar!(Sub, sub, "subtraction", VectorSlice);
impl_bin_op_scalar!(Sub, sub, "subtraction", VectorSliceMut);

macro_rules! impl_bin_op_vec (
    ($trt:ident, $op:ident, $slice:ident, $doc:expr) => (
/// Performs
#[doc=$doc]
/// between `Vector` and `VectorSlice`.
impl<'a, T> $trt<Vector<T>> for $slice<'a, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Vector<T>;
    fn $op(self, v: Vector<T>) -> Vector<T> {
        (&self).$op(&v)
    }
}

/// Performs
#[doc=$doc]
/// between `Vector` and `VectorSlice`.
impl<'a, 'b, T> $trt<&'a Vector<T>> for $slice<'b, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Vector<T>;
    fn $op(self, v: &Vector<T>) -> Vector<T> {
        (&self).$op(v)
    }
}

/// Performs
#[doc=$doc]
/// between `Vector` and `VectorSlice`.
impl<'a, 'b, T> $trt<Vector<T>> for &'a $slice<'b, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Vector<T>;
    fn $op(self, v: Vector<T>) -> Vector<T> {
        self.$op(&v)
    }
}

/// Performs
#[doc=$doc]
/// between `Vector` and `VectorSlice`.
impl<'a, 'b, 'c, T> $trt<&'a Vector<T>> for &'b $slice<'c, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Vector<T>;
    fn $op(self, v: &Vector<T>) -> Vector<T> {
        assert!(self.size == v.size, "Size dimensions do not agree.");

        let mut new_data : Vec<T> = self.iter().cloned().collect();
        utils::in_place_vec_bin_op(&mut new_data, &v.data(), |x, &y| { *x = (*x).$op(y) });

        Vector::new(new_data)
    }
}
    );
);

impl_bin_op_vec!(Add, add, VectorSlice, "addition");
impl_bin_op_vec!(Add, add, VectorSliceMut, "addition");
impl_bin_op_vec!(BitAnd, bitand, VectorSlice, "bitwise-and");
impl_bin_op_vec!(BitAnd, bitand, VectorSliceMut, "bitwise-and");
impl_bin_op_vec!(BitOr, bitor, VectorSlice, "bitwise-or");
impl_bin_op_vec!(BitOr, bitor, VectorSliceMut, "bitwise-or");
impl_bin_op_vec!(BitXor, bitxor, VectorSlice, "bitwise-xor");
impl_bin_op_vec!(BitXor, bitxor, VectorSliceMut, "bitwise-xor");
impl_bin_op_vec!(Div, div, VectorSlice, "division");
impl_bin_op_vec!(Div, div, VectorSliceMut, "division");
impl_bin_op_vec!(Mul, mul, VectorSlice, "multiplication");
impl_bin_op_vec!(Mul, mul, VectorSliceMut, "multiplication");
impl_bin_op_vec!(Rem, rem, VectorSlice, "remainder");
impl_bin_op_vec!(Rem, rem, VectorSliceMut, "remainder");
impl_bin_op_vec!(Sub, sub, VectorSlice, "subtraction");
impl_bin_op_vec!(Sub, sub, VectorSliceMut, "subtraction");