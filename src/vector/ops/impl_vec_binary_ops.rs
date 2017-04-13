use std::ops::{Mul, Add, Div, Sub, Rem, BitAnd, BitOr, BitXor};

use vector::{Vector, VectorSlice, VectorSliceMut, BaseVector};
use utils;

macro_rules! impl_bin_op (
    ($trt:ident, $op:ident, $sym:tt, $doc:expr) => (
#[doc=$doc]
impl<T> $trt<Vector<T>> for Vector<T>
    where T: Copy + $trt<T, Output = T> {
    type Output = Vector<T>;
    fn $op(self, v: Vector<T>) -> Vector<T> {
        self $sym &v
    }
}

#[doc=$doc]
impl<'a, T> $trt<&'a Vector<T>> for Vector<T>
    where T: Copy + $trt<T, Output = T> {
    type Output = Vector<T>;

    fn $op(mut self, v: &Vector<T>) -> Vector<T> {
        utils::in_place_vec_bin_op(&mut self.data, &v.data, |x, &y| *x = *x $sym y);
        self
    }
}

#[doc=$doc]
impl<'a, T> $trt<Vector<T>> for &'a Vector<T>
    where T: Copy + $trt<T, Output = T> {
    type Output = Vector<T>;

    fn $op(self, mut v: Vector<T>) -> Vector<T> {
        utils::in_place_vec_bin_op(&mut v.data, &self.data, |y, &x| *y = x $sym *y);
        v
    }
}

#[doc=$doc]
impl<'a, 'b, T> $trt<&'b Vector<T>> for &'a Vector<T>
    where T: Copy + $trt<T, Output = T> {
    type Output = Vector<T>;

    fn $op(self, v: &Vector<T>) -> Vector<T> {
        assert!(self.size == v.size);
        let new_data = utils::vec_bin_op(&self.data, &v.data, |x, y| x $sym y);
        Vector {
            size: self.size,
            data: new_data,
        }
    }
}
    );
);
impl_bin_op!(Add, add, +, "addition");
impl_bin_op!(BitAnd, bitand, &, "bitwise-and");
impl_bin_op!(BitOr, bitor, |, "bitwise-or");
impl_bin_op!(BitXor, bitxor, ^, "bitwise-xor");
impl_bin_op!(Rem, rem, %, "remainder");
impl_bin_op!(Sub, sub, -, "subtraction");

macro_rules! impl_bin_op_scalar (
    ($trt:ident, $op:ident, $sym:tt, $doc:expr) => (
#[doc=$doc]
impl<T> $trt<T> for Vector<T>
    where T: Copy + $trt<T, Output = T> {
    type Output = Vector<T>;
    fn $op(self, f: T) -> Vector<T> {
        self $sym &f
    }
}

#[doc=$doc]
impl<'a, T> $trt<&'a T> for Vector<T>
    where T: Copy + $trt<T, Output = T> {
    type Output = Vector<T>;
    fn $op(mut self, f: &T) -> Vector<T> {
        for val in &mut self.data {
            *val = *val $sym *f;
        }
        self
    }
}

#[doc=$doc]
impl<'a, T> $trt<T> for &'a Vector<T>
    where T: Copy + $trt<T, Output = T> {
    type Output = Vector<T>;
    fn $op(self, f: T) -> Vector<T> {
        self $sym &f
    }
}

#[doc=$doc]
impl<'a, 'b, T> $trt<&'b T> for &'a Vector<T>
    where T: Copy + $trt<T, Output = T> {
    type Output = Vector<T>;
    fn $op(self, f: &T) -> Vector<T> {
        let new_data = self.data.iter().map(|v| *v $sym *f).collect();
        Vector { size: self.size, data: new_data }
    }
}
    );
);
impl_bin_op_scalar!(Add, add, +, "addition");
impl_bin_op_scalar!(Mul, mul, *, "multiplication");
impl_bin_op_scalar!(Sub, sub, -, "subtraction");
impl_bin_op_scalar!(Div, div, /, "division");
impl_bin_op_scalar!(Rem, rem, %, "remainder");
impl_bin_op_scalar!(BitAnd, bitand, &, "bitwise-and");
impl_bin_op_scalar!(BitOr, bitor, |, "bitwise-or");
impl_bin_op_scalar!(BitXor, bitxor, ^, "bitwise-xor");

macro_rules! impl_bin_op_slice (
    ($trt:ident, $op:ident, $slice:ident, $doc:expr) => (
/// Performs
#[doc=$doc]
/// between `Vector` and `VectorSlice`.
impl<'a, T> $trt<$slice<'a, T>> for Vector<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Vector<T>;

    fn $op(self, s: $slice<T>) -> Vector<T> {
        (&self).$op(s)
    }
}

/// Performs
#[doc=$doc]
/// between `Vector` and `VectorSlice`.
impl<'a, 'b, T> $trt<&'a $slice<'b, T>> for Vector<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Vector<T>;

    fn $op(self, s: &$slice<T>) -> Vector<T> {
        (&self).$op(s)
    }
}

/// Performs
#[doc=$doc]
/// between `Vector` and `VectorSlice`.
impl<'a, 'b, T> $trt<$slice<'a, T>> for &'b Vector<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Vector<T>;

    fn $op(self, s: $slice<T>) -> Vector<T> {
        self.$op(&s)
    }
}

/// Performs
#[doc=$doc]
/// between `Vector` and `VectorSlice`.
impl<'a, 'b, 'c, T> $trt<&'a $slice<'b, T>> for &'c Vector<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Vector<T>;

    fn $op(self, s: &$slice<T>) -> Vector<T> {
        assert!(self.size == s.size, "Size dimensions do not agree.");

        let mut new_data : Vec<T> = s.iter().cloned().collect();
        utils::in_place_vec_bin_op(&mut new_data, self.data(), |x, &y| { *x = (y).$op(*x) });

        Vector::new(new_data)
    }
}
    );
);

impl_bin_op_slice!(Add, add, VectorSlice, "addition");
impl_bin_op_slice!(Add, add, VectorSliceMut, "addition");
impl_bin_op_slice!(BitAnd, bitand, VectorSlice, "bitwise-and");
impl_bin_op_slice!(BitAnd, bitand, VectorSliceMut, "bitwise-and");
impl_bin_op_slice!(BitOr, bitor, VectorSlice, "bitwise-or");
impl_bin_op_slice!(BitOr, bitor, VectorSliceMut, "bitwise-or");
impl_bin_op_slice!(BitXor, bitxor, VectorSlice, "bitwise-xor");
impl_bin_op_slice!(BitXor, bitxor, VectorSliceMut, "bitwise-xor");
impl_bin_op_slice!(Div, div, VectorSlice, "division");
impl_bin_op_slice!(Div, div, VectorSliceMut, "division");
impl_bin_op_slice!(Mul, mul, VectorSlice, "multiplication");
impl_bin_op_slice!(Mul, mul, VectorSliceMut, "multiplication");
impl_bin_op_slice!(Rem, rem, VectorSlice, "remainder");
impl_bin_op_slice!(Rem, rem, VectorSliceMut, "remainder");
impl_bin_op_slice!(Sub, sub, VectorSlice, "subtraction");
impl_bin_op_slice!(Sub, sub, VectorSliceMut, "subtraction");

#[cfg(test)]
mod tests {
    use super::super::{BaseVector, Vector};

    /*************************
     *                        *
     *     Arithmetic Ops     *
     *                        *
     *************************/

    #[test]
    fn vector_mul_f32_broadcast() {
        let a = vector![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = 3.0;

        let exp = vector![3.0, 6.0, 9.0, 12.0, 15.0, 18.0];

        // Allocating new memory
        let c = &a * &b;
        assert_eq!(c, exp);

        // Allocating new memory
        let c = &a * b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() * &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a * b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_mul_int_broadcast() {
        let a = vector![1, 2, 3, 4, 5];
        let b = 2;

        let exp = vector![2, 4, 6, 8, 10];

        // Allocating new memory
        let c = &a * &b;
        assert_eq!(c, exp);

        // Allocating new memory
        let c = &a * b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() * &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a * b;
        assert_eq!(c, exp);
    }

    // mul_xxx_elemwise is tested in impl_vec

    #[test]
    fn vector_div_f32_broadcast() {
        let a = vector![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = 3.0;

        let exp = vector![1. / 3., 2. / 3., 3. / 3., 4. / 3., 5. / 3., 6. / 3.];

        // Allocating new memory
        let c = &a / &b;
        assert_eq!(c, exp);

        // Allocating new memory
        let c = &a / b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() / &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a / b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_div_int_broadcast() {
        let a = vector![1, 2, 3, 4, 5];
        let b = 2;

        let exp = vector![0, 1, 1, 2, 2];

        // Allocating new memory
        let c = &a / &b;
        assert_eq!(c, exp);

        // Allocating new memory
        let c = &a / b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() / &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a / b;
        assert_eq!(c, exp);
    }

    // div_xxx_elemwise is tested in impl_vec

    #[test]
    fn vector_add_f32_broadcast() {
        let a = vector![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = 2.0;

        let exp = vector![3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Allocating new memory
        let c = &a + &b;
        assert_eq!(c, exp);

        // Allocating new memory
        let c = &a + b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() + &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a + b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_add_int_broadcast() {
        let a = vector![1, 2, 3, 4, 5];
        let b = 2;

        let exp = vector![3, 4, 5, 6, 7];

        // Allocating new memory
        let c = &a + &b;
        assert_eq!(c, exp);

        // Allocating new memory
        let c = &a + b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() + &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a + b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_add_f32_elemwise() {
        let a = vector![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vector![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let exp = vector![3.0, 5.0, 7.0, 9.0, 11.0, 13.0];

        // Allocating new memory
        let c = &a + &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = &a + b.clone();
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() + &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a + b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_add_int_elemwise() {
        let a = vector![1, 2, 3, 4, 5, 6];
        let b = vector![2, 3, 4, 5, 6, 7];

        let exp = vector![3, 5, 7, 9, 11, 13];

        // Allocating new memory
        let c = &a + &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = &a + b.clone();
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() + &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a + b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_sub_f32_broadcast() {
        let a = vector![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = 2.0;

        let exp = vector![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0];

        // Allocating new memory
        let c = &a - &b;
        assert_eq!(c, exp);

        // Allocating new memory
        let c = &a - b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() - &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a - b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_sub_int_broadcast() {
        let a = vector![1, 2, 3, 4, 5];
        let b = 2;

        let exp = vector![-1, 0, 1, 2, 3];

        // Allocating new memory
        let c = &a - &b;
        assert_eq!(c, exp);

        // Allocating new memory
        let c = &a - b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() - &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a - b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_sub_f32_elemwise() {
        let a = vector![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vector![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let exp = vector![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0];

        // Allocating new memory
        let c = &a - &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = &a - b.clone();
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() - &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a - b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_sub_int_elemwise() {
        let a = vector![10, 11, 12, 13, 14];
        let b = vector![2, 4, 6, 8, 10];

        let exp = vector![8, 7, 6, 5, 4];

        // Allocating new memory
        let c = &a - &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = &a - b.clone();
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() - &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a - b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_rem_f32_broadcast() {
        let a = vector![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = 2.0;

        let exp = vector![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        // Allocating new memory
        let c = &a % &b;
        assert_eq!(c, exp);

        // Allocating new memory
        let c = &a % b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() % &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a % b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_rem_int_broadcast() {
        let a = vector![1, 2, 3, 4, 5];
        let b = 3;

        let exp = vector![1, 2, 0, 1, 2];

        // Allocating new memory
        let c = &a % &b;
        assert_eq!(c, exp);

        // Allocating new memory
        let c = &a % b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() % &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a % b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_rem_f32_elemwise() {
        let a = vector![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vector![3.0, 3.0, 3.0, 4.0, 4.0, 4.0];

        let exp = vector![1.0, 2.0, 0.0, 0.0, 1.0, 2.0];

        // Allocating new memory
        let c = &a % &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = &a % b.clone();
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() % &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a % b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_rem_int_elemwise() {
        let a = vector![1, 2, 3, 4, 5];
        let b = vector![2, 2, 2, 3, 3];

        let exp = vector![1, 0, 1, 1, 2];

        // Allocating new memory
        let c = &a % &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = &a % b.clone();
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() % &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a % b;
        assert_eq!(c, exp);
    }

    /**********************
     *                     *
     *     Bitwise Ops     *
     *                     *
     **********************/

    #[test]
    fn vector_bitand_int_broadcast() {
        let a = vector![1, 2, 3, 4, 5];
        let b = 2;

        let exp = vector![1 & 2, 2 & 2, 3 & 2, 4 & 2, 5 & 2];

        // Allocating new memory
        let c = &a & &b;
        assert_eq!(c, exp);

        // Allocating new memory
        let c = &a & b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() & &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a & b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_bitand_bool_broadcast() {
        let a = vector![true, false, true];
        let b = true;

        let exp = vector![true, false, true];

        // Allocating new memory
        let c = &a & &b;
        assert_eq!(c, exp);

        // Allocating new memory
        let c = &a & b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() & &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a & b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_bitand_int_elemwise() {
        let a = vector![1, 2, 3, 4, 5, 6];
        let b = vector![2, 3, 4, 5, 6, 7];

        let exp = vector![1 & 2, 2 & 3, 3 & 4, 4 & 5, 5 & 6, 6 & 7];

        // Allocating new memory
        let c = &a & &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = &a & b.clone();
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() & &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a & b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_bitand_bool_elemwise() {
        let a = vector![true, true, false, false];
        let b = vector![true, false, true, false];

        let exp = vector![true, false, false, false];

        // Allocating new memory
        let c = &a & &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = &a & b.clone();
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() & &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a & b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_bitor_int_broadcast() {
        let a = vector![1, 2, 3, 4, 5];
        let b = 2;

        let exp = vector![1 | 2, 2 | 2, 3 | 2, 4 | 2, 5 | 2];

        // Allocating new memory
        let c = &a | &b;
        assert_eq!(c, exp);

        // Allocating new memory
        let c = &a | b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() | &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a | b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_bitor_bool_broadcast() {
        let a = vector![true, false, true];
        let b = true;

        let exp = vector![true, true, true];

        // Allocating new memory
        let c = &a | &b;
        assert_eq!(c, exp);

        // Allocating new memory
        let c = &a | b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() | &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a | b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_bitor_int_elemwise() {
        let a = vector![1, 2, 3, 4, 5, 6];
        let b = vector![2, 3, 4, 5, 6, 7];

        let exp = vector![1 | 2, 2 | 3, 3 | 4, 4 | 5, 5 | 6, 6 | 7];

        // Allocating new memory
        let c = &a | &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = &a | b.clone();
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() | &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a | b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_bitor_bool_elemwise() {
        let a = vector![true, true, false, false];
        let b = vector![true, false, true, false];

        let exp = vector![true, true, true, false];

        // Allocating new memory
        let c = &a | &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = &a | b.clone();
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() | &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a | b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_bitxor_int_broadcast() {
        let a = vector![1, 2, 3, 4, 5];
        let b = 2;

        let exp = vector![1 ^ 2, 2 ^ 2, 3 ^ 2, 4 ^ 2, 5 ^ 2];

        // Allocating new memory
        let c = &a ^ &b;
        assert_eq!(c, exp);

        // Allocating new memory
        let c = &a ^ b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() ^ &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a ^ b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_bitxor_bool_broadcast() {
        let a = vector![true, false, true];
        let b = true;

        let exp = vector![false, true, false];

        // Allocating new memory
        let c = &a ^ &b;
        assert_eq!(c, exp);

        // Allocating new memory
        let c = &a ^ b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() ^ &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a ^ b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_bitxor_int_elemwise() {
        let a = vector![1, 2, 3, 4, 5, 6];
        let b = vector![2, 3, 4, 5, 6, 7];

        let exp = vector![1 ^ 2, 2 ^ 3, 3 ^ 4, 4 ^ 5, 5 ^ 6, 6 ^ 7];

        // Allocating new memory
        let c = &a ^ &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = &a ^ b.clone();
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() ^ &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a ^ b;
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_bitxor_bool_elemwise() {
        let a = vector![true, true, false, false];
        let b = vector![true, false, true, false];

        let exp = vector![false, true, true, false];

        // Allocating new memory
        let c = &a ^ &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = &a ^ b.clone();
        assert_eq!(c, exp);

        // Reusing memory
        let c = a.clone() ^ &b;
        assert_eq!(c, exp);

        // Reusing memory
        let c = a ^ b;
        assert_eq!(c, exp);
    }
}