//! The vector module.
//!
//! Currently contains all code
//! relating to the vector linear algebra struct.

use std::ops::{Mul, Add, Div, Sub, Rem,
               MulAssign, AddAssign, DivAssign, SubAssign, RemAssign,
               Neg};
use utils;

use super::Vector;

impl<T: Copy + Mul<T, Output = T>> Vector<T> {
    /// The elementwise product of two vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::vector::Vector;
    ///
    /// let a = Vector::new(vec![1.0,2.0,3.0,4.0]);
    /// let b = Vector::new(vec![1.0,2.0,3.0,4.0]);
    ///
    /// let c = &a.elemul(&b);
    /// assert_eq!(*c.data(), vec![1.0, 4.0, 9.0, 16.0]);
    /// ```
    pub fn elemul(&self, v: &Vector<T>) -> Vector<T> {
        assert_eq!(self.size, v.size);
        Vector::new(utils::ele_mul(&self.data, &v.data))
    }
}

impl<T: Copy + Div<T, Output = T>> Vector<T> {
    /// The elementwise division of two vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::vector::Vector;
    ///
    /// let a = Vector::new(vec![1.0,2.0,3.0,4.0]);
    /// let b = Vector::new(vec![1.0,2.0,3.0,4.0]);
    ///
    /// let c = &a.elediv(&b);
    /// assert_eq!(*c.data(), vec![1.0; 4]);
    /// ```
    pub fn elediv(&self, v: &Vector<T>) -> Vector<T> {
        assert_eq!(self.size, v.size);
        Vector::new(utils::ele_div(&self.data, &v.data))
    }
}

macro_rules! impl_bin_op_scalar_vector (
    ($trt:ident, $op:ident, $sym:tt, $doc:expr) => (

/// Scalar
#[doc=$doc]
/// with Vector reusing current memory.
impl<T: Copy + $trt<T, Output = T>> $trt<T> for Vector<T> {
    type Output = Vector<T>;
    fn $op(self, f: T) -> Vector<T> {
        self $sym &f
    }
}

/// Scalar
#[doc=$doc]
/// with Vector reusing current memory.
impl<'a, T: Copy + $trt<T, Output = T>> $trt<&'a T> for Vector<T> {
    type Output = Vector<T>;
    fn $op(mut self, f: &T) -> Vector<T> {
        for val in &mut self.data {
            *val = *val $sym *f;
        }
        self
    }
}

/// Scalar
#[doc=$doc]
/// with Vector allocating new memory.
impl<'a, T: Copy + $trt<T, Output = T>> $trt<T> for &'a Vector<T> {
    type Output = Vector<T>;
    fn $op(self, f: T) -> Vector<T> {
        self $sym &f
    }
}

/// Scalar
#[doc=$doc]
/// with Vector allocating new memory.
impl<'a, 'b, T: Copy + $trt<T, Output = T>> $trt<&'b T> for &'a Vector<T> {
    type Output = Vector<T>;
    fn $op(self, f: &T) -> Vector<T> {
        let new_data = self.data.iter().map(|v| *v $sym *f).collect();
        Vector { size: self.size, data: new_data }
    }
}
    );
);
impl_bin_op_scalar_vector!(Add, add, +, "addition");
impl_bin_op_scalar_vector!(Mul, mul, *, "multiplication");
impl_bin_op_scalar_vector!(Sub, sub, -, "subtraction");
impl_bin_op_scalar_vector!(Div, div, /, "division");
impl_bin_op_scalar_vector!(Rem, rem, %, "remainder");

macro_rules! impl_bin_op_vector (
    ($trt:ident, $op:ident, $sym:tt, $doc:expr) => (

/// Vector
#[doc=$doc]
/// with Vector reusing current memory.
impl<T: Copy + $trt<T, Output = T>> $trt<Vector<T>> for Vector<T> {
    type Output = Vector<T>;
    fn $op(self, v: Vector<T>) -> Vector<T> {
        self $sym &v
    }
}

/// Vector addition with Vector reusing current memory.
impl<'a, T: Copy + $trt<T, Output = T>> $trt<&'a Vector<T>> for Vector<T> {
    type Output = Vector<T>;

    fn $op(mut self, v: &Vector<T>) -> Vector<T> {
        utils::in_place_vec_bin_op(&mut self.data, &v.data, |x, &y| *x = *x $sym y);
        self
    }
}

/// Vector
#[doc=$doc]
/// with Vector reusing current memory.
impl<'a, T: Copy + $trt<T, Output = T>> $trt<Vector<T>> for &'a Vector<T> {
    type Output = Vector<T>;

    fn $op(self, mut v: Vector<T>) -> Vector<T> {
        utils::in_place_vec_bin_op(&mut v.data, &self.data, |y, &x| *y = x $sym *y);
        v
    }
}

/// Vector
#[doc=$doc]
/// with Vector allocating new memory.
impl<'a, 'b, T: Copy + $trt<T, Output = T>> $trt<&'b Vector<T>> for &'a Vector<T> {
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
impl_bin_op_vector!(Add, add, +, "addition");
impl_bin_op_vector!(Sub, sub, -, "subtraction");
impl_bin_op_vector!(Rem, rem, %, "remainder");

/// Gets negative of vector.
impl<T: Neg<Output = T> + Copy> Neg for Vector<T> {
    type Output = Vector<T>;

    fn neg(mut self) -> Vector<T> {
        for val in &mut self.data {
            *val = -*val;
        }
        self
    }
}

/// Gets negative of vector.
impl<'a, T: Neg<Output = T> + Copy> Neg for &'a Vector<T> {
    type Output = Vector<T>;

    fn neg(self) -> Vector<T> {
        let new_data = self.data.iter().map(|v| -*v).collect::<Vec<_>>();
        Vector::new(new_data)
    }
}


macro_rules! impl_op_assign_vec_scalar (
    ($assign_trt:ident, $trt:ident, $op:ident, $op_assign:ident, $doc:expr) => (

/// Performs
#[doc=$doc]
/// assignment between a vector and a scalar.
impl<T : Copy + $trt<T, Output=T>> $assign_trt<T> for Vector<T> {
    fn $op_assign(&mut self, _rhs: T) {
        for x in &mut self.data {
            *x = (*x).$op(_rhs)
        }
    }
}

/// Performs
#[doc=$doc]
/// assignment between a vector and a scalar.
impl<'a, T : Copy + $trt<T, Output=T>> $assign_trt<&'a T> for Vector<T> {
    fn $op_assign(&mut self, _rhs: &T) {
        for x in &mut self.data {
            *x = (*x).$op(*_rhs)
        }
    }
}
    );
);

impl_op_assign_vec_scalar!(AddAssign, Add, add, add_assign, "addition");
impl_op_assign_vec_scalar!(SubAssign, Sub, sub, sub_assign, "subtraction");
impl_op_assign_vec_scalar!(DivAssign, Div, div, div_assign, "division");
impl_op_assign_vec_scalar!(MulAssign, Mul, mul, mul_assign, "multiplication");
impl_op_assign_vec_scalar!(RemAssign, Rem, rem, rem_assign, "reminder");

macro_rules! impl_op_assign_vec (
    ($assign_trt:ident, $trt:ident, $op:ident, $op_assign:ident, $doc:expr) => (

/// Performs elementwise
#[doc=$doc]
/// assignment between two vectors.
impl<T : Copy + $trt<T, Output=T>> $assign_trt<Vector<T>> for Vector<T> {
    fn $op_assign(&mut self, _rhs: Vector<T>) {
        utils::in_place_vec_bin_op(&mut self.data, &_rhs.data, |x, &y| {*x = (*x).$op(y) });
    }
}

/// Performs elementwise
#[doc=$doc]
/// assignment between two vectors.
impl<'a, T : Copy + $trt<T, Output=T>> $assign_trt<&'a Vector<T>> for Vector<T> {
    fn $op_assign(&mut self, _rhs: &Vector<T>) {
        utils::in_place_vec_bin_op(&mut self.data, &_rhs.data, |x, &y| {*x = (*x).$op(y) });
    }
}
    );
);

impl_op_assign_vec!(AddAssign, Add, add, add_assign, "addition");
impl_op_assign_vec!(SubAssign, Sub, sub, sub_assign, "subtraction");
impl_op_assign_vec!(RemAssign, Rem, rem, rem_assign, "remainder");


#[cfg(test)]
mod tests {
    use super::super::Vector;

    #[test]
    fn vector_mul_f32_broadcast() {
        let a = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = 3.0;

        // Allocating new memory
        let c = &a * &b;
        assert_eq!(c, Vector::new(vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0]));

        // Allocating new memory
        let c = &a * b;
        assert_eq!(c, Vector::new(vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0]));

        // Reusing memory
        let c = a.clone() * &b;
        assert_eq!(c, Vector::new(vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0]));

        // Reusing memory
        let c = a * b;
        assert_eq!(c, Vector::new(vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0]));
    }

    #[test]
    fn vector_mul_int_broadcast() {
        let a = Vector::new(vec![1, 2, 3, 4, 5]);
        let b = 2;

        // Allocating new memory
        let c = &a * &b;
        assert_eq!(c, Vector::new(vec![2, 4, 6, 8, 10]));

        // Allocating new memory
        let c = &a * b;
        assert_eq!(c, Vector::new(vec![2, 4, 6, 8, 10]));

        // Reusing memory
        let c = a.clone() * &b;
        assert_eq!(c, Vector::new(vec![2, 4, 6, 8, 10]));

        // Reusing memory
        let c = a * b;
        assert_eq!(c, Vector::new(vec![2, 4, 6, 8, 10]));
    }

    #[test]
    fn vector_mul_f32_elemwise() {
        let a = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Vector::new(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        // Allocating new memory
        let c = &a.elemul(&b);
        assert_eq!(c, &Vector::new(vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0]));

        // Allocating new memory
        let c = a.elemul(&b);
        assert_eq!(c, Vector::new(vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0]));
    }

    #[test]
    fn vector_mul_int_elemwise() {
        let a = Vector::new(vec![1, 2, 3, 4]);
        let b = Vector::new(vec![2, 4, 6, 8]);

        // Allocating new memory
        let c = &a.elemul(&b);
        assert_eq!(c, &Vector::new(vec![2, 8, 18, 32]));

        // Allocating new memory
        let c = a.elemul(&b);
        assert_eq!(c, Vector::new(vec![2, 8, 18, 32]));
    }

    #[test]
    fn vector_div_f32_broadcast() {
        let a = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = 3.0;

        // Allocating new memory
        let c = &a / &b;
        assert_eq!(c, Vector::new(vec![1. / 3., 2. / 3., 3. / 3., 4. / 3., 5. / 3., 6. / 3.]));

        // Allocating new memory
        let c = &a / b;
        assert_eq!(c, Vector::new(vec![1. / 3., 2. / 3., 3. / 3., 4. / 3., 5. / 3., 6. / 3.]));

        // Reusing memory
        let c = a.clone() / &b;
        assert_eq!(c, Vector::new(vec![1. / 3., 2. / 3., 3. / 3., 4. / 3., 5. / 3., 6. / 3.]));

        // Reusing memory
        let c = a / b;
        assert_eq!(c, Vector::new(vec![1. / 3., 2. / 3., 3. / 3., 4. / 3., 5. / 3., 6. / 3.]));
    }

    #[test]
    fn vector_div_int_broadcast() {
        let a = Vector::new(vec![1, 2, 3, 4, 5]);
        let b = 2;

        // Allocating new memory
        let c = &a / &b;
        assert_eq!(c, Vector::new(vec![0, 1, 1, 2, 2]));

        // Allocating new memory
        let c = &a / b;
        assert_eq!(c, Vector::new(vec![0, 1, 1, 2, 2]));

        // Reusing memory
        let c = a.clone() / &b;
        assert_eq!(c, Vector::new(vec![0, 1, 1, 2, 2]));

        // Reusing memory
        let c = a / b;
        assert_eq!(c, Vector::new(vec![0, 1, 1, 2, 2]));
    }

    #[test]
    fn vector_div_f32_elemwise() {
        let a = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Vector::new(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        // Allocating new memory
        let c = &a.elediv(&b);
        assert_eq!(c, &Vector::new(vec![1. / 2., 2. / 3., 3. / 4., 4. / 5., 5. / 6., 6. / 7.]));

        // Allocating new memory
        let c = a.elediv(&b);
        assert_eq!(c, Vector::new(vec![1. / 2., 2. / 3., 3. / 4., 4. / 5., 5. / 6., 6. / 7.]));
    }

    #[test]
    fn vector_div_int_elemwise() {
        let a = Vector::new(vec![2, 4, 6, 8]);
        let b = Vector::new(vec![2, 2, 3, 3]);

        // Allocating new memory
        let c = &a.elediv(&b);
        assert_eq!(c, &Vector::new(vec![1, 2, 2, 2]));

        // Allocating new memory
        let c = a.elediv(&b);
        assert_eq!(c, Vector::new(vec![1, 2, 2, 2]));
    }

    #[test]
    fn vector_add_f32_broadcast() {
        let a = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = 2.0;

        // Allocating new memory
        let c = &a + &b;
        assert_eq!(c, Vector::new(vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));

        // Allocating new memory
        let c = &a + b;
        assert_eq!(c, Vector::new(vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));

        // Reusing memory
        let c = a.clone() + &b;
        assert_eq!(c, Vector::new(vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));

        // Reusing memory
        let c = a + b;
        assert_eq!(c, Vector::new(vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
    }

    #[test]
    fn vector_add_int_broadcast() {
        let a = Vector::new(vec![1, 2, 3, 4, 5]);
        let b = 2;

        // Allocating new memory
        let c = &a + &b;
        assert_eq!(c, Vector::new(vec![3, 4, 5, 6, 7]));

        // Allocating new memory
        let c = &a + b;
        assert_eq!(c, Vector::new(vec![3, 4, 5, 6, 7]));

        // Reusing memory
        let c = a.clone() + &b;
        assert_eq!(c, Vector::new(vec![3, 4, 5, 6, 7]));

        // Reusing memory
        let c = a + b;
        assert_eq!(c, Vector::new(vec![3, 4, 5, 6, 7]));
    }

    #[test]
    fn vector_add_f32_elemwise() {
        let a = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Vector::new(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        // Allocating new memory
        let c = &a + &b;
        assert_eq!(c, Vector::new(vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0]));

        // Reusing memory
        let c = &a + b.clone();
        assert_eq!(c, Vector::new(vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0]));

        // Reusing memory
        let c = a.clone() + &b;
        assert_eq!(c, Vector::new(vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0]));

        // Reusing memory
        let c = a + b;
        assert_eq!(c, Vector::new(vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0]));
    }

    #[test]
    fn vector_add_int_elemwise() {
        let a = Vector::new(vec![1, 2, 3, 4, 5, 6]);
        let b = Vector::new(vec![2, 3, 4, 5, 6, 7]);

        // Allocating new memory
        let c = &a + &b;
        assert_eq!(c, Vector::new(vec![3, 5, 7, 9, 11, 13]));

        // Reusing memory
        let c = &a + b.clone();
        assert_eq!(c, Vector::new(vec![3, 5, 7, 9, 11, 13]));

        // Reusing memory
        let c = a.clone() + &b;
        assert_eq!(c, Vector::new(vec![3, 5, 7, 9, 11, 13]));

        // Reusing memory
        let c = a + b;
        assert_eq!(c, Vector::new(vec![3, 5, 7, 9, 11, 13]));
    }

    #[test]
    fn vector_sub_f32_broadcast() {
        let a = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = 2.0;

        // Allocating new memory
        let c = &a - &b;
        assert_eq!(c, Vector::new(vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]));

        // Allocating new memory
        let c = &a - b;
        assert_eq!(c, Vector::new(vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]));

        // Reusing memory
        let c = a.clone() - &b;
        assert_eq!(c, Vector::new(vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]));

        // Reusing memory
        let c = a - b;
        assert_eq!(c, Vector::new(vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn vector_sub_int_broadcast() {
        let a = Vector::new(vec![1, 2, 3, 4, 5]);
        let b = 2;

        // Allocating new memory
        let c = &a - &b;
        assert_eq!(c, Vector::new(vec![-1, 0, 1, 2, 3]));

        // Allocating new memory
        let c = &a - b;
        assert_eq!(c, Vector::new(vec![-1, 0, 1, 2, 3]));

        // Reusing memory
        let c = a.clone() - &b;
        assert_eq!(c, Vector::new(vec![-1, 0, 1, 2, 3]));

        // Reusing memory
        let c = a - b;
        assert_eq!(c, Vector::new(vec![-1, 0, 1, 2, 3]));
    }

    #[test]
    fn vector_sub_f32_elemwise() {
        let a = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Vector::new(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        // Allocating new memory
        let c = &a - &b;
        assert_eq!(c, Vector::new(vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]));

        // Reusing memory
        let c = &a - b.clone();
        assert_eq!(c, Vector::new(vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]));

        // Reusing memory
        let c = a.clone() - &b;
        assert_eq!(c, Vector::new(vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]));

        // Reusing memory
        let c = a - b;
        assert_eq!(c, Vector::new(vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]));
    }

    #[test]
    fn vector_sub_int_elemwise() {
        let a = Vector::new(vec![10, 11, 12, 13, 14]);
        let b = Vector::new(vec![2, 4, 6, 8, 10]);

        // Allocating new memory
        let c = &a - &b;
        assert_eq!(c, Vector::new(vec![8, 7, 6, 5, 4]));

        // Reusing memory
        let c = &a - b.clone();
        assert_eq!(c, Vector::new(vec![8, 7, 6, 5, 4]));

        // Reusing memory
        let c = a.clone() - &b;
        assert_eq!(c, Vector::new(vec![8, 7, 6, 5, 4]));

        // Reusing memory
        let c = a - b;
        assert_eq!(c, Vector::new(vec![8, 7, 6, 5, 4]));
    }

    #[test]
    fn vector_rem_f32_broadcast() {
        let a = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = 2.0;

        // Allocating new memory
        let c = &a % &b;
        assert_eq!(c, Vector::new(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]));

        // Allocating new memory
        let c = &a % b;
        assert_eq!(c, Vector::new(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]));

        // Reusing memory
        let c = a.clone() % &b;
        assert_eq!(c, Vector::new(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]));

        // Reusing memory
        let c = a % b;
        assert_eq!(c, Vector::new(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]));
    }

    #[test]
    fn vector_rem_int_broadcast() {
        let a = Vector::new(vec![1, 2, 3, 4, 5]);
        let b = 3;

        // Allocating new memory
        let c = &a % &b;
        assert_eq!(c, Vector::new(vec![1, 2, 0, 1, 2]));

        // Allocating new memory
        let c = &a % b;
        assert_eq!(c, Vector::new(vec![1, 2, 0, 1, 2]));

        // Reusing memory
        let c = a.clone() % &b;
        assert_eq!(c, Vector::new(vec![1, 2, 0, 1, 2]));

        // Reusing memory
        let c = a % b;
        assert_eq!(c, Vector::new(vec![1, 2, 0, 1, 2]));
    }

    #[test]
    fn vector_rem_f32_elemwise() {
        let a = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Vector::new(vec![3.0, 3.0, 3.0, 4.0, 4.0, 4.0]);

        // Allocating new memory
        let c = &a % &b;
        assert_eq!(c, Vector::new(vec![1.0, 2.0, 0.0, 0.0, 1.0, 2.0]));

        // Reusing memory
        let c = &a % b.clone();
        assert_eq!(c, Vector::new(vec![1.0, 2.0, 0.0, 0.0, 1.0, 2.0]));

        // Reusing memory
        let c = a.clone() % &b;
        assert_eq!(c, Vector::new(vec![1.0, 2.0, 0.0, 0.0, 1.0, 2.0]));

        // Reusing memory
        let c = a % b;
        assert_eq!(c, Vector::new(vec![1.0, 2.0, 0.0, 0.0, 1.0, 2.0]));
    }

    #[test]
    fn vector_rem_int_elemwise() {
        let a = Vector::new(vec![1, 2, 3, 4, 5]);
        let b = Vector::new(vec![2, 2, 2, 3, 3]);

        // Allocating new memory
        let c = &a % &b;
        assert_eq!(c, Vector::new(vec![1, 0, 1, 1, 2]));

        // Reusing memory
        let c = &a % b.clone();
        assert_eq!(c, Vector::new(vec![1, 0, 1, 1, 2]));

        // Reusing memory
        let c = a.clone() % &b;
        assert_eq!(c, Vector::new(vec![1, 0, 1, 1, 2]));

        // Reusing memory
        let c = a % b;
        assert_eq!(c, Vector::new(vec![1, 0, 1, 1, 2]));
    }

    // *****************************************************
    // Assignment
    // *****************************************************

    #[test]
    fn vector_add_assign_int_broadcast() {
        let mut a = Vector::new((0..9).collect::<Vec<_>>());

        a += &2;
        assert_eq!(a.into_vec(), (2..11).collect::<Vec<_>>());

        let mut a = Vector::new((0..9).collect::<Vec<_>>());

        a += 2;
        assert_eq!(a.into_vec(), (2..11).collect::<Vec<_>>());
    }

    #[test]
    fn vector_add_assign_int_elemwise() {
        let mut a = Vector::new((0..9).collect::<Vec<_>>());
        let b = Vector::new((0..9).collect::<Vec<_>>());

        a += &b;
        assert_eq!(a.into_vec(), (0..9).map(|x| 2 * x).collect::<Vec<_>>());

        let mut a = Vector::new((0..9).collect::<Vec<_>>());

        a += b;
        assert_eq!(a.into_vec(), (0..9).map(|x| 2 * x).collect::<Vec<_>>());
    }

    #[test]
    fn vector_sub_assign_int_broadcast() {
        let mut a = Vector::new((0..9).collect::<Vec<_>>());

        a -= &2;
        assert_eq!(a.into_vec(), (-2..7).collect::<Vec<_>>());

        let mut a = Vector::new((0..9).collect::<Vec<i32>>());
        a -= 2;
        assert_eq!(a.into_vec(), (-2..7).collect::<Vec<_>>());
    }

    #[test]
    fn vector_sub_assign_int_elemwise() {
        let mut a = Vector::new((0..9).collect::<Vec<_>>());
        let b = Vector::new((0..9).collect::<Vec<_>>());

        a -= &b;
        assert_eq!(a.into_vec(), vec![0; 9]);

        let mut a = Vector::new((0..9).collect::<Vec<_>>());

        a -= b;
        assert_eq!(a.into_vec(), vec![0; 9]);
    }

    #[test]
    fn vector_div_assign_f32_broadcast() {
        let a_data = vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let res_data = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
        let mut a = Vector::new(a_data.clone());

        a /= &2f32;
        assert_eq!(a.into_vec(), res_data.clone());

        let mut a = Vector::new(a_data.clone());
        a /= 2f32;
        assert_eq!(a.into_vec(), res_data.clone());
    }

    #[test]
    fn vector_mul_assign_f32_broadcast() {
        let a_data = vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let res_data = vec![2f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
        let mut a = Vector::new(a_data.clone());

        a *= &2f32;
        assert_eq!(a.into_vec(), res_data.clone());

        let mut a = Vector::new(a_data.clone());
        a *= 2f32;
        assert_eq!(a.into_vec(), res_data.clone());
    }

}
