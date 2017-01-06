use super::{Matrix, MatrixSlice, MatrixSliceMut};
use super::{Row, RowMut, Column, ColumnMut};
use super::{BaseMatrix, BaseMatrixMut};

use super::super::utils;
use super::super::vector::Vector;

use std::ops::{Mul, Add, Div, Sub, Index, IndexMut, Neg};
use std::ops::{MulAssign, AddAssign, SubAssign, DivAssign};
use libnum::Zero;

use matrix::PermutationMatrix;
use utils::Permutation;

/// Indexes matrix.
///
/// Takes row index first then column.
impl<T> Index<[usize; 2]> for Matrix<T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &T {
        assert!(idx[0] < self.rows,
                "Row index is greater than row dimension.");
        assert!(idx[1] < self.cols,
                "Column index is greater than column dimension.");

        unsafe { &*(self.get_unchecked(idx)) }
    }
}

macro_rules! impl_index_slice (
    ($slice_type:ident, $doc:expr) => (
/// Indexes
#[doc=$doc]
/// Takes row index first then column.
impl<'a, T> Index<[usize; 2]> for $slice_type<'a, T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &T {
        assert!(idx[0] < self.rows,
                "Row index is greater than row dimension.");
        assert!(idx[1] < self.cols,
                "Column index is greater than column dimension.");

        unsafe {
            &*(self.get_unchecked(idx))
        }
    }
}
    );
);

impl_index_slice!(MatrixSlice, "matrix slice.");
impl_index_slice!(MatrixSliceMut, "mutable matrix slice.");

/// Indexes mutable matrix slice.
///
/// Takes row index first then column.
impl<'a, T> IndexMut<[usize; 2]> for MatrixSliceMut<'a, T> {
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut T {
        assert!(idx[0] < self.rows,
                "Row index is greater than row dimension.");
        assert!(idx[1] < self.cols,
                "Column index is greater than column dimension.");

        unsafe { &mut *(self.ptr.offset((idx[0] * self.row_stride + idx[1]) as isize)) }
    }
}

/// Indexes mutable matrix.
///
/// Takes row index first then column.
impl<T> IndexMut<[usize; 2]> for Matrix<T> {
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut T {
        assert!(idx[0] < self.rows,
                "Row index is greater than row dimension.");
        assert!(idx[1] < self.cols,
                "Column index is greater than column dimension.");
        let self_cols = self.cols;
        unsafe { self.data.get_unchecked_mut(idx[0] * self_cols + idx[1]) }
    }
}

impl<'a, T> Index<usize> for Row<'a, T> {
    type Output = T;
    fn index(&self, idx: usize) -> &T {
        &self.row[[0, idx]]
    }
}

impl<'a, T> Index<usize> for RowMut<'a, T> {
    type Output = T;
    fn index(&self, idx: usize) -> &T {
        &self.row[[0, idx]]
    }
}

impl<'a, T> IndexMut<usize> for RowMut<'a, T> {
    fn index_mut(&mut self, idx: usize) -> &mut T {
        &mut self.row[[0, idx]]
    }
}

impl<'a, T> Index<usize> for Column<'a, T> {
    type Output = T;
    fn index(&self, idx: usize) -> &T {
        &self.col[[idx, 0]]
    }
}

impl<'a, T> Index<usize> for ColumnMut<'a, T> {
    type Output = T;
    fn index(&self, idx: usize) -> &T {
        &self.col[[idx, 0]]
    }
}

impl<'a, T> IndexMut<usize> for ColumnMut<'a, T> {
    fn index_mut(&mut self, idx: usize) -> &mut T {
        &mut self.col[[idx, 0]]
    }
}

macro_rules! impl_bin_op_scalar_slice (
    ($trt:ident, $op:ident, $slice:ident, $doc:expr) => (

/// Scalar
#[doc=$doc]
/// with matrix slice.
impl<'a, T> $trt<T> for $slice<'a, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, f: T) -> Matrix<T> {
        (&self).$op(&f)
    }
}

/// Scalar
#[doc=$doc]
/// with matrix slice.
impl<'a, 'b, T> $trt<&'b T> for $slice<'a, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, f: &T) -> Matrix<T> {
        (&self).$op(f)
    }
}

/// Scalar
#[doc=$doc]
/// with matrix slice.
impl<'a, 'b, T> $trt<T> for &'b $slice<'a, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, f: T) -> Matrix<T> {
        (&self).$op(&f)
    }
}

/// Scalar
#[doc=$doc]
/// with matrix slice.
impl<'a, 'b, 'c, T> $trt<&'c T> for &'b $slice<'a, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, f: &T) -> Matrix<T> {
        let new_data: Vec<T> = self.iter().map(|v| (*v).$op(*f)).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}
    );
);

impl_bin_op_scalar_slice!(Mul, mul, MatrixSlice, "multiplication");
impl_bin_op_scalar_slice!(Mul, mul, MatrixSliceMut, "multiplication");
impl_bin_op_scalar_slice!(Div, div, MatrixSlice, "division");
impl_bin_op_scalar_slice!(Div, div, MatrixSliceMut, "division");
impl_bin_op_scalar_slice!(Add, add, MatrixSlice, "addition");
impl_bin_op_scalar_slice!(Add, add, MatrixSliceMut, "addition");
impl_bin_op_scalar_slice!(Sub, sub, MatrixSlice, "subtraction");
impl_bin_op_scalar_slice!(Sub, sub, MatrixSliceMut, "subtraction");

macro_rules! impl_bin_op_scalar_matrix (
    ($trt:ident, $op:ident, $doc:expr) => (

/// Scalar
#[doc=$doc]
/// with matrix.
///
/// Will reuse the memory allocated for the existing matrix.
impl<T> $trt<T> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, f: T) -> Matrix<T> {
        (self).$op(&f)
    }
}


/// Scalar
#[doc=$doc]
/// with matrix.
///
/// Will reuse the memory allocated for the existing matrix.
impl<'a, T> $trt<&'a T> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(mut self, f: &T) -> Matrix<T> {
        for val in &mut self.data {
        	*val = (*val).$op(*f)
        }

        self
    }
}


/// Scalar
#[doc=$doc]
/// with matrix.
impl<'a, T> $trt<T> for &'a Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, f: T) -> Matrix<T> {
        (&self).$op(&f)
    }
}


/// Scalar
#[doc=$doc]
/// with matrix.
impl<'a, 'b, T> $trt<&'b T> for &'a Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, f: &T) -> Matrix<T> {
        let new_data: Vec<T> = self.data.iter().map(|v| (*v).$op(*f)).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}
    );
);

impl_bin_op_scalar_matrix!(Add, add, "addition");
impl_bin_op_scalar_matrix!(Mul, mul, "multiplication");
impl_bin_op_scalar_matrix!(Sub, sub, "subtraction");
impl_bin_op_scalar_matrix!(Div, div, "division");

/// Multiplies matrix by vector.
impl<T> Mul<Vector<T>> for Matrix<T>
    where T: Copy + Zero + Mul<T, Output = T> + Add<T, Output = T>
{
    type Output = Vector<T>;

    fn mul(self, m: Vector<T>) -> Vector<T> {
        (&self) * (&m)
    }
}

/// Multiplies matrix by vector.
impl<'a, T> Mul<Vector<T>> for &'a Matrix<T>
    where T: Copy + Zero + Mul<T, Output = T> + Add<T, Output = T>
{
    type Output = Vector<T>;

    fn mul(self, m: Vector<T>) -> Vector<T> {
        self * (&m)
    }
}

/// Multiplies matrix by vector.
impl<'a, T> Mul<&'a Vector<T>> for Matrix<T>
    where T: Copy + Zero + Mul<T, Output = T> + Add<T, Output = T>
{
    type Output = Vector<T>;

    fn mul(self, m: &Vector<T>) -> Vector<T> {
        (&self) * m
    }
}

/// Multiplies matrix by vector.
impl<'a, 'b, T> Mul<&'b Vector<T>> for &'a Matrix<T>
    where T: Copy + Zero + Mul<T, Output = T> + Add<T, Output = T>
{
    type Output = Vector<T>;

    fn mul(self, v: &Vector<T>) -> Vector<T> {
        assert!(v.size() == self.cols, "Matrix and Vector dimensions do not agree.");

        let mut new_data = Vec::with_capacity(self.rows);

        for i in 0..self.rows {
            new_data.push(utils::dot(&self.data[i * self.cols..(i + 1) * self.cols], v.data()));
        }

        Vector::new(new_data)
    }
}

macro_rules! impl_bin_op_slice (
    ($trt:ident, $op:ident, $slice_1:ident, $slice_2:ident, $doc:expr) => (

/// Performs elementwise
#[doc=$doc]
/// between the slices.
impl<'a, 'b, T> $trt<$slice_2<'b, T>> for $slice_1<'a, T>
    where T : Copy + $trt<T, Output=T>
{
    type Output = Matrix<T>;

    fn $op(self, s: $slice_2<T>) -> Matrix<T> {
        (&self).$op(&s)
    }
}

/// Performs elementwise
#[doc=$doc]
/// between the slices.
impl<'a, 'b, 'c, T> $trt<$slice_2<'b, T>> for &'c $slice_1<'a, T>
    where T : Copy + $trt<T, Output=T>
{
    type Output = Matrix<T>;

    fn $op(self, s: $slice_2<T>) -> Matrix<T> {
        (self).$op(&s)
    }
}

/// Performs elementwise
#[doc=$doc]
/// between the slices.
impl<'a, 'b, 'c, T> $trt<&'c $slice_2<'b, T>> for $slice_1<'a, T>
    where T : Copy + $trt<T, Output=T>
{
    type Output = Matrix<T>;

    fn $op(self, s: &$slice_2<T>) -> Matrix<T> {
        (&self).$op(s)
    }
}

/// Performs elementwise
#[doc=$doc]
/// between the slices.
impl<'a, 'b, 'c, 'd, T> $trt<&'d $slice_2<'b, T>> for &'c $slice_1<'a, T>
    where T : Copy + $trt<T, Output=T>
{
    type Output = Matrix<T>;

    fn $op(self, s: &$slice_2<T>) -> Matrix<T> {
        assert!(self.cols == s.cols, "Column dimensions do not agree.");
        assert!(self.rows == s.rows, "Row dimensions do not agree.");

        let mut res_data : Vec<T> = self.iter().cloned().collect();
        let s_data : Vec<T> = s.iter().cloned().collect();

        utils::in_place_vec_bin_op(&mut res_data, &s_data, |x, &y| { *x = (*x).$op(y) });

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: res_data,
        }
    }
}
    );
);

impl_bin_op_slice!(Add, add, MatrixSlice, MatrixSlice, "addition");
impl_bin_op_slice!(Add, add, MatrixSliceMut, MatrixSlice, "addition");
impl_bin_op_slice!(Add, add, MatrixSlice, MatrixSliceMut, "addition");
impl_bin_op_slice!(Add, add, MatrixSliceMut, MatrixSliceMut, "addition");

impl_bin_op_slice!(Sub, sub, MatrixSlice, MatrixSlice, "subtraction");
impl_bin_op_slice!(Sub, sub, MatrixSliceMut, MatrixSlice, "subtraction");
impl_bin_op_slice!(Sub, sub, MatrixSlice, MatrixSliceMut, "subtraction");
impl_bin_op_slice!(Sub, sub, MatrixSliceMut, MatrixSliceMut, "subtraction");

macro_rules! impl_bin_op_mat_slice (
    ($trt:ident, $op:ident, $slice:ident, $doc:expr) => (

/// Performs elementwise
#[doc=$doc]
/// between `Matrix` and `MatrixSlice`.
impl<'a, T> $trt<Matrix<T>> for $slice<'a, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, m: Matrix<T>) -> Matrix<T> {
        (&self).$op(&m)
    }
}

/// Performs elementwise
#[doc=$doc]
/// between `Matrix` and `MatrixSlice`.
impl<'a, 'b, T> $trt<Matrix<T>> for &'b $slice<'a, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, m: Matrix<T>) -> Matrix<T> {
        self.$op(&m)
    }
}

/// Performs elementwise
#[doc=$doc]
/// between `Matrix` and `MatrixSlice`.
impl<'a, 'b, T> $trt<&'b Matrix<T>> for $slice<'a, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, m: &Matrix<T>) -> Matrix<T> {
        (&self).$op(m)
    }
}

/// Performs elementwise
#[doc=$doc]
/// between `Matrix` and `MatrixSlice`.
impl<'a, 'b, 'c, T> $trt<&'c Matrix<T>> for &'b $slice<'a, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, m: &Matrix<T>) -> Matrix<T> {
        assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        let mut new_data : Vec<T> = self.iter().cloned().collect();
        utils::in_place_vec_bin_op(&mut new_data, &m.data(), |x, &y| { *x = (*x).$op(y) });

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Performs elementwise
#[doc=$doc]
/// between `Matrix` and `MatrixSlice`.
impl<'a, T> $trt<$slice<'a, T>> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, s: $slice<T>) -> Matrix<T> {
        (&self).$op(s)
    }
}

/// Performs elementwise
#[doc=$doc]
/// between `Matrix` and `MatrixSlice`.
impl<'a, 'b, T> $trt<$slice<'a, T>> for &'b Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, s: $slice<T>) -> Matrix<T> {
        self.$op(&s)
    }
}

/// Performs elementwise
#[doc=$doc]
/// between `Matrix` and `MatrixSlice`.
impl<'a, 'b, T> $trt<&'b $slice<'a, T>> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, s: &$slice<T>) -> Matrix<T> {
        (&self).$op(s)
    }
}

/// Performs elementwise
#[doc=$doc]
/// between `Matrix` and `MatrixSlice`.
impl<'a, 'b, 'c, T> $trt<&'c $slice<'a, T>> for &'b Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, s: &$slice<T>) -> Matrix<T> {
        assert!(self.cols == s.cols, "Column dimensions do not agree.");
        assert!(self.rows == s.rows, "Row dimensions do not agree.");

        let mut new_data : Vec<T> = s.iter().cloned().collect();
        utils::in_place_vec_bin_op(&mut new_data, self.data(), |x, &y| { *x = (y).$op(*x) });

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}
    );
);

impl_bin_op_mat_slice!(Add, add, MatrixSlice, "addition");
impl_bin_op_mat_slice!(Add, add, MatrixSliceMut, "addition");

impl_bin_op_mat_slice!(Sub, sub, MatrixSlice, "subtraction");
impl_bin_op_mat_slice!(Sub, sub, MatrixSliceMut, "subtraction");

macro_rules! impl_bin_op_mat (
    ($trt:ident, $op:ident, $doc:expr) => (

/// Performs elementwise
#[doc=$doc]
/// between two matrices.
///
/// This will reuse allocated memory from `self`.
impl<T> $trt<Matrix<T>> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, m: Matrix<T>) -> Matrix<T> {
        self.$op(&m)
    }
}

/// Performs elementwise
#[doc=$doc]
/// between two matrices.
///
/// This will reuse allocated memory from `m`.
impl<'a, T> $trt<Matrix<T>> for &'a Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, mut m: Matrix<T>) -> Matrix<T> {
    	assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        utils::in_place_vec_bin_op(&mut m.data, &self.data, |x,&y| {*x = (y).$op(*x)});
        m
    }
}

/// Performs elementwise
#[doc=$doc]
/// between two matrices.
///
/// This will reuse allocated memory from `self`.
impl<'a, T> $trt<&'a Matrix<T>> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(mut self, m: &Matrix<T>) -> Matrix<T> {
    	assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        utils::in_place_vec_bin_op(&mut self.data, &m.data, |x,&y| {*x = (*x).$op(y)});
        self
    }
}

/// Performs elementwise
#[doc=$doc]
/// between two matrices.
impl<'a, 'b, T> $trt<&'b Matrix<T>> for &'a Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, m: &Matrix<T>) -> Matrix<T> {
        assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        let new_data = utils::vec_bin_op(&self.data, &m.data, |x, y| { x.$op(y) });

        Matrix {
        	rows: self.rows,
        	cols: self.cols,
        	data: new_data,
        }
    }
}
    );
);

impl_bin_op_mat!(Add, add, "addition");
impl_bin_op_mat!(Sub, sub, "subtraction");

macro_rules! impl_op_assign_mat_scalar (
    ($assign_trt:ident, $trt:ident, $op:ident, $op_assign:ident, $doc:expr) => (

/// Performs
#[doc=$doc]
/// assignment between a matrix and a scalar.
impl<T> $assign_trt<T> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: T) {
        for x in &mut self.data {
            *x = (*x).$op(_rhs)
        }
    }
}

/// Performs
#[doc=$doc]
/// assignment between a matrix and a scalar.
impl<'a, T> $assign_trt<&'a T> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: &T) {
        for x in &mut self.data {
            *x = (*x).$op(*_rhs)
        }
    }
}
    );
);

impl_op_assign_mat_scalar!(AddAssign, Add, add, add_assign, "addition");
impl_op_assign_mat_scalar!(SubAssign, Sub, sub, sub_assign, "subtraction");
impl_op_assign_mat_scalar!(DivAssign, Div, div, div_assign, "division");
impl_op_assign_mat_scalar!(MulAssign, Mul, mul, mul_assign, "multiplication");

macro_rules! impl_op_assign_slice_scalar (
    ($assign_trt:ident, $trt:ident, $op:ident, $op_assign:ident, $doc:expr) => (

/// Performs
#[doc=$doc]
/// assignment between a mutable matrix slice and a scalar.
impl<'a, T> $assign_trt<T> for MatrixSliceMut<'a, T>
    where T: Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: T) {
        for x in self.iter_mut() {
            *x = (*x).$op(_rhs)
        }
    }
}

/// Performs
#[doc=$doc]
/// assignment between a mutable matrix slice and a scalar.
impl<'a, 'b, T> $assign_trt<&'b T> for MatrixSliceMut<'a, T>
    where T: Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: &T) {
        for x in self.iter_mut() {
            *x = (*x).$op(*_rhs)
        }
    }
}
    );
);

impl_op_assign_slice_scalar!(AddAssign, Add, add, add_assign, "addition");
impl_op_assign_slice_scalar!(SubAssign, Sub, sub, sub_assign, "subtraction");
impl_op_assign_slice_scalar!(DivAssign, Div, div, div_assign, "division");
impl_op_assign_slice_scalar!(MulAssign, Mul, mul, mul_assign, "multiplication");

macro_rules! impl_op_assign_mat (
    ($assign_trt:ident, $trt:ident, $op:ident, $op_assign:ident, $doc:expr) => (

/// Performs elementwise
#[doc=$doc]
/// assignment between two matrices.
impl<T> $assign_trt<Matrix<T>> for Matrix<T>
    where T: Copy + $trt<T, Output=T>  {
    fn $op_assign(&mut self, _rhs: Matrix<T>) {
        utils::in_place_vec_bin_op(&mut self.data, &_rhs.data, |x, &y| {*x = (*x).$op(y) });
    }
}

/// Performs elementwise
#[doc=$doc]
/// assignment between two matrices.
impl<'a, T> $assign_trt<&'a Matrix<T>> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: &Matrix<T>) {
        utils::in_place_vec_bin_op(&mut self.data, &_rhs.data, |x, &y| {*x = (*x).$op(y) });
    }
}
    );
);

impl_op_assign_mat!(AddAssign, Add, add, add_assign, "addition");
impl_op_assign_mat!(SubAssign, Sub, sub, sub_assign, "subtraction");

macro_rules! impl_op_assign_slice_mat (
    ($assign_trt:ident, $trt:ident, $op:ident, $op_assign:ident, $doc:expr) => (

/// Performs elementwise
#[doc=$doc]
/// assignment between two matrices.
impl<'a, T> $assign_trt<Matrix<T>> for MatrixSliceMut<'a, T>
    where T: Copy + $trt<T, Output=T>
{
    fn $op_assign(&mut self, _rhs: Matrix<T>) {
        for (mut slice_row, target_row) in self.row_iter_mut().zip(_rhs.row_iter()) {
            utils::in_place_vec_bin_op(slice_row.raw_slice_mut(),
                                        target_row.raw_slice(),
                                        |x, &y| {*x = (*x).$op(y) });
        }
    }
}

/// Performs elementwise
#[doc=$doc]
/// assignment between two matrices.
impl<'a, 'b, T> $assign_trt<&'b Matrix<T>> for MatrixSliceMut<'a, T>
    where T: Copy + $trt<T, Output=T>
{
    fn $op_assign(&mut self, _rhs: &Matrix<T>) {
        for (mut slice_row, target_row) in self.row_iter_mut()
                                        .zip(_rhs.row_iter()) {
            utils::in_place_vec_bin_op(slice_row.raw_slice_mut(),
                                        target_row.raw_slice(),
                                        |x, &y| {*x = (*x).$op(y) });
        }
    }
}
    );
);

impl_op_assign_slice_mat!(AddAssign, Add, add, add_assign, "addition");
impl_op_assign_slice_mat!(SubAssign, Sub, sub, sub_assign, "subtraction");

macro_rules! impl_op_assign_slice (
    ($target_slice:ident, $assign_trt:ident,
        $trt:ident, $op:ident,
        $op_assign:ident, $doc:expr) => (

/// Performs elementwise
#[doc=$doc]
/// assignment between two matrices.
impl<'a, 'b, T> $assign_trt<$target_slice<'b, T>> for MatrixSliceMut<'a, T>
    where T: Copy + $trt<T, Output=T>
{
    fn $op_assign(&mut self, _rhs: $target_slice<T>) {
        for (mut slice_row, target_row) in self.row_iter_mut()
                                            .zip(_rhs.row_iter()) {
            utils::in_place_vec_bin_op(slice_row.raw_slice_mut(),
                                        target_row.raw_slice(),
                                        |x, &y| {*x = (*x).$op(y) });
        }
    }
}

/// Performs elementwise
#[doc=$doc]
/// assignment between two matrices.
impl<'a, 'b, 'c, T> $assign_trt<&'c $target_slice<'b, T>> for MatrixSliceMut<'a, T>
    where T: Copy + $trt<T, Output=T>
{
    fn $op_assign(&mut self, _rhs: &$target_slice<T>) {
        for (mut slice_row, target_row) in self.row_iter_mut()
                                            .zip(_rhs.row_iter()) {
            utils::in_place_vec_bin_op(slice_row.raw_slice_mut(),
                                        target_row.raw_slice(),
                                        |x, &y| {*x = (*x).$op(y) });
        }
    }
}
    );
);

impl_op_assign_slice!(MatrixSlice, AddAssign, Add, add, add_assign, "addition");
impl_op_assign_slice!(MatrixSlice, SubAssign, Sub, sub, sub_assign, "subtraction");
impl_op_assign_slice!(MatrixSliceMut, AddAssign, Add, add, add_assign, "addition");
impl_op_assign_slice!(MatrixSliceMut, SubAssign, Sub, sub, sub_assign, "subtraction");

macro_rules! impl_op_assign_mat_slice (
    ($target_mat:ident, $assign_trt:ident, $trt:ident, $op:ident, $op_assign:ident, $doc:expr) => (

/// Performs elementwise
#[doc=$doc]
/// assignment between two matrices.
impl<'a, T> $assign_trt<$target_mat<'a, T>> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: $target_mat<T>) {
        for (mut slice_row, target_row) in self.row_iter_mut().zip(_rhs.row_iter()) {
            utils::in_place_vec_bin_op(slice_row.raw_slice_mut(),
                                        target_row.raw_slice(),
                                        |x, &y| {*x = (*x).$op(y) });
        }
    }
}

/// Performs elementwise
#[doc=$doc]
/// assignment between two matrices.
impl<'a, 'b, T> $assign_trt<&'b $target_mat<'a, T>> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: &$target_mat<T>) {
        for (mut slice_row, target_row) in self.row_iter_mut().zip(_rhs.row_iter()) {
            utils::in_place_vec_bin_op(slice_row.raw_slice_mut(),
                                        target_row.raw_slice(),
                                        |x, &y| {*x = (*x).$op(y) });
        }
    }
}
    );
);

impl_op_assign_mat_slice!(MatrixSlice, AddAssign, Add, add, add_assign, "addition");
impl_op_assign_mat_slice!(MatrixSlice, SubAssign, Sub, sub, sub_assign, "subtraction");
impl_op_assign_mat_slice!(MatrixSliceMut, AddAssign, Add, add, add_assign, "addition");
impl_op_assign_mat_slice!(MatrixSliceMut, SubAssign, Sub, sub, sub_assign, "subtraction");

macro_rules! impl_neg_slice (
    ($slice:ident) => (

/// Gets negative of matrix slice.
impl<'a, T> Neg for $slice<'a, T>
    where T: Neg<Output = T> + Copy {
    type Output = Matrix<T>;

    fn neg(self) -> Matrix<T> {
        - &self
    }
}

/// Gets negative of matrix slice.
impl<'a, 'b, T> Neg for &'b $slice<'a, T>
    where T: Neg<Output = T> + Copy {
    type Output = Matrix<T>;

    fn neg(self) -> Matrix<T> {
        let new_data = self.iter().map(|v| -*v).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

    );
);

impl_neg_slice!(MatrixSlice);
impl_neg_slice!(MatrixSliceMut);

/// Gets negative of matrix.
impl<T> Neg for Matrix<T>
    where T: Neg<Output = T> + Copy
{
    type Output = Matrix<T>;

    fn neg(mut self) -> Matrix<T> {
        for val in &mut self.data {
            *val = -*val;
        }

        self
    }
}

/// Gets negative of matrix.
impl<'a, T> Neg for &'a Matrix<T>
    where T: Neg<Output = T> + Copy
{
    type Output = Matrix<T>;

    fn neg(self) -> Matrix<T> {
        let new_data = self.data.iter().map(|v| -*v).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Multiplication of a permutation matrix and a vector.
///
/// # Complexity
/// Given a vector of size *n* and a permutation matrix of
/// dimensions *n* x *n*:
///
/// - O(1) memory usage
/// - O(*n*) memory accesses
impl<T> Mul<Vector<T>> for PermutationMatrix<T> {
    type Output = Vector<T>;

    fn mul(self, mut rhs: Vector<T>) -> Vector<T> {
        assert!(rhs.size() == self.dim(),
            "Permutation matrix and Vector dimensions are not compatible.");
        let permutation: Permutation = self.into();
        permutation.permute_by_swap(|i, j| rhs.mut_data().swap(i, j));
        rhs
    }
}

/// Multiplication of a permutation matrix and a vector.
///
/// # Complexity
/// Given a vector of size *n* and a permutation matrix of
/// dimensions *n* x *n*:
///
/// - O(*n*) memory usage
/// - O(*n*) memory accesses
impl<'a, T> Mul<Vector<T>> for &'a PermutationMatrix<T> where T: Clone + Zero {
    type Output = Vector<T>;

    fn mul(self, rhs: Vector<T>) -> Vector<T> {
        // Here we have the choice of using `permute_by_copy`
        // `permute_by_swap`, as we can reuse one of the existing
        // implementations.
        self * &rhs
    }
}

/// Multiplication of a permutation matrix and a vector.
///
/// # Complexity
/// Given a vector of size *n* and a permutation matrix of
/// dimensions *n* x *n*:
///
/// - O(*n*) memory usage
/// - O(*n*) memory accesses
impl<'a, 'b, T> Mul<&'a Vector<T>> for &'b PermutationMatrix<T> where T: Clone + Zero {
    type Output = Vector<T>;

    fn mul(self, rhs: &'a Vector<T>) -> Vector<T> {
        assert!(rhs.size() == self.dim(),
            "Permutation matrix and Vector dimensions are not compatible.");

        let permutation: &Permutation = self.into();
        let mut permuted_rhs = Vector::zeros(rhs.size());
        permutation.permute_by_copy(|i, j| permuted_rhs[j] = rhs[i].to_owned());
        permuted_rhs
    }
}

/// Multiplication of a permutation matrix and a vector.
///
/// # Complexity
/// Given a vector of size *n* and a permutation matrix of
/// dimensions *n* x *n*:
///
/// - O(*n*) memory usage
/// - O(*n*) memory accesses
impl<'a, T> Mul<&'a Vector<T>> for PermutationMatrix<T> where T: Clone + Zero {
    type Output = Vector<T>;

    fn mul(self, rhs: &'a Vector<T>) -> Vector<T> {
        &self * rhs
    }
}

fn validate_permutation_left_mul_dimensions<T, M>(p: &PermutationMatrix<T>, rhs: &M)
    where M: BaseMatrix<T> {
     assert!(p.dim() == rhs.rows(),
            "Permutation matrix and right-hand side matrix dimensions
             are not compatible.");
}

impl<T> Mul<Matrix<T>> for PermutationMatrix<T> {
    type Output = Matrix<T>;

    fn mul(self, mut rhs: Matrix<T>) -> Matrix<T> {
        validate_permutation_left_mul_dimensions(&self, &rhs);
        let permutation: Permutation = self.into();
        permutation.permute_by_swap(|i, j| rhs.swap_rows(i, j));
        rhs
    }
}

impl<'b, T> Mul<Matrix<T>> for &'b PermutationMatrix<T> {
    type Output = Matrix<T>;

    fn mul(self, mut rhs: Matrix<T>) -> Matrix<T> {
        validate_permutation_left_mul_dimensions(self, &rhs);
        let permutation: &Permutation = self.into();
        permutation.clone().permute_by_swap(|i, j| rhs.swap_rows(i, j));
        rhs
    }
}

macro_rules! impl_permutation_matrix_left_multiply_reference_type {
    ($MatrixType:ty) => (

impl<'a, 'm, T> Mul<&'a $MatrixType> for PermutationMatrix<T> where T: Zero + Clone {
    type Output = Matrix<T>;

    fn mul(self, rhs: &'a $MatrixType) -> Matrix<T> {
        validate_permutation_left_mul_dimensions(&self, rhs);
        let permutation: Permutation = self.into();
        let mut permuted_matrix = Matrix::zeros(rhs.rows(), rhs.cols());
        {
            let copy_row = |i, j| permuted_matrix.row_mut(j)
                                             .raw_slice_mut()
                                             .clone_from_slice(rhs.row(i).raw_slice());
            permutation.permute_by_copy(copy_row);
        }
        permuted_matrix
    }
}

impl<'a, 'b, 'm, T> Mul<&'a $MatrixType> for &'b PermutationMatrix<T> where T: Zero + Clone {
    type Output = Matrix<T>;

    fn mul(self, rhs: &'a $MatrixType) -> Matrix<T> {
        validate_permutation_left_mul_dimensions(self, rhs);
        let permutation: &Permutation = self.into();
        let mut permuted_matrix = Matrix::zeros(rhs.rows(), rhs.cols());
        {
            let copy_row = |i, j| permuted_matrix.row_mut(j)
                                             .raw_slice_mut()
                                             .clone_from_slice(rhs.row(i).raw_slice());
            permutation.permute_by_copy(copy_row);
        }
        permuted_matrix
    }
}

    )
}

impl_permutation_matrix_left_multiply_reference_type!(Matrix<T>);
impl_permutation_matrix_left_multiply_reference_type!(MatrixSlice<'m, T>);
impl_permutation_matrix_left_multiply_reference_type!(MatrixSliceMut<'m, T>);

fn validate_permutation_right_mul_dimensions<T, M>(lhs: &M, p: &PermutationMatrix<T>)
    where M: BaseMatrix<T> {
     assert!(lhs.cols() == p.dim(),
            "Left-hand side matrix and permutation matrix dimensions
             are not compatible.");
}

impl<T> Mul<PermutationMatrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(mut self, rhs: PermutationMatrix<T>) -> Matrix<T> {
        validate_permutation_right_mul_dimensions(&self, &rhs);
        let permutation: Permutation = rhs.into();
        permutation.permute_by_swap(|i, j| self.swap_cols(i, j));
        self
    }
}

impl<'a, T> Mul<&'a PermutationMatrix<T>> for Matrix<T> where T: Clone {
    type Output = Matrix<T>;

    fn mul(mut self, rhs: &'a PermutationMatrix<T>) -> Matrix<T> {
        validate_permutation_right_mul_dimensions(&self, &rhs);
        let permutation: Permutation = rhs.clone().into();
        permutation.permute_by_swap(|i, j| self.swap_cols(i, j));
        self
    }
}

macro_rules! impl_permutation_matrix_right_multiply_reference_type {
    ($MatrixType:ty) => (

impl<'a, 'm, T> Mul<PermutationMatrix<T>> for &'a $MatrixType where T: Zero + Clone {
    type Output = Matrix<T>;

    fn mul(self, rhs: PermutationMatrix<T>) -> Matrix<T> {
        validate_permutation_right_mul_dimensions(self, &rhs);
        let permutation: Permutation = rhs.into();
        let mut permuted_matrix = Matrix::zeros(self.rows(), self.cols());
        // Permute columns in one row at a time for (presumably) better cache performance
        for (index, source_row) in self.row_iter()
                                       .map(|r| r.raw_slice())
                                       .enumerate() {
            let target_row = permuted_matrix.row_mut(index).raw_slice_mut();
            permutation.permute_by_copy(|i, j| target_row[j] = source_row[i].clone());
        }
        permuted_matrix
    }
}

impl<'a, 'b, 'm, T> Mul<&'b PermutationMatrix<T>> for &'a $MatrixType where T: Zero + Clone {
    type Output = Matrix<T>;

    fn mul(self, rhs: &'b PermutationMatrix<T>) -> Matrix<T> {
        validate_permutation_right_mul_dimensions(self, &rhs);
        let permutation: &Permutation = rhs.into();
        let mut permuted_matrix = Matrix::zeros(self.rows(), self.cols());
        // Permute columns in one row at a time for (presumably) better cache performance
        for (index, source_row) in self.row_iter()
                                       .map(|r| r.raw_slice())
                                       .enumerate() {
            let target_row = permuted_matrix.row_mut(index).raw_slice_mut();
            permutation.permute_by_copy(|i, j| target_row[j] = source_row[i].clone());
        }
        permuted_matrix
    }
}

    )
}

impl_permutation_matrix_right_multiply_reference_type!(Matrix<T>);
impl_permutation_matrix_right_multiply_reference_type!(MatrixSlice<'m, T>);
impl_permutation_matrix_right_multiply_reference_type!(MatrixSliceMut<'m, T>);

#[cfg(test)]
mod tests {

    use super::super::Matrix;
    use super::super::MatrixSlice;
    use super::super::MatrixSliceMut;

    use matrix::{BaseMatrix, BaseMatrixMut};
    use matrix::PermutationMatrix;

    #[test]
    fn indexing_mat() {
        let a = matrix![1., 2.;
                        3., 4.;
                        5., 6.];

        assert_eq!(a[[0, 0]], 1.0);
        assert_eq!(a[[0, 1]], 2.0);
        assert_eq!(a[[1, 0]], 3.0);
        assert_eq!(a[[1, 1]], 4.0);
        assert_eq!(a[[2, 0]], 5.0);
        assert_eq!(a[[2, 1]], 6.0);
    }

    #[test]
    fn matrix_vec_mul() {
        let a = matrix![1., 2.;
                        3., 4.;
                        5., 6.];
        let b = vector![4., 7.];

        let c = a * b;

        assert_eq!(c.size(), 3);

        assert_eq!(c[0], 18.0);
        assert_eq!(c[1], 40.0);
        assert_eq!(c[2], 62.0);
    }

    #[test]
    fn matrix_f32_mul() {
        let a = matrix![1., 2.;
                        3., 4.;
                        5., 6.];

        // Allocating new memory
        let c = &a * &2.0;

        assert_eq!(c[[0, 0]], 2.0);
        assert_eq!(c[[0, 1]], 4.0);
        assert_eq!(c[[1, 0]], 6.0);
        assert_eq!(c[[1, 1]], 8.0);
        assert_eq!(c[[2, 0]], 10.0);
        assert_eq!(c[[2, 1]], 12.0);

        // Allocating new memory
        let c = &a * 2.0;

        assert_eq!(c[[0, 0]], 2.0);
        assert_eq!(c[[0, 1]], 4.0);
        assert_eq!(c[[1, 0]], 6.0);
        assert_eq!(c[[1, 1]], 8.0);
        assert_eq!(c[[2, 0]], 10.0);
        assert_eq!(c[[2, 1]], 12.0);

        // Reusing memory
        let c = a.clone() * &2.0;

        assert_eq!(c[[0, 0]], 2.0);
        assert_eq!(c[[0, 1]], 4.0);
        assert_eq!(c[[1, 0]], 6.0);
        assert_eq!(c[[1, 1]], 8.0);
        assert_eq!(c[[2, 0]], 10.0);
        assert_eq!(c[[2, 1]], 12.0);

        // Reusing memory
        let c = a * 2.0;

        assert_eq!(c[[0, 0]], 2.0);
        assert_eq!(c[[0, 1]], 4.0);
        assert_eq!(c[[1, 0]], 6.0);
        assert_eq!(c[[1, 1]], 8.0);
        assert_eq!(c[[2, 0]], 10.0);
        assert_eq!(c[[2, 1]], 12.0);
    }

    #[test]
    fn matrix_add() {
        let a = matrix![1., 2.;
                        3., 4.;
                        5., 6.];
        let b = matrix![2., 3.;
                        4., 5.;
                        6., 7.];

        // Allocating new memory
        let c = &a + &b;

        assert_eq!(c[[0, 0]], 3.0);
        assert_eq!(c[[0, 1]], 5.0);
        assert_eq!(c[[1, 0]], 7.0);
        assert_eq!(c[[1, 1]], 9.0);
        assert_eq!(c[[2, 0]], 11.0);
        assert_eq!(c[[2, 1]], 13.0);

        // Reusing memory
        let c = a.clone() + &b;

        assert_eq!(c[[0, 0]], 3.0);
        assert_eq!(c[[0, 1]], 5.0);
        assert_eq!(c[[1, 0]], 7.0);
        assert_eq!(c[[1, 1]], 9.0);
        assert_eq!(c[[2, 0]], 11.0);
        assert_eq!(c[[2, 1]], 13.0);

        // Reusing memory
        let c = &a + b.clone();

        assert_eq!(c[[0, 0]], 3.0);
        assert_eq!(c[[0, 1]], 5.0);
        assert_eq!(c[[1, 0]], 7.0);
        assert_eq!(c[[1, 1]], 9.0);
        assert_eq!(c[[2, 0]], 11.0);
        assert_eq!(c[[2, 1]], 13.0);

        // Reusing memory
        let c = a + b;

        assert_eq!(c[[0, 0]], 3.0);
        assert_eq!(c[[0, 1]], 5.0);
        assert_eq!(c[[1, 0]], 7.0);
        assert_eq!(c[[1, 1]], 9.0);
        assert_eq!(c[[2, 0]], 11.0);
        assert_eq!(c[[2, 1]], 13.0);
    }

    #[test]
    fn matrix_f32_add() {
        let a = matrix![1., 2.;
                        3., 4.;
                        5., 6.];
        let b = 3.0;

        // Allocating new memory
        let c = &a + &b;

        assert_eq!(c[[0, 0]], 4.0);
        assert_eq!(c[[0, 1]], 5.0);
        assert_eq!(c[[1, 0]], 6.0);
        assert_eq!(c[[1, 1]], 7.0);
        assert_eq!(c[[2, 0]], 8.0);
        assert_eq!(c[[2, 1]], 9.0);

        // Allocating new memory
        let c = &a + b;

        assert_eq!(c[[0, 0]], 4.0);
        assert_eq!(c[[0, 1]], 5.0);
        assert_eq!(c[[1, 0]], 6.0);
        assert_eq!(c[[1, 1]], 7.0);
        assert_eq!(c[[2, 0]], 8.0);
        assert_eq!(c[[2, 1]], 9.0);

        // Reusing memory
        let c = a.clone() + &b;

        assert_eq!(c[[0, 0]], 4.0);
        assert_eq!(c[[0, 1]], 5.0);
        assert_eq!(c[[1, 0]], 6.0);
        assert_eq!(c[[1, 1]], 7.0);
        assert_eq!(c[[2, 0]], 8.0);
        assert_eq!(c[[2, 1]], 9.0);

        // Reusing memory
        let c = a + b;

        assert_eq!(c[[0, 0]], 4.0);
        assert_eq!(c[[0, 1]], 5.0);
        assert_eq!(c[[1, 0]], 6.0);
        assert_eq!(c[[1, 1]], 7.0);
        assert_eq!(c[[2, 0]], 8.0);
        assert_eq!(c[[2, 1]], 9.0);
    }

    #[test]
    fn matrix_sub() {
        let a = matrix![1., 2.;
                        3., 4.;
                        5., 6.];
        let b = matrix![2., 3.;
                        4., 5.;
                        6., 7.];

        // Allocate new memory
        let c = &a - &b;

        assert_eq!(c[[0, 0]], -1.0);
        assert_eq!(c[[0, 1]], -1.0);
        assert_eq!(c[[1, 0]], -1.0);
        assert_eq!(c[[1, 1]], -1.0);
        assert_eq!(c[[2, 0]], -1.0);
        assert_eq!(c[[2, 1]], -1.0);

        // Reusing memory
        let c = a.clone() - &b;

        assert_eq!(c[[0, 0]], -1.0);
        assert_eq!(c[[0, 1]], -1.0);
        assert_eq!(c[[1, 0]], -1.0);
        assert_eq!(c[[1, 1]], -1.0);
        assert_eq!(c[[2, 0]], -1.0);
        assert_eq!(c[[2, 1]], -1.0);

        // Reusing memory
        let c = &a - b.clone();

        assert_eq!(c[[0, 0]], -1.0);
        assert_eq!(c[[0, 1]], -1.0);
        assert_eq!(c[[1, 0]], -1.0);
        assert_eq!(c[[1, 1]], -1.0);
        assert_eq!(c[[2, 0]], -1.0);
        assert_eq!(c[[2, 1]], -1.0);

        // Reusing memory
        let c = &a - b;

        assert_eq!(c[[0, 0]], -1.0);
        assert_eq!(c[[0, 1]], -1.0);
        assert_eq!(c[[1, 0]], -1.0);
        assert_eq!(c[[1, 1]], -1.0);
        assert_eq!(c[[2, 0]], -1.0);
        assert_eq!(c[[2, 1]], -1.0);
    }

    #[test]
    fn matrix_f32_sub() {
        let a = matrix![1., 2.;
                        3., 4.;
                        5., 6.];
        let b = 3.0;

        // Allocating new memory
        let c = &a - &b;

        assert_eq!(c[[0, 0]], -2.0);
        assert_eq!(c[[0, 1]], -1.0);
        assert_eq!(c[[1, 0]], 0.0);
        assert_eq!(c[[1, 1]], 1.0);
        assert_eq!(c[[2, 0]], 2.0);
        assert_eq!(c[[2, 1]], 3.0);

        // Allocating new memory
        let c = &a - b;

        assert_eq!(c[[0, 0]], -2.0);
        assert_eq!(c[[0, 1]], -1.0);
        assert_eq!(c[[1, 0]], 0.0);
        assert_eq!(c[[1, 1]], 1.0);
        assert_eq!(c[[2, 0]], 2.0);
        assert_eq!(c[[2, 1]], 3.0);

        // Reusing memory
        let c = a.clone() - &b;

        assert_eq!(c[[0, 0]], -2.0);
        assert_eq!(c[[0, 1]], -1.0);
        assert_eq!(c[[1, 0]], 0.0);
        assert_eq!(c[[1, 1]], 1.0);
        assert_eq!(c[[2, 0]], 2.0);
        assert_eq!(c[[2, 1]], 3.0);

        // Reusing memory
        let c = a - b;

        assert_eq!(c[[0, 0]], -2.0);
        assert_eq!(c[[0, 1]], -1.0);
        assert_eq!(c[[1, 0]], 0.0);
        assert_eq!(c[[1, 1]], 1.0);
        assert_eq!(c[[2, 0]], 2.0);
        assert_eq!(c[[2, 1]], 3.0);
    }

    #[test]
    fn matrix_f32_div() {
        let a = matrix![1., 2.;
                        3., 4.;
                        5., 6.];
        let b = 3.0;

        // Allocating new memory
        let c = &a / &b;

        assert_eq!(c[[0, 0]], 1.0 / 3.0);
        assert_eq!(c[[0, 1]], 2.0 / 3.0);
        assert_eq!(c[[1, 0]], 1.0);
        assert_eq!(c[[1, 1]], 4.0 / 3.0);
        assert_eq!(c[[2, 0]], 5.0 / 3.0);
        assert_eq!(c[[2, 1]], 2.0);

        // Allocating new memory
        let c = &a / b;

        assert_eq!(c[[0, 0]], 1.0 / 3.0);
        assert_eq!(c[[0, 1]], 2.0 / 3.0);
        assert_eq!(c[[1, 0]], 1.0);
        assert_eq!(c[[1, 1]], 4.0 / 3.0);
        assert_eq!(c[[2, 0]], 5.0 / 3.0);
        assert_eq!(c[[2, 1]], 2.0);

        // Reusing memory
        let c = a.clone() / &b;

        assert_eq!(c[[0, 0]], 1.0 / 3.0);
        assert_eq!(c[[0, 1]], 2.0 / 3.0);
        assert_eq!(c[[1, 0]], 1.0);
        assert_eq!(c[[1, 1]], 4.0 / 3.0);
        assert_eq!(c[[2, 0]], 5.0 / 3.0);
        assert_eq!(c[[2, 1]], 2.0);

        // Reusing memory
        let c = a / b;

        assert_eq!(c[[0, 0]], 1.0 / 3.0);
        assert_eq!(c[[0, 1]], 2.0 / 3.0);
        assert_eq!(c[[1, 0]], 1.0);
        assert_eq!(c[[1, 1]], 4.0 / 3.0);
        assert_eq!(c[[2, 0]], 5.0 / 3.0);
        assert_eq!(c[[2, 1]], 2.0);
    }

    #[test]
    fn add_slice() {
        let a = 3.0;
        let mut b = Matrix::ones(3, 3) * 2.;
        let c = Matrix::ones(2, 2);

        {
            let d = MatrixSlice::from_matrix(&b, [1, 1], 2, 2);

            let m_1 = &d + a.clone();
            assert_eq!(m_1.into_vec(), vec![5.0; 4]);

            let m_2 = c.clone() + &d;
            assert_eq!(m_2.into_vec(), vec![3.0; 4]);

            let m_3 = &d + c.clone();
            assert_eq!(m_3.into_vec(), vec![3.0; 4]);

            let m_4 = &d + &d;
            assert_eq!(m_4.into_vec(), vec![4.0; 4]);
        }

        let e = MatrixSliceMut::from_matrix(&mut b, [1, 1], 2, 2);

        let m_1 = &e + a.clone();
        assert_eq!(m_1.into_vec(), vec![5.0; 4]);

        let m_2 = c.clone() + &e;
        assert_eq!(m_2.into_vec(), vec![3.0; 4]);

        let m_3 = &e + c;
        assert_eq!(m_3.into_vec(), vec![3.0; 4]);

        let m_4 = &e + &e;
        assert_eq!(m_4.into_vec(), vec![4.0; 4]);
    }

    #[test]
    fn sub_slice() {
        let a = 3.0;
        let b = Matrix::ones(2, 2);
        let mut c = Matrix::ones(3, 3) * 2.;

        {
            let d = MatrixSlice::from_matrix(&c, [1, 1], 2, 2);

            let m_1 = &d - a.clone();
            assert_eq!(m_1.into_vec(), vec![-1.0; 4]);

            let m_2 = b.clone() - &d;
            assert_eq!(m_2.into_vec(), vec![-1.0; 4]);

            let m_3 = &d - b.clone();
            assert_eq!(m_3.into_vec(), vec![1.0; 4]);

            let m_4 = &d - &d;
            assert_eq!(m_4.into_vec(), vec![0.0; 4]);
        }

        let e = MatrixSliceMut::from_matrix(&mut c, [1, 1], 2, 2);

        let m_1 = &e - a;
        assert_eq!(m_1.into_vec(), vec![-1.0; 4]);

        let m_2 = b.clone() - &e;
        assert_eq!(m_2.into_vec(), vec![-1.0; 4]);

        let m_3 = &e - b;
        assert_eq!(m_3.into_vec(), vec![1.0; 4]);

        let m_4 = &e - &e;
        assert_eq!(m_4.into_vec(), vec![0.0; 4]);
    }

    #[test]
    fn div_slice() {
        let a = 3.0;

        let mut b = Matrix::ones(3, 3) * 2.;

        {
            let c = MatrixSlice::from_matrix(&b, [1, 1], 2, 2);

            let m = c / a;
            assert_eq!(m.into_vec(), vec![2.0/3.0 ;4]);
        }

        let d = MatrixSliceMut::from_matrix(&mut b, [1, 1], 2, 2);

        let m = d / a;
        assert_eq!(m.into_vec(), vec![2.0/3.0 ;4]);
    }

    #[test]
    fn neg_slice() {
        let b = Matrix::ones(3, 3) * 2.;

        let c = MatrixSlice::from_matrix(&b, [1, 1], 2, 2);

        let m = -c;
        assert_eq!(m.into_vec(), vec![-2.0;4]);

        let mut b = Matrix::ones(3, 3) * 2.;

        let c = MatrixSliceMut::from_matrix(&mut b, [1, 1], 2, 2);

        let m = -c;
        assert_eq!(m.into_vec(), vec![-2.0;4]);
    }

    #[test]
    fn index_slice() {
        let mut b = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());

        {
            let c = MatrixSlice::from_matrix(&b, [1, 1], 2, 2);

            assert_eq!(c[[0, 0]], 4);
            assert_eq!(c[[0, 1]], 5);
            assert_eq!(c[[1, 0]], 7);
            assert_eq!(c[[1, 1]], 8);
        }


        let mut c = MatrixSliceMut::from_matrix(&mut b, [1, 1], 2, 2);

        assert_eq!(c[[0, 0]], 4);
        assert_eq!(c[[0, 1]], 5);
        assert_eq!(c[[1, 0]], 7);
        assert_eq!(c[[1, 1]], 8);

        c[[0, 0]] = 9;

        assert_eq!(c[[0, 0]], 9);
        assert_eq!(c[[0, 1]], 5);
        assert_eq!(c[[1, 0]], 7);
        assert_eq!(c[[1, 1]], 8);
    }

    #[test]
    fn matrix_add_assign() {
        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());

        a += &2;
        assert_eq!(a.into_vec(), (2..11).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());

        a += 2;
        assert_eq!(a.into_vec(), (2..11).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
        let b = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());

        a += &b;
        assert_eq!(a.into_vec(), (0..9).map(|x| 2 * x).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());

        a += b;
        assert_eq!(a.into_vec(), (0..9).map(|x| 2 * x).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
        let mut b = Matrix::new(4, 4, (0..16).collect::<Vec<_>>());
        {
            let c = MatrixSlice::from_matrix(&b, [0, 0], 3, 3);

            a += &c;
            assert_eq!(a.into_vec(), vec![0, 2, 4, 7, 9, 11, 14, 16, 18]);

            let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
            a += c;
            assert_eq!(a.into_vec(), vec![0, 2, 4, 7, 9, 11, 14, 16, 18]);
        }

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
        let c = MatrixSliceMut::from_matrix(&mut b, [0, 0], 3, 3);
        a += &c;
        assert_eq!(a.into_vec(), vec![0, 2, 4, 7, 9, 11, 14, 16, 18]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
        a += c;
        assert_eq!(a.into_vec(), vec![0, 2, 4, 7, 9, 11, 14, 16, 18]);

    }

    #[test]
    fn matrix_sub_assign() {
        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());

        a -= &2;
        assert_eq!(a.into_vec(), (-2..7).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        a -= 2;
        assert_eq!(a.into_vec(), (-2..7).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
        let b = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());

        a -= &b;
        assert_eq!(a.into_vec(), vec![0; 9]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());

        a -= b;
        assert_eq!(a.into_vec(), vec![0; 9]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
        let mut b = Matrix::new(4, 4, (0..16).collect::<Vec<_>>());
        {
            let c = MatrixSlice::from_matrix(&b, [0, 0], 3, 3);
            a -= &c;
            assert_eq!(a.into_vec(), vec![0, 0, 0, -1, -1, -1, -2, -2, -2]);

            let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
            a -= c;
            assert_eq!(a.into_vec(), vec![0, 0, 0, -1, -1, -1, -2, -2, -2]);
        }

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
        let c = MatrixSliceMut::from_matrix(&mut b, [0, 0], 3, 3);
        a -= &c;
        assert_eq!(a.into_vec(), vec![0, 0, 0, -1, -1, -1, -2, -2, -2]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
        a -= c;
        assert_eq!(a.into_vec(), vec![0, 0, 0, -1, -1, -1, -2, -2, -2]);
    }

    #[test]
    fn matrix_div_assign() {
        let a_data = vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let res_data = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
        let mut a = Matrix::new(3, 3, a_data.clone());

        a /= &2f32;
        assert_eq!(a.into_vec(), res_data.clone());

        let mut a = Matrix::new(3, 3, a_data.clone());
        a /= 2f32;
        assert_eq!(a.into_vec(), res_data.clone());
    }

    #[test]
    fn matrix_mul_assign() {
        let a_data = vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let res_data = vec![2f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
        let mut a = Matrix::new(3, 3, a_data.clone());

        a *= &2f32;
        assert_eq!(a.into_vec(), res_data.clone());

        let mut a = Matrix::new(3, 3, a_data.clone());
        a *= 2f32;
        assert_eq!(a.into_vec(), res_data.clone());
    }

    #[test]
    #[allow(unused_assignments, unused_variables)]
    fn slice_add_assign() {
        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice += &2;
        }
        assert_eq!(a.into_vec(), (2..11).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());

        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice += 2;
        }
        assert_eq!(a.into_vec(), (2..11).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        let b = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());

        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice += &b;
        }
        assert_eq!(a.into_vec(), (0..9).map(|x| 2 * x).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice += b;
        }
        assert_eq!(a.into_vec(), (0..9).map(|x| 2 * x).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        let mut b = Matrix::new(4, 4, (0..16).collect::<Vec<_>>());
        {
            let c = MatrixSlice::from_matrix(&b, [0, 0], 3, 3);
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice += &c;
        }
        assert_eq!(a.into_vec(), vec![0, 2, 4, 7, 9, 11, 14, 16, 18]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let c = MatrixSlice::from_matrix(&b, [0, 0], 3, 3);
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice += c;
        }
        assert_eq!(a.into_vec(), vec![0, 2, 4, 7, 9, 11, 14, 16, 18]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let c = MatrixSlice::from_matrix(&b, [0, 0], 3, 3);
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice += &c;
        }
        assert_eq!(a.into_vec(), vec![0, 2, 4, 7, 9, 11, 14, 16, 18]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let c = MatrixSliceMut::from_matrix(&mut b, [0, 0], 3, 3);
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice += c;
        }
        assert_eq!(a.into_vec(), vec![0, 2, 4, 7, 9, 11, 14, 16, 18]);

    }

    #[test]
    #[allow(unused_assignments, unused_variables)]
    fn slice_sub_assign() {
        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice -= &2;
        }
        assert_eq!(a.into_vec(), (-2..7).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice -= 2;
        }
        assert_eq!(a.into_vec(), (-2..7).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            let b = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
            a_slice -= &b;
        }
        assert_eq!(a.into_vec(), vec![0; 9]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            let b = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
            a_slice -= b;
        }
        assert_eq!(a.into_vec(), vec![0; 9]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        let mut b = Matrix::new(4, 4, (0..16).collect::<Vec<_>>());
        {
            let c = MatrixSlice::from_matrix(&b, [0, 0], 3, 3);
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice -= &c;
        }

        assert_eq!(a.into_vec(), vec![0, 0, 0, -1, -1, -1, -2, -2, -2]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let c = MatrixSlice::from_matrix(&b, [0, 0], 3, 3);
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice -= c;
        }
        assert_eq!(a.into_vec(), vec![0, 0, 0, -1, -1, -1, -2, -2, -2]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let c = MatrixSliceMut::from_matrix(&mut b, [0, 0], 3, 3);
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice -= &c;
        }
        assert_eq!(a.into_vec(), vec![0, 0, 0, -1, -1, -1, -2, -2, -2]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let c = MatrixSliceMut::from_matrix(&mut b, [0, 0], 3, 3);
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice -= c;
        }
        assert_eq!(a.into_vec(), vec![0, 0, 0, -1, -1, -1, -2, -2, -2]);
    }

    #[test]
    #[allow(unused_assignments, unused_variables)]
    fn slice_div_assign() {
        let a_data = vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let res_data = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
        let mut a = Matrix::new(3, 3, a_data.clone());

        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice /= &2f32;
        }
        assert_eq!(a.into_vec(), res_data.clone());

        let mut a = Matrix::new(3, 3, a_data.clone());
        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice /= 2f32;
        }
        assert_eq!(a.into_vec(), res_data.clone());
    }

    #[test]
    #[allow(unused_assignments, unused_variables)]
    fn slice_mul_assign() {
        let a_data = vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let res_data = vec![2f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
        let mut a = Matrix::new(3, 3, a_data.clone());

        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice *= &2f32;
        }
        assert_eq!(a.into_vec(), res_data.clone());

        let mut a = Matrix::new(3, 3, a_data.clone());
        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice *= 2f32;
        }
        assert_eq!(a.into_vec(), res_data.clone());
    }

    #[test]
    fn permutation_vector_mul() {
        let p = PermutationMatrix::from_array(vec![1, 2, 0]).unwrap();
        let x = vector![1, 2, 3];
        let expected = vector![3, 1, 2];

        {
            let y = p.clone() * x.clone();
            assert_eq!(y, expected);
        }

        {
            let y = p.clone() * &x;
            assert_eq!(y, expected);
        }

        {
            let y = &p * x.clone();
            assert_eq!(y, expected);
        }

        {
            let y = &p * &x;
            assert_eq!(y, expected);
        }
    }

    #[test]
    fn permutation_matrix_left_mul_for_matrix() {
        let p = PermutationMatrix::from_array(vec![1, 2, 0]).unwrap();
        let x = matrix![1, 2, 3;
                        4, 5, 6;
                        7, 8, 9];
        let expected = matrix![7, 8, 9;
                               1, 2, 3;
                               4, 5, 6];

        {
            // Consume p, consume rhs
            let y = p.clone() * x.clone();
            assert_eq!(y, expected);
        }

        {
            // Consume p, borrow rhs
            let y = p.clone() * &x;
            assert_eq!(y, expected);
        }

        {
            // Borrow p, consume rhs
            let y = &p * x.clone();
            assert_eq!(y, expected);
        }

        {
            // Borrow p, borrow rhs
            let y = &p * &x;
            assert_eq!(y, expected);
        }
    }

    #[test]
    fn permutation_matrix_left_mul_for_matrix_slice() {
        let p = PermutationMatrix::from_array(vec![1, 2, 0]).unwrap();
        let x_source = matrix![1, 2, 3;
                                   4, 5, 6;
                                   7, 8, 9];
        let expected = matrix![7, 8, 9;
                               1, 2, 3;
                               4, 5, 6];

        {
            // Immutable, consume p
            let x = x_source.sub_slice([0, 0], 3, 3);
            let y = p.clone() * &x;
            assert_eq!(y, expected);
        }

        {
            // Immutable, borrow p
            let x = x_source.sub_slice([0, 0], 3, 3);
            let y = &p * &x;
            assert_eq!(y, expected);
        }

        {
            // Mutable, consume p
            let mut x_source = x_source.clone();
            let x = x_source.sub_slice_mut([0, 0], 3, 3);
            let y = p.clone() * &x;
            assert_eq!(y, expected);
        }

        {
            // Mutable, borrow p
            let mut x_source = x_source.clone();
            let x = x_source.sub_slice_mut([0, 0], 3, 3);
            let y = &p * &x;
            assert_eq!(y, expected);
        }
    }

    #[test]
    fn permutation_matrix_right_mul_for_matrix() {
        let p = PermutationMatrix::from_array(vec![1, 2, 0]).unwrap();
        let x = matrix![1, 2, 3;
                        4, 5, 6;
                        7, 8, 9];
        let expected = matrix![3, 1, 2;
                               6, 4, 5;
                               9, 7, 8];

        {
            // Consume lhs, consume p
            let y = x.clone() * p.clone();
            assert_eq!(y, expected);
        }

        {
            // Consume lhs, borrow p
            let y = x.clone() * &p;
            assert_eq!(y, expected);
        }

        {
            // Borrow lhs, consume p
            let y = &x * p.clone();
            assert_eq!(y, expected);
        }

        {
            // Borrow lhs, borrow p
            let y = &x * &p;
            assert_eq!(y, expected);
        }
    }

     #[test]
    fn permutation_matrix_right_mul_for_matrix_slice() {
        let p = PermutationMatrix::from_array(vec![1, 2, 0]).unwrap();
        let x_source = matrix![1, 2, 3;
                        4, 5, 6;
                        7, 8, 9];
        let expected = matrix![3, 1, 2;
                               6, 4, 5;
                               9, 7, 8];

        {
            // Immutable lhs, consume p
            let x = x_source.sub_slice([0, 0], 3, 3);
            let y = &x * p.clone();
            assert_eq!(y, expected);
        }

        {
            // Immutable lhs, borrow p
            let x = x_source.sub_slice([0, 0], 3, 3);
            let y = &x * &p;
            assert_eq!(y, expected);
        }

        {
            // Mutable lhs, consume p
            let mut x_source = x_source.clone();
            let x = x_source.sub_slice_mut([0, 0], 3, 3);
            let y = &x * p.clone();
            assert_eq!(y, expected);
        }

        {
            // Mutable lhs, borrow p
            let mut x_source = x_source.clone();
            let x = x_source.sub_slice_mut([0, 0], 3, 3);
            let y = &x * &p;
            assert_eq!(y, expected);
        }
    }
}
