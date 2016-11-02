//! The matrix module.
//!
//! Currently contains all code
//! relating to the matrix linear algebra struct.
//!
//! Most of the logic for manipulating matrices is generically implemented
//! via `BaseMatrix` and `BaseMatrixMut` trait.

use std::any::Any;
use std::fmt;
use std::marker::PhantomData;
use libnum::{One, Zero, Float, FromPrimitive};

use Metric;
use error::{Error, ErrorKind};
use utils;
use vector::Vector;

mod decomposition;
mod impl_ops;
mod mat_mul;
mod iter;
pub mod slice;

pub use self::slice::{BaseMatrix, BaseMatrixMut};

/// Matrix dimensions
#[derive(Debug, Clone, Copy)]
pub enum Axes {
    /// The row axis.
    Row,
    /// The column axis.
    Col,
}

/// The `Matrix` struct.
///
/// Can be instantiated with any type.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

/// A `MatrixSlice`
///
/// This struct provides a slice into a matrix.
///
/// The struct contains the upper left point of the slice
/// and the width and height of the slice.
#[derive(Debug, Clone, Copy)]
pub struct MatrixSlice<'a, T: 'a> {
    ptr: *const T,
    rows: usize,
    cols: usize,
    row_stride: usize,
    marker: PhantomData<&'a T>,
}

/// A mutable `MatrixSliceMut`
///
/// This struct provides a mutable slice into a matrix.
///
/// The struct contains the upper left point of the slice
/// and the width and height of the slice.
#[derive(Debug)]
pub struct MatrixSliceMut<'a, T: 'a> {
    ptr: *mut T,
    rows: usize,
    cols: usize,
    row_stride: usize,
    marker: PhantomData<&'a mut T>,
}

/// Row iterator.
#[derive(Debug)]
pub struct Rows<'a, T: 'a> {
    slice_start: *const T,
    row_pos: usize,
    slice_rows: usize,
    slice_cols: usize,
    row_stride: isize,
    _marker: PhantomData<&'a T>,
}

/// Mutable row iterator.
#[derive(Debug)]
pub struct RowsMut<'a, T: 'a> {
    slice_start: *mut T,
    row_pos: usize,
    slice_rows: usize,
    slice_cols: usize,
    row_stride: isize,
    _marker: PhantomData<&'a mut T>,
}

/// Column iterator.
#[derive(Debug)]
pub struct Cols<'a, T: 'a> {
    _marker: PhantomData<&'a T>,
    col_pos: usize,
    slice_cols: usize,
    slice_rows: usize,
    slice_start: *const T,
}

/// Mutable column iterator.
#[derive(Debug)]
pub struct ColsMut<'a, T: 'a> {
    _marker: PhantomData<&'a mut T>,
    col_pos: usize,
    slice_cols: usize,
    slice_rows: usize,
    slice_start: *mut T,
}

/// Diagonal offset (used by Diagonal iterator).
#[derive(Debug, PartialEq)]
pub enum DiagOffset {
    /// The main diagonal of the matrix.
    Main,
    /// An offset above the main diagonal.
    Above(usize),
    /// An offset below the main diagonal.
    Below(usize),
}

/// An iterator over the diagonal elements of a matrix.
#[derive(Debug)]
pub struct Diagonal<'a, T: 'a, M: 'a + BaseMatrix<T>> {
    matrix: &'a M,
    diag_pos: usize,
    diag_end: usize,
    _marker: PhantomData<&'a T>,
}

/// An iterator over the mutable diagonal elements of a matrix.
#[derive(Debug)]
pub struct DiagonalMut<'a, T: 'a, M: 'a + BaseMatrixMut<T>> {
    matrix: &'a mut M,
    diag_pos: usize,
    diag_end: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<T> Matrix<T> {
    /// Constructor for Matrix struct.
    ///
    /// Requires both the row and column dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, BaseMatrix};
    ///
    /// let mat = Matrix::new(2,2, vec![1.0,2.0,3.0,4.0]);
    ///
    /// assert_eq!(mat.rows(), 2);
    /// assert_eq!(mat.cols(), 2);
    /// ```
    ///
    /// # Panics
    ///
    /// - The input data does not match the given dimensions.
    pub fn new<U: Into<Vec<T>>>(rows: usize, cols: usize, data: U) -> Matrix<T> {
        let our_data = data.into();

        assert!(cols * rows == our_data.len(),
                "Data does not match given dimensions.");
        Matrix {
            cols: cols,
            rows: rows,
            data: our_data,
        }
    }

    /// Constructor for Matrix struct that takes a function `f`
    /// and constructs a new matrix such that `A_ij = f(i, j)`,
    /// where `i` is the row index and `j` the column index.
    ///
    /// Requires both the row and column dimensions
    /// as well as a generating function.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, BaseMatrix};
    ///
    /// // Let's assume you have an array of "things" for
    /// // which you want to generate a distance matrix:
    /// let things: [i32; 3] = [1, 2, 3];
    /// let distances: Matrix<f64> = Matrix::from_fn(things.len(), things.len(), |col, row| {
    ///     (things[col] - things[row]).abs().into()
    /// });
    ///
    /// assert_eq!(distances.rows(), 3);
    /// assert_eq!(distances.cols(), 3);
    /// assert_eq!(distances.data(), &vec![
    ///     0.0, 1.0, 2.0,
    ///     1.0, 0.0, 1.0,
    ///     2.0, 1.0, 0.0,
    /// ]);
    /// ```
    pub fn from_fn<F>(rows: usize, cols: usize, mut f: F) -> Matrix<T>
        where F: FnMut(usize, usize) -> T
    {
        let mut data = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                data.push(f(col, row));
            }
        }
        Matrix::new(rows, cols, data)
    }

    /// Returns a non-mutable reference to the underlying data.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a mutable slice of the underlying data.
    pub fn mut_data(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consumes the Matrix and returns the Vec of data.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T: Clone> Clone for Matrix<T> {
    /// Clones the Matrix.
    fn clone(&self) -> Matrix<T> {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.clone(),
        }
    }
}

impl<T: Clone + Zero> Matrix<T> {
    /// Constructs matrix of all zeros.
    ///
    /// Requires both the row and the column dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let mat = Matrix::<f64>::zeros(2,3);
    /// ```
    pub fn zeros(rows: usize, cols: usize) -> Matrix<T> {
        Matrix {
            cols: cols,
            rows: rows,
            data: vec![T::zero(); cols*rows],
        }
    }

    /// Constructs matrix with given diagonal.
    ///
    /// Requires slice of diagonal elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let mat = Matrix::from_diag(&vec![1.0,2.0,3.0,4.0]);
    /// ```
    pub fn from_diag(diag: &[T]) -> Matrix<T> {
        let size = diag.len();
        let mut data = vec![T::zero(); size * size];

        for (i, item) in diag.into_iter().enumerate().take(size) {
            data[i * (size + 1)] = item.clone();
        }

        Matrix {
            cols: size,
            rows: size,
            data: data,
        }
    }
}

impl<T: Clone + One> Matrix<T> {
    /// Constructs matrix of all ones.
    ///
    /// Requires both the row and the column dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let mat = Matrix::<f64>::ones(2,3);
    /// ```
    pub fn ones(rows: usize, cols: usize) -> Matrix<T> {
        Matrix {
            cols: cols,
            rows: rows,
            data: vec![T::one(); cols*rows],
        }
    }
}

impl<T: Clone + Zero + One> Matrix<T> {
    /// Constructs the identity matrix.
    ///
    /// Requires the size of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let I = Matrix::<f64>::identity(4);
    /// ```
    pub fn identity(size: usize) -> Matrix<T> {
        let mut data = vec![T::zero(); size * size];

        for i in 0..size {
            data[(i * (size + 1)) as usize] = T::one();
        }

        Matrix {
            cols: size,
            rows: size,
            data: data,
        }
    }
}

impl<T: Float + FromPrimitive> Matrix<T> {
    /// The mean of the matrix along the specified axis.
    ///
    /// - Axis Row - Arithmetic mean of rows.
    /// - Axis Col - Arithmetic mean of columns.
    ///
    /// Calling `mean()` on an empty matrix will return an empty matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, Axes};
    ///
    /// let a = Matrix::<f64>::new(2,2, vec![1.0,2.0,3.0,4.0]);
    ///
    /// let c = a.mean(Axes::Row);
    /// assert_eq!(*c.data(), vec![2.0, 3.0]);
    ///
    /// let d = a.mean(Axes::Col);
    /// assert_eq!(*d.data(), vec![1.5, 3.5]);
    /// ```
    pub fn mean(&self, axis: Axes) -> Vector<T> {
        if self.data.len() == 0 {
            // If the matrix is empty, there are no means to calculate.
            return Vector::new(vec![]);
        }

        let m: Vector<T>;
        let n: T;
        match axis {
            Axes::Row => {
                m = self.sum_rows();
                n = FromPrimitive::from_usize(self.rows).unwrap();
            }
            Axes::Col => {
                m = self.sum_cols();
                n = FromPrimitive::from_usize(self.cols).unwrap();
            }
        }
        m / n
    }

    /// The variance of the matrix along the specified axis.
    ///
    /// - Axis Row - Sample variance of rows.
    /// - Axis Col - Sample variance of columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, Axes};
    ///
    /// let a = Matrix::<f32>::new(2,2,vec![1.0,2.0,3.0,4.0]);
    ///
    /// let c = a.variance(Axes::Row).unwrap();
    /// assert_eq!(*c.data(), vec![2.0, 2.0]);
    ///
    /// let d = a.variance(Axes::Col).unwrap();
    /// assert_eq!(*d.data(), vec![0.5, 0.5]);
    /// ```
    ///
    /// # Failures
    ///
    /// - There are one or fewer row/columns in the working axis.
    pub fn variance(&self, axis: Axes) -> Result<Vector<T>, Error> {
        let mean = self.mean(axis);

        let n: usize;
        let m: usize;

        match axis {
            Axes::Row => {
                n = self.rows;
                m = self.cols;
            }
            Axes::Col => {
                n = self.cols;
                m = self.rows;
            }
        }

        if n < 2 {
            return Err(Error::new(ErrorKind::InvalidArg,
                                  "There must be at least two rows or columns in the working \
                                   axis."));
        }

        let mut variance = Vector::zeros(m);

        for i in 0..n {
            let mut t = Vec::<T>::with_capacity(m);

            unsafe {
                t.set_len(m);

                for j in 0..m {
                    t[j] = match axis {
                        Axes::Row => *self.data.get_unchecked(i * m + j),
                        Axes::Col => *self.data.get_unchecked(j * n + i),
                    }

                }
            }

            let v = Vector::new(t);

            variance = variance + &(&v - &mean).elemul(&(&v - &mean));
        }

        let var_size: T = FromPrimitive::from_usize(n - 1).unwrap();
        Ok(variance / var_size)
    }
}

impl<T: Any + Float> Matrix<T> {
    /// Solves the equation `Ax = y`.
    ///
    /// Requires a Vector `y` as input.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::vector::Vector;
    ///
    /// let a = Matrix::new(2,2, vec![2.0,3.0,1.0,2.0]);
    /// let y = Vector::new(vec![13.0,8.0]);
    ///
    /// let x = a.solve(y).unwrap();
    ///
    /// assert_eq!(*x.data(), vec![2.0, 3.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix column count and vector size are different.
    /// - The matrix is not square.
    ///
    /// # Failures
    ///
    /// - The matrix cannot be decomposed into an LUP form to solve.
    /// - There is no valid solution as the matrix is singular.
    pub fn solve(&self, y: Vector<T>) -> Result<Vector<T>, Error> {
        let (l, u, p) = try!(self.lup_decomp());

        let b = try!(forward_substitution(&l, p * y));
        back_substitution(&u, b)
    }

    /// Computes the inverse of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(2,2, vec![2.,3.,1.,2.]);
    /// let inv = a.inverse().expect("This matrix should have an inverse!");
    ///
    /// let I = a * inv;
    ///
    /// assert_eq!(*I.data(), vec![1.0,0.0,0.0,1.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    ///
    /// # Failures
    ///
    /// - The matrix could not be LUP decomposed.
    /// - The matrix has zero determinant.
    pub fn inverse(&self) -> Result<Matrix<T>, Error> {
        assert!(self.rows == self.cols, "Matrix is not square.");

        let mut inv_t_data = Vec::<T>::new();
        let (l, u, p) = try!(self.lup_decomp().map_err(|_| {
            Error::new(ErrorKind::DecompFailure,
                       "Could not compute LUP factorization for inverse.")
        }));

        let mut d = T::one();

        unsafe {
            for i in 0..l.cols {
                d = d * *l.get_unchecked([i, i]);
                d = d * *u.get_unchecked([i, i]);
            }
        }

        if d == T::zero() {
            return Err(Error::new(ErrorKind::DecompFailure,
                                  "Matrix is singular and cannot be inverted."));
        }

        for i in 0..self.rows {
            let mut id_col = vec![T::zero(); self.cols];
            id_col[i] = T::one();

            let b = forward_substitution(&l, &p * Vector::new(id_col))
                .expect("Matrix is singular AND has non-zero determinant!?");
            inv_t_data.append(&mut back_substitution(&u, b)
                .expect("Matrix is singular AND has non-zero determinant!?")
                .into_vec());

        }

        Ok(Matrix::new(self.rows, self.cols, inv_t_data).transpose())
    }

    /// Computes the determinant of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(3,3, vec![1.0,2.0,0.0,
    ///                               0.0,3.0,4.0,
    ///                               5.0, 1.0, 2.0]);
    ///
    /// let det = a.det();
    ///
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    pub fn det(&self) -> T {
        assert!(self.rows == self.cols, "Matrix is not square.");

        let n = self.cols;

        if self.is_diag() {
            let mut d = T::one();

            unsafe {
                for i in 0..n {
                    d = d * *self.get_unchecked([i, i]);
                }
            }

            return d;
        }

        if n == 2 {
            (self[[0, 0]] * self[[1, 1]]) - (self[[0, 1]] * self[[1, 0]])
        } else if n == 3 {
            (self[[0, 0]] * self[[1, 1]] * self[[2, 2]]) +
            (self[[0, 1]] * self[[1, 2]] * self[[2, 0]]) +
            (self[[0, 2]] * self[[1, 0]] * self[[2, 1]]) -
            (self[[0, 0]] * self[[1, 2]] * self[[2, 1]]) -
            (self[[0, 1]] * self[[1, 0]] * self[[2, 2]]) -
            (self[[0, 2]] * self[[1, 1]] * self[[2, 0]])
        } else {
            let (l, u, p) = match self.lup_decomp() {
                Ok(x) => x,
                Err(ref e) if *e.kind() == ErrorKind::DivByZero => return T::zero(),
                _ => {
                    panic!("Could not compute LUP decomposition.");
                }
            };

            let mut d = T::one();

            unsafe {
                for i in 0..l.cols {
                    d = d * *l.get_unchecked([i, i]);
                    d = d * *u.get_unchecked([i, i]);
                }
            }

            let sgn = parity(&p);

            sgn * d
        }
    }
}

impl<T: Float> Metric<T> for Matrix<T> {
    /// Compute euclidean norm for matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::Metric;
    ///
    /// let a = Matrix::new(2,1, vec![3.0,4.0]);
    /// let c = a.norm();
    ///
    /// assert_eq!(c, 5.0);
    /// ```
    fn norm(&self) -> T {
        let s = utils::dot(&self.data, &self.data);

        s.sqrt()
    }
}

impl<'a, T: Float> Metric<T> for MatrixSlice<'a, T> {
    /// Compute euclidean norm for matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, MatrixSlice};
    /// use rulinalg::Metric;
    ///
    /// let a = Matrix::new(2,1, vec![3.0,4.0]);
    /// let b = MatrixSlice::from_matrix(&a, [0,0], 2, 1);
    /// let c = b.norm();
    ///
    /// assert_eq!(c, 5.0);
    /// ```
    fn norm(&self) -> T {
        let mut s = T::zero();

        for row in self.iter_rows() {
            s = s + utils::dot(row, row);
        }
        s.sqrt()
    }
}

impl<'a, T: Float> Metric<T> for MatrixSliceMut<'a, T> {
    /// Compute euclidean norm for matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, MatrixSliceMut};
    /// use rulinalg::Metric;
    ///
    /// let mut a = Matrix::new(2,1, vec![3.0,4.0]);
    /// let b = MatrixSliceMut::from_matrix(&mut a, [0,0], 2, 1);
    /// let c = b.norm();
    ///
    /// assert_eq!(c, 5.0);
    /// ```
    fn norm(&self) -> T {
        let mut s = T::zero();

        for row in self.iter_rows() {
            s = s + utils::dot(row, row);
        }
        s.sqrt()
    }
}

impl<T: fmt::Display> fmt::Display for Matrix<T> {
    /// Formats the Matrix for display.
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let mut max_datum_width = 0;
        for datum in &self.data {
            let datum_width = match f.precision() {
                Some(places) => format!("{:.1$}", datum, places).len(),
                None => format!("{}", datum).len(),
            };
            if datum_width > max_datum_width {
                max_datum_width = datum_width;
            }
        }
        let width = max_datum_width;

        fn write_row<T>(f: &mut fmt::Formatter,
                        row: &[T],
                        left_delimiter: &str,
                        right_delimiter: &str,
                        width: usize)
                        -> Result<(), fmt::Error>
            where T: fmt::Display
        {
            try!(write!(f, "{}", left_delimiter));
            for (index, datum) in row.iter().enumerate() {
                match f.precision() {
                    Some(places) => {
                        try!(write!(f, "{:1$.2$}", datum, width, places));
                    }
                    None => {
                        try!(write!(f, "{:1$}", datum, width));
                    }
                }
                if index < row.len() - 1 {
                    try!(write!(f, " "));
                }
            }
            write!(f, "{}", right_delimiter)
        }

        match self.rows {
            1 => write_row(f, &self.data, "[", "]", width),
            _ => {
                try!(write_row(f,
                               &self.data[0..self.cols],
                               "⎡", // \u{23a1} LEFT SQUARE BRACKET UPPER CORNER
                               "⎤", // \u{23a4} RIGHT SQUARE BRACKET UPPER CORNER
                               width));
                try!(f.write_str("\n"));
                for row_index in 1..self.rows - 1 {
                    try!(write_row(f,
                                   &self.data[row_index * self.cols..(row_index + 1) * self.cols],
                                   "⎢", // \u{23a2} LEFT SQUARE BRACKET EXTENSION
                                   "⎥", // \u{23a5} RIGHT SQUARE BRACKET EXTENSION
                                   width));
                    try!(f.write_str("\n"));
                }
                write_row(f,
                          &self.data[(self.rows - 1) * self.cols..self.rows * self.cols],
                          "⎣", // \u{23a3} LEFT SQUARE BRACKET LOWER CORNER
                          "⎦", // \u{23a6} RIGHT SQUARE BRACKET LOWER CORNER
                          width)
            }
        }

    }
}

/// Back substitution
fn back_substitution<T, M>(m: &M, y: Vector<T>) -> Result<Vector<T>, Error>
    where T: Any + Float,
          M: BaseMatrix<T>
{
    if m.is_empty() {
        return Err(Error::new(ErrorKind::InvalidArg, "Matrix is empty."));
    }

    let mut x = vec![T::zero(); y.size()];

    unsafe {
        for i in (0..y.size()).rev() {
            let mut holding_u_sum = T::zero();
            for j in (i + 1..y.size()).rev() {
                holding_u_sum = holding_u_sum + *m.get_unchecked([i, j]) * x[j];
            }

            let diag = *m.get_unchecked([i, i]);
            if diag.abs() < T::min_positive_value() + T::min_positive_value() {
                return Err(Error::new(ErrorKind::AlgebraFailure,
                                      "Linear system cannot be solved (matrix is singular)."));
            }
            x[i] = (y[i] - holding_u_sum) / diag;
        }
    }

    Ok(Vector::new(x))
}

/// forward substitution
fn forward_substitution<T, M>(m: &M, y: Vector<T>) -> Result<Vector<T>, Error>
    where T: Any + Float,
          M: BaseMatrix<T>
{
    if m.is_empty() {
        return Err(Error::new(ErrorKind::InvalidArg, "Matrix is empty."));
    }

    let mut x = Vec::with_capacity(y.size());

    unsafe {
        for (i, y_item) in y.data().iter().enumerate().take(y.size()) {
            let mut holding_l_sum = T::zero();
            for (j, x_item) in x.iter().enumerate().take(i) {
                holding_l_sum = holding_l_sum + *m.get_unchecked([i, j]) * *x_item;
            }

            let diag = *m.get_unchecked([i, i]);

            if diag.abs() < T::min_positive_value() + T::min_positive_value() {
                return Err(Error::new(ErrorKind::AlgebraFailure,
                                      "Linear system cannot be solved (matrix is singular)."));
            }
            x.push((*y_item - holding_l_sum) / diag);
        }
    }

    Ok(Vector::new(x))
}

/// Computes the parity of a permutation matrix.
fn parity<T, M>(m: &M) -> T
    where T: Any + Float,
          M: BaseMatrix<T>
{
    let mut visited = vec![false; m.rows()];
    let mut sgn = T::one();

    for k in 0..m.rows() {
        if !visited[k] {
            let mut next = k;
            let mut len = 0;

            while !visited[next] {
                len += 1;
                visited[next] = true;
                next = utils::find(&m.get_row(next).unwrap(), T::one());
            }

            if len % 2 == 0 {
                sgn = -sgn;
            }
        }
    }
    sgn
}

#[cfg(test)]
mod tests {
    use super::super::vector::Vector;
    use super::Matrix;
    use super::slice::BaseMatrix;
    use libnum::abs;

    #[test]
    fn test_new_mat() {
        let a = vec![2.0; 9];
        let b = Matrix::new(3, 3, a);

        assert_eq!(b.rows(), 3);
        assert_eq!(b.cols(), 3);
        assert_eq!(b.into_vec(), vec![2.0; 9]);
    }

    #[test]
    #[should_panic]
    fn test_new_mat_bad_data() {
        let a = vec![2.0; 7];
        let _ = Matrix::new(3, 3, a);
    }

    #[test]
    fn test_new_mat_from_fn() {
        let mut counter = 0;
        let m: Matrix<usize> = Matrix::from_fn(3, 2, |_, _| {
            let value = counter;
            counter += 1;
            value
        });
        assert!(m.rows() == 3);
        assert!(m.cols() == 2);
        assert!(m.data == vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_equality() {
        // well, "PartialEq", at least
        let a = matrix!(1., 2., 3.;
                        4., 5., 6.);
        let a_redux = a.clone();
        assert_eq!(a, a_redux);
    }

    #[test]
    fn test_new_from_slice() {
        let data_vec: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let data_slice: &[u32] = &data_vec[..];
        let from_vec = Matrix::new(3, 2, data_vec.clone());
        let from_slice = Matrix::new(3, 2, data_slice);
        assert_eq!(from_vec, from_slice);
    }

    #[test]
    fn test_display_formatting() {
        let first_matrix = matrix!(1, 2, 3;
                                   4, 5, 6);
        let first_expectation = "⎡1 2 3⎤\n⎣4 5 6⎦";
        assert_eq!(first_expectation, format!("{}", first_matrix));

        let second_matrix = matrix!(3.14, 2.718, 1.414;
                                    2.503, 4.669, 1.202;
                                    1.618, 0.5772, 1.3;
                                    2.68545, 1.282, 10000.);
        let second_exp = "⎡   3.14   2.718   1.414⎤\n⎢  2.503   4.669   1.202⎥\n⎢  \
                        1.618  0.5772     1.3⎥\n⎣2.68545   1.282   10000⎦";
        assert_eq!(second_exp, format!("{}", second_matrix));
    }

    #[test]
    fn test_single_row_display_formatting() {
        let one_row_matrix = matrix!(1, 2, 3, 4);
        assert_eq!("[1 2 3 4]", format!("{}", one_row_matrix));
    }

    #[test]
    fn test_display_formatting_precision() {
        let our_matrix = matrix!(1.2, 1.23, 1.234;
                                 1.2345, 1.23456, 1.234567);
        let expectations = vec!["⎡1.2 1.2 1.2⎤\n⎣1.2 1.2 1.2⎦",

                                "⎡1.20 1.23 1.23⎤\n⎣1.23 1.23 1.23⎦",

                                "⎡1.200 1.230 1.234⎤\n⎣1.234 1.235 1.235⎦",

                                "⎡1.2000 1.2300 1.2340⎤\n⎣1.2345 1.2346 1.2346⎦"];

        for (places, &expectation) in (1..5).zip(expectations.iter()) {
            assert_eq!(expectation, format!("{:.1$}", our_matrix, places));
        }
    }

    #[test]
    fn test_matrix_index_mut() {
        let mut a = Matrix::ones(3, 3) * 2.0;

        a[[0, 0]] = 13.0;

        for i in 1..9 {
            assert_eq!(a.data()[i], 2.0);
        }

        assert_eq!(a[[0, 0]], 13.0);
    }

    #[test]
    fn test_matrix_select_rows() {
        let a = Matrix::new(4, 2, (0..8).collect::<Vec<usize>>());

        let b = a.select_rows(&[0, 2, 3]);

        assert_eq!(b.into_vec(), vec![0, 1, 4, 5, 6, 7]);
    }

    #[test]
    fn test_matrix_select_cols() {
        let a = Matrix::new(4, 2, (0..8).collect::<Vec<usize>>());

        let b = a.select_cols(&[1]);

        assert_eq!(b.into_vec(), vec![1, 3, 5, 7]);
    }

    #[test]
    fn test_matrix_select() {
        let a = Matrix::new(4, 2, (0..8).collect::<Vec<usize>>());

        let b = a.select(&[0, 2], &[1]);

        assert_eq!(b.into_vec(), vec![1, 5]);
    }

    #[test]
    fn matrix_diag() {
        let a = matrix!(1., 3., 5.;
                        2., 4., 7.;
                        1., 1., 0.);

        let b = a.is_diag();

        assert!(!b);

        let c = matrix!(1., 0., 0.;
                        0., 2., 0.;
                        0., 0., 3.);
        let d = c.is_diag();

        assert!(d);
    }

    #[test]
    fn matrix_det() {
        let a = matrix!(2., 3.;
                        1., 2.);
        let b = a.det();

        assert_eq!(b, 1.);

        let c = matrix!(1., 2., 3.;
                        4., 5., 6.;
                        7., 8., 9.);
        let d = c.det();

        assert_eq!(d, 0.);

        let e: Matrix<f64> = matrix!(1., 2., 3., 4., 5.;
                                     3., 0., 4., 5., 6.;
                                     2., 1., 2., 3., 4.;
                                     0., 0., 0., 6., 5.;
                                     0., 0., 0., 5., 6.);

        let f = e.det();

        println!("det is {0}", f);
        let error = abs(f - 99.);
        assert!(error < 1e-10);

        let g: Matrix<f64> = matrix!(1., 2., 3., 4.;
                                     0., 0., 0., 0.;
                                     0., 0., 0., 0.;
                                     0., 0., 0., 0.);
        let h = g.det();
        assert_eq!(h, 0.);
    }

    #[test]
    fn matrix_solve() {
        let a = matrix!(2., 3.;
                        1., 2.);

        let y = Vector::new(vec![8., 5.]);

        let x = a.solve(y).unwrap();

        assert_eq!(x.size(), 2);

        assert_eq!(x[0], 1.);
        assert_eq!(x[1], 2.);
    }

    #[test]
    fn create_mat_zeros() {
        let a = Matrix::<f32>::zeros(10, 10);

        assert_eq!(a.rows(), 10);
        assert_eq!(a.cols(), 10);

        for i in 0..10 {
            for j in 0..10 {
                assert_eq!(a[[i, j]], 0.0);
            }
        }
    }

    #[test]
    fn create_mat_identity() {
        let a = Matrix::<f32>::identity(4);

        assert_eq!(a.rows(), 4);
        assert_eq!(a.cols(), 4);

        assert_eq!(a[[0, 0]], 1.0);
        assert_eq!(a[[1, 1]], 1.0);
        assert_eq!(a[[2, 2]], 1.0);
        assert_eq!(a[[3, 3]], 1.0);

        assert_eq!(a[[0, 1]], 0.0);
        assert_eq!(a[[2, 1]], 0.0);
        assert_eq!(a[[3, 0]], 0.0);
    }

    #[test]
    fn create_mat_diag() {
        let a = Matrix::from_diag(&[1.0, 2.0, 3.0, 4.0]);

        assert_eq!(a.rows(), 4);
        assert_eq!(a.cols(), 4);

        assert_eq!(a[[0, 0]], 1.0);
        assert_eq!(a[[1, 1]], 2.0);
        assert_eq!(a[[2, 2]], 3.0);
        assert_eq!(a[[3, 3]], 4.0);

        assert_eq!(a[[0, 1]], 0.0);
        assert_eq!(a[[2, 1]], 0.0);
        assert_eq!(a[[3, 0]], 0.0);
    }

    #[test]
    fn test_empty_mean() {
        use super::Axes;

        let a: Matrix<f64> = matrix!();

        let c = a.mean(Axes::Row);
        assert_eq!(*c.data(), vec![]);

        let d = a.mean(Axes::Col);
        assert_eq!(*d.data(), vec![]);
    }

    #[test]
    fn test_invalid_variance() {
        use super::Axes;

        // Only one row
        let a: Matrix<f32> = matrix!(1.0, 2.0);

        let a_row = a.variance(Axes::Row);
        assert!(a_row.is_err());

        let a_col = a.variance(Axes::Col).unwrap();
        assert_eq!(*a_col.data(), vec![0.5]);

        // Only one column
        let b: Matrix<f32> = matrix!(1.0; 2.0);

        let b_row = b.variance(Axes::Row).unwrap();
        assert_eq!(*b_row.data(), vec![0.5]);

        let b_col = b.variance(Axes::Col);
        assert!(b_col.is_err());

        // Empty matrix
        let d: Matrix<f32> = matrix!();

        let d_row = d.variance(Axes::Row);
        assert!(d_row.is_err());

        let d_col = d.variance(Axes::Col);
        assert!(d_col.is_err());
    }
}
