//! The matrix module.
//!
//! Currently contains all code
//! relating to the matrix linear algebra struct.

use std::any::Any;
use std::fmt;
use std::marker::PhantomData;
use libnum::{One, Zero, Float};

use Metric;
use error::{Error, ErrorKind};
use utils;
use vector::Vector;
use self::slice::BaseSlice;

pub mod decomposition;
mod impl_ops;
mod mat_mul;
mod iter;
pub mod slice;

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

impl<T> Matrix<T> {
    /// Constructor for Matrix struct.
    ///
    /// Requires both the row and column dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::slice::BaseSlice;
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

    /// Split the matrix at the specified axis returning two `MatrixSlice`s.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::Axes;
    ///
    /// let a = Matrix::new(3,3, vec![2.0; 9]);
    /// let (b,c) = a.split_at(1, Axes::Row);
    /// ```
    pub fn split_at(&self, mid: usize, axis: Axes) -> (MatrixSlice<T>, MatrixSlice<T>) {
        let slice_1: MatrixSlice<T>;
        let slice_2: MatrixSlice<T>;

        match axis {
            Axes::Row => {
                assert!(mid < self.rows);

                slice_1 = MatrixSlice::from_matrix(self, [0, 0], mid, self.cols);
                slice_2 = MatrixSlice::from_matrix(self, [mid, 0], self.rows - mid, self.cols);
            }
            Axes::Col => {
                assert!(mid < self.cols);

                slice_1 = MatrixSlice::from_matrix(self, [0, 0], self.rows, mid);
                slice_2 = MatrixSlice::from_matrix(self, [0, mid], self.rows, self.cols - mid);
            }
        }

        (slice_1, slice_2)
    }

    /// Split the matrix at the specified axis returning two `MatrixSlice`s.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::Axes;
    ///
    /// let mut a = Matrix::new(3,3, vec![2.0; 9]);
    /// let (b,c) = a.split_at_mut(1, Axes::Col);
    /// ```
    pub fn split_at_mut(&mut self,
                        mid: usize,
                        axis: Axes)
                        -> (MatrixSliceMut<T>, MatrixSliceMut<T>) {

        let mat_cols = self.cols;
        let mat_rows = self.rows;

        let slice_1: MatrixSliceMut<T>;
        let slice_2: MatrixSliceMut<T>;

        match axis {
            Axes::Row => {
                assert!(mid < self.rows);

                unsafe {
                    slice_1 = MatrixSliceMut::from_raw_parts(self.data.as_mut_ptr(),
                                                             mid,
                                                             mat_cols,
                                                             mat_cols);
                    slice_2 =
                        MatrixSliceMut::from_raw_parts(self.data
                                                           .as_mut_ptr()
                                                           .offset((mid * mat_cols) as isize),
                                                       mat_rows - mid,
                                                       mat_cols,
                                                       mat_cols);
                }
            }
            Axes::Col => {
                assert!(mid < self.cols);
                unsafe {
                    slice_1 = MatrixSliceMut::from_raw_parts(self.data.as_mut_ptr(),
                                                             mat_rows,
                                                             mid,
                                                             mat_cols);
                    slice_2 = MatrixSliceMut::from_raw_parts(self.data
                                                                 .as_mut_ptr()
                                                                 .offset(mid as isize),
                                                             mat_rows,
                                                             mat_cols - mid,
                                                             mat_cols);
                }
            }
        }

        (slice_1, slice_2)
    }

    /// Returns a `MatrixSlice` over the whole matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(3, 3, vec![2.0; 9]);
    /// let b = a.as_slice();
    /// ```
    pub fn as_slice(&self) -> MatrixSlice<T> {
        MatrixSlice::from_matrix(self, [0, 0], self.rows, self.cols)
    }

    /// Returns a mutable `MatrixSlice` over the whole matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let mut a = Matrix::new(3, 3, vec![2.0; 9]);
    /// let b = a.as_mut_slice();
    /// ```
    pub fn as_mut_slice(&mut self) -> MatrixSliceMut<T> {
        let rows = self.rows;
        let cols = self.cols;
        MatrixSliceMut::from_matrix(self, [0, 0], rows, cols)
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

        fn write_row<T: fmt::Display>(f: &mut fmt::Formatter,
                                      row: &[T],
                                      left_delimiter: &str,
                                      right_delimiter: &str,
                                      width: usize)
                                      -> Result<(), fmt::Error> {
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
          M: BaseSlice<T>,
{
    let mut x = vec![T::zero(); y.size()];

    unsafe {
        x[y.size() - 1] = y[y.size() - 1] / *m.get_unchecked([y.size() - 1, y.size() - 1]);

        for i in (0..y.size() - 1).rev() {
            let mut holding_u_sum = T::zero();
            for j in (i + 1..y.size()).rev() {
                holding_u_sum = holding_u_sum + *m.get_unchecked([i, j]) * x[j];
            }

            let diag = *m.get_unchecked([i, i]);
            if diag.abs() < T::min_positive_value() + 
                T::min_positive_value() 
            {
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
          M: BaseSlice<T>,
{
    let mut x = Vec::with_capacity(y.size());

    unsafe {
        x.push(y[0] / *m.get_unchecked([0, 0]));
        for (i, y_item) in y.data().iter().enumerate().take(y.size()).skip(1) {
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
          M: BaseSlice<T>,
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
    use super::Axes;
    use super::slice::BaseSlice;
    use super::decomposition::Decomposition;
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
    fn test_equality() {
        // well, "PartialEq", at least
        let a = Matrix::new(2, 3, vec![1., 2., 3., 4., 5., 6.]);
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
        let first_matrix = Matrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
        let first_expectation = "⎡1 2 3⎤\n⎣4 5 6⎦";
        assert_eq!(first_expectation, format!("{}", first_matrix));

        let second_matrix = Matrix::new(4,
                                        3,
                                        vec![3.14, 2.718, 1.414, 2.503, 4.669, 1.202, 1.618,
                                             0.5772, 1.3, 2.68545, 1.282, 10000.]);
        let second_exp = "⎡   3.14   2.718   1.414⎤\n⎢  2.503   4.669   1.202⎥\n⎢  \
                        1.618  0.5772     1.3⎥\n⎣2.68545   1.282   10000⎦";
        assert_eq!(second_exp, format!("{}", second_matrix));
    }

    #[test]
    fn test_single_row_display_formatting() {
        let one_row_matrix = Matrix::new(1, 4, vec![1, 2, 3, 4]);
        assert_eq!("[1 2 3 4]", format!("{}", one_row_matrix));
    }

    #[test]
    fn test_display_formatting_precision() {
        let our_matrix = Matrix::new(2, 3, vec![1.2, 1.23, 1.234, 1.2345, 1.23456, 1.234567]);
        let expectations = vec!["⎡1.2 1.2 1.2⎤\n⎣1.2 1.2 1.2⎦",

                                "⎡1.20 1.23 1.23⎤\n⎣1.23 1.23 1.23⎦",

                                "⎡1.200 1.230 1.234⎤\n⎣1.234 1.235 1.235⎦",

                                "⎡1.2000 1.2300 1.2340⎤\n⎣1.2345 1.2346 1.2346⎦"];

        for (places, &expectation) in (1..5).zip(expectations.iter()) {
            assert_eq!(expectation, format!("{:.1$}", our_matrix, places));
        }
    }

    #[test]
    fn test_split_matrix() {
        let a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());

        let (b, c) = a.split_at(1, Axes::Row);

        assert_eq!(b.rows(), 1);
        assert_eq!(b.cols(), 3);
        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 3);

        assert_eq!(b[[0, 0]], 0);
        assert_eq!(b[[0, 1]], 1);
        assert_eq!(b[[0, 2]], 2);
        assert_eq!(c[[0, 0]], 3);
        assert_eq!(c[[0, 1]], 4);
        assert_eq!(c[[0, 2]], 5);
        assert_eq!(c[[1, 0]], 6);
        assert_eq!(c[[1, 1]], 7);
        assert_eq!(c[[1, 2]], 8);
    }

    #[test]
    fn test_split_matrix_mut() {
        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());

        {
            let (mut b, mut c) = a.split_at_mut(1, Axes::Row);

            assert_eq!(b.rows(), 1);
            assert_eq!(b.cols(), 3);
            assert_eq!(c.rows(), 2);
            assert_eq!(c.cols(), 3);

            assert_eq!(b[[0, 0]], 0);
            assert_eq!(b[[0, 1]], 1);
            assert_eq!(b[[0, 2]], 2);
            assert_eq!(c[[0, 0]], 3);
            assert_eq!(c[[0, 1]], 4);
            assert_eq!(c[[0, 2]], 5);
            assert_eq!(c[[1, 0]], 6);
            assert_eq!(c[[1, 1]], 7);
            assert_eq!(c[[1, 2]], 8);

            b[[0, 0]] = 4;
            c[[0, 0]] = 5;
        }

        assert_eq!(a[[0, 0]], 4);
        assert_eq!(a[[0, 1]], 1);
        assert_eq!(a[[0, 2]], 2);
        assert_eq!(a[[1, 0]], 5);
        assert_eq!(a[[1, 1]], 4);
        assert_eq!(a[[1, 2]], 5);
        assert_eq!(a[[2, 0]], 6);
        assert_eq!(a[[2, 1]], 7);
        assert_eq!(a[[2, 2]], 8);

    }

    #[test]
    fn test_matrix_index_mut() {
        let mut a = Matrix::new(3, 3, vec![2.0; 9]);

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
        let a = Matrix::new(3, 3, vec![1., 3., 5., 2., 4., 7., 1., 1., 0.]);

        let b = a.is_diag();

        assert!(!b);

        let c = Matrix::new(3, 3, vec![1., 0., 0., 0., 2., 0., 0., 0., 3.]);
        let d = c.is_diag();

        assert!(d);
    }

    #[test]
    fn matrix_det() {
        let a = Matrix::new(2, 2, vec![2., 3., 1., 2.]);
        let b = a.det();

        assert_eq!(b, 1.);

        let c = Matrix::new(3, 3, vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let d = c.det();

        assert_eq!(d, 0.);

        let e = Matrix::<f64>::new(5,
                                   5,
                                   vec![1., 2., 3., 4., 5., 3., 0., 4., 5., 6., 2., 1., 2., 3.,
                                        4., 0., 0., 0., 6., 5., 0., 0., 0., 5., 6.]);

        let f = e.det();

        println!("det is {0}", f);
        let error = abs(f - 99.);
        assert!(error < 1e-10);
    }

    #[test]
    fn matrix_solve() {
        let a = Matrix::new(2, 2, vec![2., 3., 1., 2.]);

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
    fn transpose_mat() {
        let a = Matrix::new(5, 2, vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);

        let c = a.transpose();

        assert_eq!(c.cols(), a.rows());
        assert_eq!(c.rows(), a.cols());

        assert_eq!(a[[0, 0]], c[[0, 0]]);
        assert_eq!(a[[1, 0]], c[[0, 1]]);
        assert_eq!(a[[2, 0]], c[[0, 2]]);
        assert_eq!(a[[3, 0]], c[[0, 3]]);
        assert_eq!(a[[4, 0]], c[[0, 4]]);
        assert_eq!(a[[0, 1]], c[[1, 0]]);
        assert_eq!(a[[1, 1]], c[[1, 1]]);
        assert_eq!(a[[2, 1]], c[[1, 2]]);
        assert_eq!(a[[3, 1]], c[[1, 3]]);
        assert_eq!(a[[4, 1]], c[[1, 4]]);

    }
}
