//! Slices for the `Matrix` struct.
//!
//! These slices provide a view into the matrix data.
//! The view must be a contiguous block of the matrix.
//!
//! ```
//! use rulinalg::matrix::Matrix;
//! use rulinalg::matrix::MatrixSlice;
//!
//! let a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
//!
//! // Manually create our slice - [[4,5],[7,8]].
//! let mat_slice = MatrixSlice::from_matrix(&a, [1,1], 2, 2);
//!
//! // We can perform arithmetic with slices.
//! let new_mat = &mat_slice * &mat_slice;
//! ```

use matrix::{Matrix, MatrixSlice, MatrixSliceMut, Rows, RowsMut};
use matrix::{back_substitution, forward_substitution};
use vector::Vector;
use utils;
use libnum::{Zero, Float};
use error::Error;

use std::any::Any;
use std::cmp::min;
use std::marker::PhantomData;
use std::mem;

/// Trait for Matrix Slices.
pub trait BaseSlice<T>: Sized {

    /// Rows in the slice.
    fn rows(&self) -> usize;

    /// Columns in the slice.
    fn cols(&self) -> usize;

    /// Row stride in the slice.
    fn row_stride(&self) -> usize;

    /// Top left index of the slice.
    fn as_ptr(&self) -> *const T;

    /// Get a reference to a point in the slice without bounds checking.
    unsafe fn get_unchecked(&self, index: [usize; 2]) -> &T {
        &*(self.as_ptr().offset((index[0] * self.row_stride() + index[1]) as isize))
    }

    /// Returns the row of a `Matrix` at the given index.
    /// `None` if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, MatrixSlice};
    /// use rulinalg::matrix::slice::BaseSlice;
    ///
    /// let mut a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let mut slice = MatrixSlice::from_matrix(&mut a, [1,1], 2, 2);
    /// let row = slice.get_row(1);
    /// let mut expected = vec![7usize, 8];
    /// assert_eq!(row, Some(&*expected));
    /// assert!(slice.get_row(5).is_none());
    /// ```
    fn get_row(&self, index: usize) -> Option<&[T]> {
        if index < self.rows() {
            unsafe { Some(self.get_row_unchecked(index)) }
        } else {
            None
        }
    }
    
    /// Returns the row of a `BaseSlice` at the given index without doing unbounds checking
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, MatrixSlice};
    /// use rulinalg::matrix::slice::BaseSlice;
    ///
    /// let mut a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let mut slice = MatrixSlice::from_matrix(&mut a, [1,1], 2, 2);
    /// let row = unsafe { slice.get_row_unchecked(1) };
    /// let mut expected = vec![7usize, 8];
    /// assert_eq!(row, &*expected);
    /// ```
    unsafe fn get_row_unchecked(&self, index: usize) -> &[T] {
        let ptr = self.as_ptr().offset((self.row_stride() * index) as isize);
        ::std::slice::from_raw_parts(ptr, self.cols())
    }

    /// Returns an iterator over the matrix slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::MatrixSlice;
    /// use rulinalg::matrix::slice::BaseSlice;
    ///
    /// let a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let slice = MatrixSlice::from_matrix(&a, [1,1], 2, 2);
    ///
    /// let slice_data = slice.iter().map(|v| *v).collect::<Vec<usize>>();
    /// assert_eq!(slice_data, vec![4,5,7,8]);
    /// ```
    fn iter<'a>(&self) -> SliceIter<'a, T> 
        where T: 'a
    {
        SliceIter {
            slice_start: self.as_ptr(),
            row_pos: 0,
            col_pos: 0,
            slice_rows: self.rows(),
            slice_cols: self.cols(),
            row_stride: self.row_stride(),
            _marker: PhantomData::<&T>,
        }
    }

    /// Iterate over the rows of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::slice::BaseSlice;
    ///
    /// let a = Matrix::new(3, 2, (0..6).collect::<Vec<usize>>());
    ///
    /// // Prints "2" three times.
    /// for row in a.iter_rows() {
    ///     println!("{}", row.len());
    /// }
    /// ```
    fn iter_rows(&self) -> Rows<T> {
        Rows {
            slice_start: self.as_ptr(),
            row_pos: 0,
            slice_rows: self.rows(),
            slice_cols: self.cols(),
            row_stride: self.row_stride() as isize,
            _marker: PhantomData::<&T>,
        }
    }

    /// Convert the matrix slice into a new Matrix.
    fn into_matrix(self) -> Matrix<T>
        where T: Copy
    {
        self.iter_rows().collect()
    }

    /// Select rows from matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::slice::BaseSlice;
    ///
    /// let a = Matrix::<f64>::ones(3,3);
    ///
    /// let b = &a.select_rows(&[2]);
    /// assert_eq!(b.rows(), 1);
    /// assert_eq!(b.cols(), 3);
    ///
    /// let c = &a.select_rows(&[1,2]);
    /// assert_eq!(c.rows(), 2);
    /// assert_eq!(c.cols(), 3);
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if row indices exceed the matrix dimensions.
    fn select_rows(&self, rows: &[usize]) -> Matrix<T> 
        where T: Copy
    {

        let mut mat_vec = Vec::with_capacity(rows.len() * self.cols());

        for row in rows {
            assert!(*row < self.rows(),
                    "Row index is greater than number of rows.");
        }

        for row in rows {
            unsafe {
                let slice = self.get_row_unchecked(*row);
                mat_vec.extend_from_slice(slice);
            }
        }

        Matrix {
            cols: self.cols(),
            rows: rows.len(),
            data: mat_vec,
        }
    }

    /// Select columns from matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::slice::BaseSlice;
    ///
    /// let a = Matrix::<f64>::ones(3,3);
    /// let b = &a.select_cols(&[2]);
    /// assert_eq!(b.rows(), 3);
    /// assert_eq!(b.cols(), 1);
    ///
    /// let c = &a.select_cols(&[1,2]);
    /// assert_eq!(c.rows(), 3);
    /// assert_eq!(c.cols(), 2);
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if column indices exceed the matrix dimensions.
    fn select_cols(&self, cols: &[usize]) -> Matrix<T>
        where T: Copy
    {
        let mut mat_vec = Vec::with_capacity(cols.len() * self.rows());

        for col in cols {
            assert!(*col < self.cols(),
                    "Column index is greater than number of columns.");
        }

        unsafe {
            for i in 0..self.rows() {
                for col in cols {
                    mat_vec.push(*self.get_unchecked([i, *col]));
                }
            }
        }

        Matrix {
            cols: cols.len(),
            rows: self.rows(),
            data: mat_vec,
        }
    }

    /// Select block matrix from matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::slice::BaseSlice;
    ///
    /// let a = Matrix::<f64>::identity(3);
    /// let b = &a.select(&[0,1], &[1,2]);
    ///
    /// // We get the 2x2 block matrix in the upper right corner.
    /// assert_eq!(b.rows(), 2);
    /// assert_eq!(b.cols(), 2);
    ///
    /// // Prints [0,0,1,0]
    /// println!("{:?}", b.data());
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if row or column indices exceed the matrix dimensions.
    fn select(&self, rows: &[usize], cols: &[usize]) -> Matrix<T>
        where T: Copy
    {

        let mut mat_vec = Vec::with_capacity(cols.len() * rows.len());

        for col in cols {
            assert!(*col < self.cols(),
                    "Column index is greater than number of columns.");
        }

        for row in rows {
            assert!(*row < self.rows(),
                    "Row index is greater than number of columns.");
        }

        unsafe {
            for row in rows {
                for col in cols {
                    mat_vec.push(*self.get_unchecked([*row, *col]));
                }
            }
        }

        Matrix {
            cols: cols.len(),
            rows: rows.len(),
            data: mat_vec,
        }
    }

    /// Horizontally concatenates two matrices. With self on the left.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::slice::BaseSlice;
    ///
    /// let a = Matrix::new(3,2, vec![1.0,2.0,3.0,4.0,5.0,6.0]);
    /// let b = Matrix::new(3,1, vec![4.0,5.0,6.0]);
    ///
    /// let c = &a.hcat(&b);
    /// assert_eq!(c.cols(), a.cols() + b.cols());
    /// assert_eq!(c[[1, 2]], 5.0);
    /// ```
    ///
    /// # Panics
    ///
    /// - Self and m have different row counts.
    fn hcat<S>(&self, m: &S) -> Matrix<T>
        where T: Copy,
              S: BaseSlice<T>,
    {
        assert!(self.rows() == m.rows(), "Matrix row counts are not equal.");

        let mut new_data = Vec::with_capacity((self.cols() + m.cols()) * self.rows());

        for (self_row, m_row) in self.iter_rows().zip(m.iter_rows()) {
            new_data.extend_from_slice(self_row);
            new_data.extend_from_slice(m_row);
        }

        Matrix {
            cols: (self.cols() + m.cols()),
            rows: self.rows(),
            data: new_data,
        }
    }

    /// Vertically concatenates two matrices. With self on top.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::slice::BaseSlice;
    ///
    /// let a = Matrix::new(2,3, vec![1.0,2.0,3.0,4.0,5.0,6.0]);
    /// let b = Matrix::new(1,3, vec![4.0,5.0,6.0]);
    ///
    /// let c = &a.vcat(&b);
    /// assert_eq!(c.rows(), a.rows() + b.rows());
    /// assert_eq!(c[[2, 2]], 6.0);
    /// ```
    ///
    /// # Panics
    ///
    /// - Self and m have different column counts.
    fn vcat<S>(&self, m: &S) -> Matrix<T>
        where T: Copy,
              S: BaseSlice<T>,
    {
        assert!(self.cols() == m.cols(), "Matrix column counts are not equal.");

        let mut new_data = Vec::with_capacity((self.rows() + m.rows()) * self.cols());

        for row in self.iter_rows().chain(m.iter_rows()) {
            new_data.extend_from_slice(row);
        }

        Matrix {
            cols: self.cols(),
            rows: (self.rows() + m.rows()),
            data: new_data,
        }
    }

    /// Extract the diagonal of the matrix
    ///
    /// Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::vector::Vector;
    /// use rulinalg::matrix::slice::BaseSlice;
    ///
    /// let a = Matrix::new(3,3,vec![1,2,3,4,5,6,7,8,9]);
    /// let b = Matrix::new(3,2,vec![1,2,3,4,5,6]);
    /// let c = Matrix::new(2,3,vec![1,2,3,4,5,6]);
    ///
    /// let d = &a.diag(); // 1,5,9
    /// let e = &b.diag(); // 1,4
    /// let f = &c.diag(); // 1,5
    ///
    /// assert_eq!(*d.data(), vec![1,5,9]);
    /// assert_eq!(*e.data(), vec![1,4]);
    /// assert_eq!(*f.data(), vec![1,5]);
    /// ```
    fn diag(&self) -> Vector<T>
        where T: Copy,
    {
        let mat_min = min(self.rows(), self.cols());

        let mut diagonal = Vec::with_capacity(mat_min);
        unsafe {
            for i in 0..mat_min {
                diagonal.push(*self.get_unchecked([i, i]));
            }
        }
        Vector::new(diagonal)
    }

    /// Tranposes the given matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::slice::BaseSlice;
    ///
    /// let mat = Matrix::new(2,3, vec![1.0,2.0,3.0,4.0,5.0,6.0]);
    ///
    /// let mt = mat.transpose();
    /// ```
    fn transpose(&self) -> Matrix<T>
        where T: Copy,
    {
        let mut new_data = Vec::with_capacity(self.rows() * self.cols());

        unsafe {
            new_data.set_len(self.rows() * self.cols());
            for i in 0..self.cols() {
                for j in 0..self.rows() {
                    *new_data.get_unchecked_mut(i * self.rows() + j) = 
                        *self.get_unchecked([j, i]);
                }
            }
        }

        Matrix {
            cols: self.rows(),
            rows: self.cols(),
            data: new_data,
        }
    }

    /// Checks if matrix is diagonal.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::slice::BaseSlice;
    ///
    /// let a = Matrix::new(2,2, vec![1.0,0.0,0.0,1.0]);
    /// let a_diag = a.is_diag();
    ///
    /// assert_eq!(a_diag, true);
    ///
    /// let b = Matrix::new(2,2, vec![1.0,0.0,1.0,0.0]);
    /// let b_diag = b.is_diag();
    ///
    /// assert_eq!(b_diag, false);
    /// ```
    fn is_diag(&self) -> bool 
        where T: Zero + PartialEq,
    {
        let mut next_diag = 0usize;
        self.iter().enumerate().all(|(i, data)| if i == next_diag {
            next_diag += self.cols() + 1;
            true
        } else {
            data == &T::zero()
        })
    }

    /// Solves an upper triangular linear system.
    ///
    /// Given a matrix `U`, which is upper triangular, and a vector `y`, this function returns `x`
    /// such that `Ux = y`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::vector::Vector;
    /// use rulinalg::matrix::slice::BaseSlice;
    /// use std::f32;
    ///
    /// let u = Matrix::new(2,2, vec![1.0, 2.0, 0.0, 1.0]);
    /// let y = Vector::new(vec![3.0, 1.0]);
    ///
    /// let x = u.solve_u_triangular(y).expect("A solution should exist!");
    /// assert!((x[0] - 1.0) < f32::EPSILON);
    /// assert!((x[1] - 1.0) < f32::EPSILON);
    /// ```
    ///
    /// # Panics
    ///
    /// - Vector size and matrix column count are not equal.
    /// - Matrix is not upper triangular.
    ///
    /// # Failures
    ///
    /// Fails if there is no valid solution to the system (matrix is singular).
    fn solve_u_triangular(&self, y: Vector<T>) -> Result<Vector<T>, Error>
        where T: Any + Float,
    {
        assert!(self.cols() == y.size(),
                format!("Vector size {0} != {1} Matrix column count.",
                        y.size(),
                        self.cols()));

        // Make sure we are upper triangular.
        for (row_idx, row) in self.iter_rows().enumerate() {
            if row.iter().take(row_idx).any(|data| data != &T::zero()) {
                panic!("Matrix is not upper triangular");
            }
        }

        back_substitution(self, y)
    }

    /// Solves a lower triangular linear system.
    ///
    /// Given a matrix `L`, which is lower triangular, and a vector `y`, this function returns `x`
    /// such that `Lx = y`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::vector::Vector;
    /// use rulinalg::matrix::slice::BaseSlice;
    /// use std::f32;
    ///
    /// let l = Matrix::new(2,2, vec![1.0, 0.0, 2.0, 1.0]);
    /// let y = Vector::new(vec![1.0, 3.0]);
    ///
    /// let x = l.solve_l_triangular(y).expect("A solution should exist!");
    /// println!("{:?}", x);
    /// assert!((x[0] - 1.0) < f32::EPSILON);
    /// assert!((x[1] - 1.0) < f32::EPSILON);
    /// ```
    ///
    /// # Panics
    ///
    /// - Vector size and matrix column count are not equal.
    /// - Matrix is not lower triangular.
    ///
    /// # Failures
    ///
    /// Fails if there is no valid solution to the system (matrix is singular).
    fn solve_l_triangular(&self, y: Vector<T>) -> Result<Vector<T>, Error>
        where T: Any + Float,
    {
        assert!(self.cols() == y.size(),
                format!("Vector size {0} != {1} Matrix column count.",
                        y.size(),
                        self.cols()));

        // Make sure we are lower triangular.
        for (row_idx, row) in self.iter_rows().enumerate() {
            if row.iter().skip(row_idx + 1).any(|data| data != &T::zero()) {
                panic!("Matrix is not lower triangular.");
            }
        }

        forward_substitution(self, y)
    }
}

/// Trait for Mutable Matrix Slices.
pub trait BaseSliceMut<T>: BaseSlice<T> {

    /// Top left index of the slice.
    fn as_mut_ptr(&mut self) -> *mut T;

    /// Get a mutable reference to a point in the matrix without bounds checks.
    unsafe fn get_unchecked_mut(&mut self, index: [usize; 2]) -> &mut T {
        &mut *(self.as_mut_ptr().offset((index[0] * self.row_stride() + index[1]) as isize))
    }

    /// Returns a mutable iterator over the matrix slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::MatrixSliceMut;
    /// use rulinalg::matrix::slice::BaseSliceMut;
    ///
    /// let mut a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    ///
    /// {
    ///     let mut slice = MatrixSliceMut::from_matrix(&mut a, [1,1], 2, 2);
    ///
    ///     for d in slice.iter_mut() {
    ///         *d = *d + 2;
    ///     }
    /// }
    ///
    /// // Only the matrix slice is updated.
    /// assert_eq!(a.into_vec(), vec![0,1,2,3,6,7,6,9,10]);
    /// ```
    fn iter_mut<'a>(&mut self) -> SliceIterMut<'a, T> 
        where T: 'a,
    {
        SliceIterMut {
            slice_start: self.as_mut_ptr(),
            row_pos: 0,
            col_pos: 0,
            slice_rows: self.rows(),
            slice_cols: self.cols(),
            row_stride: self.row_stride(),
            _marker: PhantomData::<&mut T>,
        }
    }

    /// Returns a mutable reference to the row of a `MatrixSliceMut` at the given index.
    /// `None` if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::MatrixSliceMut;
    /// use rulinalg::matrix::slice::BaseSliceMut;
    ///
    /// let mut a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let mut slice = MatrixSliceMut::from_matrix(&mut a, [1,1], 2, 2);
    /// {
    ///     let row = slice.get_row_mut(1);
    ///     let mut expected = vec![7usize, 8];
    ///     assert_eq!(row, Some(&mut *expected));
    /// }
    /// assert!(slice.get_row_mut(5).is_none());
    /// ```
    fn get_row_mut(&mut self, index: usize) -> Option<&mut [T]> {
        if index < self.rows() {
            unsafe { Some(self.get_row_unchecked_mut(index)) }
        } else {
            None
        }
    }

    /// Returns a mutable reference to the row of a `MatrixSliceMut` at the given index
    /// without doing unbounds checking
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::MatrixSliceMut;
    /// use rulinalg::matrix::slice::BaseSliceMut;
    ///
    /// let mut a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let mut slice = MatrixSliceMut::from_matrix(&mut a, [1,1], 2, 2);
    /// let row = unsafe { slice.get_row_unchecked_mut(1) };
    /// let mut expected = vec![7usize, 8];
    /// assert_eq!(row, &mut *expected);
    /// ```
    unsafe fn get_row_unchecked_mut(&mut self, index: usize) -> &mut [T] {
        let ptr = self.as_mut_ptr().offset((self.row_stride() * index) as isize);
        ::std::slice::from_raw_parts_mut(ptr, self.cols())
    }

    /// Iterate over the mutable rows of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::slice::BaseSliceMut;
    ///
    /// let mut a = Matrix::new(3, 2, (0..6).collect::<Vec<usize>>());
    ///
    /// for row in a.iter_rows_mut() {
    ///     for r in row {
    ///         *r = *r + 1;
    ///     }
    /// }
    ///
    /// // Now contains the range 1..7
    /// println!("{}", a);
    /// ```
    fn iter_rows_mut(&mut self) -> RowsMut<T> {
        RowsMut {
            slice_start: self.as_mut_ptr(),
            row_pos: 0,
            slice_rows: self.rows(),
            slice_cols: self.cols(),
            row_stride: self.row_stride() as isize,
            _marker: PhantomData::<&mut T>,
        }
    }
}

impl<T> BaseSlice<T> for Matrix<T> {
    fn rows(&self) -> usize { self.rows } 
    fn cols(&self) -> usize { self.cols } 
    fn row_stride(&self) -> usize { self.cols } 
    fn as_ptr(&self) -> *const T { self.data.as_ptr() }

    fn into_matrix(self) -> Matrix<T>
        where T: Copy
    {
        // for Matrix, this is a no-op
        self
    }
}

impl<'a, T> BaseSlice<T> for MatrixSlice<'a, T> {
    fn rows(&self) -> usize { self.rows } 
    fn cols(&self) -> usize { self.cols } 
    fn row_stride(&self) -> usize { self.row_stride } 
    fn as_ptr(&self) -> *const T { self.ptr }
}

impl<'a, T> BaseSlice<T> for MatrixSliceMut<'a, T> {
    fn rows(&self) -> usize { self.rows } 
    fn cols(&self) -> usize { self.cols } 
    fn row_stride(&self) -> usize { self.row_stride } 
    fn as_ptr(&self) -> *const T { self.ptr as *const T }
}

impl<T> BaseSliceMut<T> for Matrix<T> {
    /// Top left index of the slice.
    fn as_mut_ptr(&mut self) -> *mut T { self.data.as_mut_ptr() }
}

impl<'a, T> BaseSliceMut<T> for MatrixSliceMut<'a, T> {
    /// Top left index of the slice.
    fn as_mut_ptr(&mut self) -> *mut T { self.ptr }
}

impl<'a, T> MatrixSlice<'a, T> {
    /// Produce a matrix slice from a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::MatrixSlice;
    ///
    /// let a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let slice = MatrixSlice::from_matrix(&a, [1,1], 2, 2);
    /// ```
    pub fn from_matrix(mat: &'a Matrix<T>,
                       start: [usize; 2],
                       rows: usize,
                       cols: usize)
                       -> MatrixSlice<T> {
        assert!(start[0] + rows <= mat.rows(),
                "View dimensions exceed matrix dimensions.");
        assert!(start[1] + cols <= mat.cols(),
                "View dimensions exceed matrix dimensions.");
        unsafe {
            MatrixSlice {
                ptr: mat.data().get_unchecked(start[0] * mat.cols + start[1]) as *const T,
                rows: rows,
                cols: cols,
                row_stride: mat.cols,
                marker: PhantomData::<&'a T>,
            }
        }
    }

    /// Creates a matrix slice from raw parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::MatrixSlice;
    ///
    /// let mut a = vec![4.0; 16];
    ///
    /// unsafe {
    ///     // Create a matrix slice with 3 rows, and 3 cols
    ///     // The row stride of 4 specifies the distance between the start of each row in the data.
    ///     let b = MatrixSlice::from_raw_parts(a.as_ptr(), 3, 3, 4);
    /// }
    /// ```
    ///
    /// # Safety
    ///
    /// The pointer must be followed by a contiguous slice of data larger than `row_stride * rows`.
    /// If not then other operations will produce undefined behaviour.
    ///
    /// Additionally `cols` should be less than the `row_stride`. It is possible to use this
    /// function safely whilst violating this condition. So long as
    /// `max(cols, row_stride) * rows` is less than the data size.
    pub unsafe fn from_raw_parts(ptr: *const T,
                                 rows: usize,
                                 cols: usize,
                                 row_stride: usize)
                                 -> MatrixSlice<'a, T> {
        MatrixSlice {
            ptr: ptr,
            rows: rows,
            cols: cols,
            row_stride: row_stride,
            marker: PhantomData::<&'a T>,
        }
    }

    /// Produce a matrix slice from an existing matrix slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::MatrixSlice;
    ///
    /// let a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let slice = MatrixSlice::from_matrix(&a, [1,1], 2, 2);
    /// let new_slice = slice.reslice([0,0], 1, 1);
    /// ```
    pub fn reslice(mut self, start: [usize; 2], rows: usize, cols: usize) -> MatrixSlice<'a, T> {
        assert!(start[0] + rows <= self.rows,
                "View dimensions exceed matrix dimensions.");
        assert!(start[1] + cols <= self.cols,
                "View dimensions exceed matrix dimensions.");

        unsafe {
            self.ptr = self.ptr.offset((start[0] * self.cols + start[1]) as isize);
        }
        self.rows = rows;
        self.cols = cols;

        self
    }

}

impl<'a, T> MatrixSliceMut<'a, T> {
    /// Produce a matrix slice from a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::MatrixSliceMut;
    ///
    /// let mut a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let slice = MatrixSliceMut::from_matrix(&mut a, [1,1], 2, 2);
    /// ```
    pub fn from_matrix(mat: &'a mut Matrix<T>,
                       start: [usize; 2],
                       rows: usize,
                       cols: usize)
                       -> MatrixSliceMut<T> {
        assert!(start[0] + rows <= mat.rows(),
                "View dimensions exceed matrix dimensions.");
        assert!(start[1] + cols <= mat.cols(),
                "View dimensions exceed matrix dimensions.");

        let mat_cols = mat.cols();

        unsafe {
            MatrixSliceMut {
                ptr: mat.mut_data().get_unchecked_mut(start[0] * mat_cols + start[1]) as *mut T,
                rows: rows,
                cols: cols,
                row_stride: mat_cols,
                marker: PhantomData::<&'a mut T>,
            }
        }
    }

    /// Creates a mutable matrix slice from raw parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::MatrixSliceMut;
    ///
    /// let mut a = vec![4.0; 16];
    ///
    /// unsafe {
    ///     // Create a mutable matrix slice with 3 rows, and 3 cols
    ///     // The row stride of 4 specifies the distance between the start of each row in the data.
    ///     let b = MatrixSliceMut::from_raw_parts(a.as_mut_ptr(), 3, 3, 4);
    /// }
    /// ```
    ///
    /// # Safety
    ///
    /// The pointer must be followed by a contiguous slice of data larger than `row_stride * rows`.
    /// If not then other operations will produce undefined behaviour.
    ///
    /// Additionally `cols` should be less than the `row_stride`. It is possible to use this
    /// function safely whilst violating this condition. So long as
    /// `max(cols, row_stride) * rows` is less than the data size.
    pub unsafe fn from_raw_parts(ptr: *mut T,
                                 rows: usize,
                                 cols: usize,
                                 row_stride: usize)
                                 -> MatrixSliceMut<'a, T> {
        MatrixSliceMut {
            ptr: ptr,
            rows: rows,
            cols: cols,
            row_stride: row_stride,
            marker: PhantomData::<&'a mut T>,
        }
    }

    /// Produce a matrix slice from an existing matrix slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::MatrixSliceMut;
    ///
    /// let mut a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let slice = MatrixSliceMut::from_matrix(&mut a, [1,1], 2, 2);
    /// let new_slice = slice.reslice([0,0], 1, 1);
    /// ```
    pub fn reslice(mut self, start: [usize; 2], rows: usize, cols: usize) -> MatrixSliceMut<'a, T> {
        assert!(start[0] + rows <= self.rows,
                "View dimensions exceed matrix dimensions.");
        assert!(start[1] + cols <= self.cols,
                "View dimensions exceed matrix dimensions.");

        unsafe {
            self.ptr = self.ptr.offset((start[0] * self.cols + start[1]) as isize);
        }
        self.rows = rows;
        self.cols = cols;

        self
    }
}

impl<'a, T: Copy> MatrixSliceMut<'a, T> {

    /// Sets the underlying matrix data to the target data.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, MatrixSliceMut};
    ///
    /// let mut mat = Matrix::<f32>::zeros(4,4);
    /// let one_block = Matrix::<f32>::ones(2,2);
    ///
    /// // Get a mutable slice of the upper left 2x2 block.
    /// let mat_block = MatrixSliceMut::from_matrix(&mut mat, [0,0], 2, 2);
    ///
    /// // Set the upper left 2x2 block to be ones.
    /// mat_block.set_to(one_block.as_slice());
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the dimensions of `self` and `target` are not the same.
    pub fn set_to(mut self, target: MatrixSlice<T>) {
        // TODO: Should this method take an Into<MatrixSlice> or something similar?
        // So we can use `Matrix` and `MatrixSlice` and `MatrixSliceMut`.
        assert!(self.rows == target.rows,
                "Target has different row count to self.");
        assert!(self.cols == target.cols,
                "Target has different column count to self.");
        for (s, t) in self.iter_rows_mut().zip(target.iter_rows()) {
            // Vectorized assignment per row.
            utils::in_place_vec_bin_op(s, t, |x, &y| *x = y);
        }
    }
}

/// Iterator for `MatrixSlice`
///
/// Iterates over the underlying slice data
/// in row-major order.
#[derive(Debug)]
pub struct SliceIter<'a, T: 'a> {
    slice_start: *const T,
    row_pos: usize,
    col_pos: usize,
    slice_rows: usize,
    slice_cols: usize,
    row_stride: usize,
    _marker: PhantomData<&'a T>,
}

/// Iterator for `MatrixSliceMut`.
///
/// Iterates over the underlying slice data
/// in row-major order.
#[derive(Debug)]
pub struct SliceIterMut<'a, T: 'a> {
    slice_start: *mut T,
    row_pos: usize,
    col_pos: usize,
    slice_rows: usize,
    slice_cols: usize,
    row_stride: usize,
    _marker: PhantomData<&'a mut T>,
}

macro_rules! impl_slice_iter (
    ($slice_iter:ident, $data_type:ty) => (
/// Iterates over the matrix slice data in row-major order.
impl<'a, T> Iterator for $slice_iter<'a, T> {
    type Item = $data_type;

    fn next(&mut self) -> Option<$data_type> {
        // Set the position of the next element
        if self.row_pos < self.slice_rows {
            unsafe {
                let iter_ptr = self.slice_start.offset((
                                self.row_pos * self.row_stride + self.col_pos)
                                as isize);

                // If end of row, set to start of next row
                if self.col_pos == self.slice_cols - 1 {
                    self.row_pos += 1usize;
                    self.col_pos = 0usize;
                } else {
                    self.col_pos += 1usize;
                }

                Some(mem::transmute(iter_ptr))
            }
        } else {
            None
        }
    }
}
    );
);

impl_slice_iter!(SliceIter, &'a T);
impl_slice_iter!(SliceIterMut, &'a mut T);

#[cfg(test)]
mod tests {
    use super::BaseSlice;
    use super::super::MatrixSlice;
    use super::super::MatrixSliceMut;
    use super::super::Matrix;

    #[test]
    #[should_panic]
    fn make_slice_bad_dim() {
        let a = Matrix::new(3, 3, vec![2.0; 9]);
        let _ = MatrixSlice::from_matrix(&a, [1, 1], 3, 2);
    }

    #[test]
    fn make_slice() {
        let a = Matrix::new(3, 3, vec![2.0; 9]);
        let b = MatrixSlice::from_matrix(&a, [1, 1], 2, 2);

        assert_eq!(b.rows(), 2);
        assert_eq!(b.cols(), 2);
    }

    #[test]
    fn reslice() {
        let mut a = Matrix::new(4, 4, (0..16).collect::<Vec<_>>());

        {
            let b = MatrixSlice::from_matrix(&a, [1, 1], 3, 3);
            let c = b.reslice([0, 1], 2, 2);

            assert_eq!(c.rows(), 2);
            assert_eq!(c.cols(), 2);

            assert_eq!(c[[0, 0]], 6);
            assert_eq!(c[[0, 1]], 7);
            assert_eq!(c[[1, 0]], 10);
            assert_eq!(c[[1, 1]], 11);
        }

        let b = MatrixSliceMut::from_matrix(&mut a, [1, 1], 3, 3);

        let c = b.reslice([0, 1], 2, 2);

        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 2);

        assert_eq!(c[[0, 0]], 6);
        assert_eq!(c[[0, 1]], 7);
        assert_eq!(c[[1, 0]], 10);
        assert_eq!(c[[1, 1]], 11);
    }

    #[test]
    fn slice_into_matrix() {
        let mut a = Matrix::new(3, 3, vec![2.0; 9]);

        {
            let b = MatrixSlice::from_matrix(&a, [1, 1], 2, 2);
            let c = b.into_matrix();
            assert_eq!(c.rows(), 2);
            assert_eq!(c.cols(), 2);
        }

        let d = MatrixSliceMut::from_matrix(&mut a, [1, 1], 2, 2);
        let e = d.into_matrix();
        assert_eq!(e.rows(), 2);
        assert_eq!(e.cols(), 2);
    }
}
