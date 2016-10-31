//! CSC matrix addition module.
//!
//! # Examples
//!
//! ```
//! use rulinalg::matrix::Matrix;
//! use rulinalg::sparse_matrix::{CompressedMatrix, CscMatrix};
//! use rulinalg::vector::Vector;
//!
//! let a = CscMatrix::new(5, 4, vec![1, 2, 3], vec![0, 0, 0], vec![0, 1, 2, 3, 3]);
//! let b = Matrix::new(5, 4, (1..21).collect::<Vec<i32>>());
//! let c = Vector::new(vec![1, 2, 3, 4, 5]);
//!
//! let _ = a.clone() + 10;
//! let _ = a.clone() + b;
//! let _ = a + c;
//! ```

use std::ops::Add;

use libnum::{One, Zero};

use matrix::{BaseMatrix, Matrix};
use vector::Vector;
use sparse_matrix::{CompressedMatrix, CscMatrix, SparseMatrix};

/// CSC matrix plus dense matrix.
impl<T> Add<Matrix<T>> for CscMatrix<T>
    where T: Copy + Add<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn add(self, matrix: Matrix<T>) -> CscMatrix<T> {
        &self + &matrix
    }
}
/// CSC matrix plus dense matrix.
impl<'a, T> Add<&'a Matrix<T>> for CscMatrix<T>
    where T: Copy + Add<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn add(self, matrix: &Matrix<T>) -> CscMatrix<T> {
        &self + matrix
    }
}
/// CSC matrix plus dense matrix.
impl<'a, T> Add<Matrix<T>> for &'a CscMatrix<T>
    where T: Copy + Add<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn add(self, matrix: Matrix<T>) -> CscMatrix<T> {
        self + &matrix
    }
}
/// CSC matrix plus dense matrix.
impl<'a, 'b, T> Add<&'b Matrix<T>> for &'a CscMatrix<T>
    where T: Copy + Add<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn add(self, matrix: &Matrix<T>) -> CscMatrix<T> {
        assert!(self.cols() == matrix.cols() && self.rows() == matrix.rows(),
                "The shape of the dense matrix must be equal the shape of the CSC matrix");

        let nnz = self.rows() * self.cols();
        let mut data: Vec<T> = Vec::with_capacity(nnz);
        let mut indices: Vec<usize> = Vec::with_capacity(nnz);
        let mut ptrs: Vec<usize> = Vec::with_capacity(self.cols() + 1);

        ptrs.push(0);

        for col_idx in 0..self.cols() {
            for row_idx in 0..self.rows() {
                let value;

                unsafe {
                    value = self.get(row_idx, col_idx) + *matrix.get_unchecked([row_idx, col_idx]);
                }

                if !value.is_zero() {
                    data.push(value);
                    indices.push(row_idx);
                }
            }

            ptrs.push(data.len());
        }

        CscMatrix::new(self.rows(), self.cols(), data, indices, ptrs)
    }
}

/// CSC matrix plus dense vector.
impl<T> Add<Vector<T>> for CscMatrix<T>
    where T: Copy + Add<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn add(self, vector: Vector<T>) -> CscMatrix<T> {
        &self + &vector
    }
}
/// / CSC matrix plus dense vector.
impl<'a, T> Add<&'a Vector<T>> for CscMatrix<T>
    where T: Copy + Add<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn add(self, vector: &Vector<T>) -> CscMatrix<T> {
        &self + vector
    }
}
/// CSC matrix plus dense vector.
impl<'a, T> Add<Vector<T>> for &'a CscMatrix<T>
    where T: Copy + Add<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn add(self, vector: Vector<T>) -> CscMatrix<T> {
        self + &vector
    }
}
/// CSC matrix plus dense vector.
impl<'a, 'b, T> Add<&'b Vector<T>> for &'a CscMatrix<T>
    where T: Copy + Add<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn add(self, vector: &Vector<T>) -> CscMatrix<T> {
        assert!(self.rows() == vector.size(),
                "The size of the vector must be equal the size of the CSC matrix rows");

        let nnz = self.rows() * self.cols();
        let vector_data = vector.data();
        let mut data: Vec<T> = Vec::with_capacity(nnz);
        let mut indices: Vec<usize> = Vec::with_capacity(nnz);
        let mut ptrs: Vec<usize> = Vec::with_capacity(self.cols() + 1);

        ptrs.push(0);

        for col_idx in 0..self.cols() {
            for row_idx in 0..self.rows() {
                let value = self.get(row_idx, col_idx) + vector_data[row_idx];

                if !value.is_zero() {
                    data.push(value);
                    indices.push(row_idx);
                }
            }

            ptrs.push(data.len());
        }

        CscMatrix::new(self.rows(), self.cols(), data, indices, ptrs)
    }
}


/// CSC matrix plus scalar.
impl<T> Add<T> for CscMatrix<T>
    where T: Copy + Add<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn add(self, scalar: T) -> CscMatrix<T> {
        self + &scalar
    }
}
/// CSC matrix plus scalar.
impl<'a, T> Add<&'a T> for CscMatrix<T>
    where T: Copy + Add<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn add(self, scalar: &T) -> CscMatrix<T> {
        &self + scalar
    }
}
/// CSC matrix plus scalar.
impl<'a, T> Add<T> for &'a CscMatrix<T>
    where T: Copy + Add<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn add(self, scalar: T) -> CscMatrix<T> {
        self + &scalar
    }
}
/// CSC matrix plus scalar.
impl<'a, 'b, T> Add<&'b T> for &'a CscMatrix<T>
    where T: Copy + Add<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn add(self, scalar: &T) -> CscMatrix<T> {
        let nnz = self.rows() * self.cols();
        let mut data: Vec<T> = Vec::with_capacity(nnz);
        let mut indices: Vec<usize> = Vec::with_capacity(nnz);
        let mut ptrs: Vec<usize> = Vec::with_capacity(self.cols() + 1);

        ptrs.push(0);

        for col_idx in 0..self.cols() {
            for row_idx in 0..self.rows() {
                let value = self.get(row_idx, col_idx) + *scalar;

                if !value.is_zero() {
                    data.push(value);
                    indices.push(row_idx);
                }
            }

            ptrs.push(data.len());
        }

        CscMatrix::new(self.rows(), self.cols(), data, indices, ptrs)
    }
}

#[cfg(test)]
mod tests {
    use matrix::Matrix;
    use sparse_matrix::{CompressedMatrix, CscMatrix};
    use vector::Vector;

    #[test]
    fn test_csc_add_matrix() {
        let a = CscMatrix::new(5,
                               5,
                               vec![4, 2, 3, 5],
                               vec![1, 2, 0, 3],
                               vec![0, 0, 2, 2, 4, 4]);

        let b = Matrix::new(5,
                            5,
                            vec![6, 5, 7, 9, 2, 8, 4, 5, 1, 1, 8, 9, 5, 4, 6, 9, 7, 2, 9, 8, 1,
                                 3, 5, 7, 9]);

        let c = CscMatrix::new(5,
                               5,
                               vec![6, 8, 8, 9, 1, 5, 8, 11, 7, 3, 7, 5, 5, 2, 5, 12, 1, 4, 14,
                                    7, 2, 1, 6, 8, 9],
                               vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
                                    0, 1, 2, 3, 4],
                               vec![0, 5, 10, 15, 20, 25]);

        assert_eq!(a + b, c);
    }

    #[test]
    fn test_csc_add_scalar() {
        let a = CscMatrix::new(5,
                               5,
                               vec![4, 2, 3, 5],
                               vec![1, 2, 0, 3],
                               vec![0, 0, 2, 2, 4, 4]);

        let b = CscMatrix::new(5,
                               5,
                               vec![2, 2, 2, 2, 2, 2, 6, 4, 2, 2, 2, 2, 2, 2, 2, 5, 2, 2, 7, 2,
                                    2, 2, 2, 2, 2],
                               vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
                                    0, 1, 2, 3, 4],
                               vec![0, 5, 10, 15, 20, 25]);

        assert_eq!(a + 2, b);
    }

    #[test]
    fn test_csc_add_vector() {
        let a = CscMatrix::new(5,
                               5,
                               vec![4, 2, 3, 5],
                               vec![1, 2, 0, 3],
                               vec![0, 0, 2, 2, 4, 4]);

        let b = Vector::new(vec![1, 4, 2, 8, 5]);

        let c = CscMatrix::new(5,
                               5,
                               vec![1, 4, 2, 8, 5, 1, 8, 4, 8, 5, 1, 4, 2, 8, 5, 4, 4, 2, 13, 5,
                                    1, 4, 2, 8, 5],
                               vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
                                    0, 1, 2, 3, 4],
                               vec![0, 5, 10, 15, 20, 25]);

        assert_eq!(a + b, c);
    }
}
