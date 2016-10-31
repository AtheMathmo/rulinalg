//! CSC matrix division module.
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
//! let _ = a.clone() / 10;
//! let _ = a.clone() / b;
//! let _ = a / c;
//! ```

use std::ops::Div;

use libnum::{One, Zero};

use matrix::{BaseMatrix, Matrix};
use vector::Vector;
use sparse_matrix::{CompressedMatrix, CscMatrix, SparseMatrix};

/// CSC matrix divided by dense matrix.
impl<T> Div<Matrix<T>> for CscMatrix<T>
    where T: Copy + Div<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn div(self, matrix: Matrix<T>) -> CscMatrix<T> {
        &self / &matrix
    }
}
/// CSC matrix divided by dense matrix.
impl<'a, T> Div<&'a Matrix<T>> for CscMatrix<T>
    where T: Copy + Div<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn div(self, matrix: &Matrix<T>) -> CscMatrix<T> {
        &self / matrix
    }
}
/// CSC matrix divided by dense matrix.
impl<'a, T> Div<Matrix<T>> for &'a CscMatrix<T>
    where T: Copy + Div<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn div(self, matrix: Matrix<T>) -> CscMatrix<T> {
        self / &matrix
    }
}
/// CSC matrix divided by dense matrix.
impl<'a, 'b, T> Div<&'b Matrix<T>> for &'a CscMatrix<T>
    where T: Copy + Div<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn div(self, matrix: &Matrix<T>) -> CscMatrix<T> {
        assert!(self.cols() == matrix.cols() && self.rows() == matrix.rows(),
                "The shape of the dense matrix must be equal the shape of the CSC matrix");

        let nnz = self.nnz();
        let mut data: Vec<T> = Vec::with_capacity(nnz);
        let mut indices: Vec<usize> = Vec::with_capacity(nnz);
        let mut ptrs: Vec<usize> = Vec::with_capacity(self.cols() + 1);

        ptrs.push(0);

        for (col_idx, (col_rows, col_data)) in self.iter_linear().enumerate() {
            for idx in 0..col_data.len() {
                let value: T;

                unsafe {
                    value = col_data[idx] / *matrix.get_unchecked([col_rows[idx], col_idx]);
                };

                // Check if value is zero truncated
                if !value.is_zero() {
                    data.push(value);
                    indices.push(col_rows[idx]);
                }
            }

            ptrs.push(data.len());
        }

        CscMatrix::new(self.rows(), self.cols(), data, indices, ptrs)
    }
}

/// CSC matrix divided by dense vector.
impl<T> Div<Vector<T>> for CscMatrix<T>
    where T: Copy + Div<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn div(self, vector: Vector<T>) -> CscMatrix<T> {
        &self / &vector
    }
}
/// / CSC matrix divided by dense vector.
impl<'a, T> Div<&'a Vector<T>> for CscMatrix<T>
    where T: Copy + Div<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn div(self, vector: &Vector<T>) -> CscMatrix<T> {
        &self / vector
    }
}
/// CSC matrix divided by dense vector.
impl<'a, T> Div<Vector<T>> for &'a CscMatrix<T>
    where T: Copy + Div<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn div(self, vector: Vector<T>) -> CscMatrix<T> {
        self / &vector
    }
}
/// CSC matrix divided by dense vector.
impl<'a, 'b, T> Div<&'b Vector<T>> for &'a CscMatrix<T>
    where T: Copy + Div<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn div(self, vector: &Vector<T>) -> CscMatrix<T> {
        assert!(self.rows() == vector.size(),
                "The size of the vector must be equal the size of the CSC matrix rows");

        let nnz = self.nnz();
        let vector_data = vector.data();
        let mut data: Vec<T> = Vec::with_capacity(nnz);
        let mut indices: Vec<usize> = Vec::with_capacity(nnz);
        let mut ptrs: Vec<usize> = Vec::with_capacity(self.cols() + 1);

        ptrs.push(0);

        for (col_rows, col_data) in self.iter_linear() {
            for idx in 0..col_data.len() {
                let value = col_data[idx] / vector_data[col_rows[idx]];

                // Check if value is zero truncated
                if !value.is_zero() {
                    data.push(value);
                    indices.push(col_rows[idx]);
                }
            }

            ptrs.push(data.len());
        }

        CscMatrix::new(self.rows(), self.cols(), data, indices, ptrs)
    }
}


/// CSC matrix divided by scalar.
impl<T> Div<T> for CscMatrix<T>
    where T: Copy + Div<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn div(self, scalar: T) -> CscMatrix<T> {
        self / &scalar
    }
}
/// CSC matrix divided by scalar.
impl<'a, T> Div<&'a T> for CscMatrix<T>
    where T: Copy + Div<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn div(self, scalar: &T) -> CscMatrix<T> {
        &self / scalar
    }
}
/// CSC matrix divided by scalar.
impl<'a, T> Div<T> for &'a CscMatrix<T>
    where T: Copy + Div<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn div(self, scalar: T) -> CscMatrix<T> {
        self / &scalar
    }
}
/// CSC matrix divided by scalar.
impl<'a, 'b, T> Div<&'b T> for &'a CscMatrix<T>
    where T: Copy + Div<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn div(self, scalar: &T) -> CscMatrix<T> {
        let nnz = self.nnz();
        let mut data: Vec<T> = Vec::with_capacity(nnz);
        let mut indices: Vec<usize> = Vec::with_capacity(nnz);
        let mut ptrs: Vec<usize> = Vec::with_capacity(self.cols() + 1);

        ptrs.push(0);

        for (col_rows, col_data) in self.iter_linear() {
            for idx in 0..col_data.len() {
                println!("{}", idx);
                let value = col_data[idx] / *scalar;

                // Check if value is zero truncated
                if !value.is_zero() {
                    data.push(value);
                    indices.push(col_rows[idx]);
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
    fn test_csc_div_matrix() {
        let a = CscMatrix::new(5,
                               5,
                               vec![4, 2, 3, 5],
                               vec![1, 2, 0, 3],
                               vec![0, 0, 2, 2, 4, 4]);

        let b = Matrix::new(5,
                            5,
                            vec![6, 5, 7, 9, 2, 8, 4, 5, 1, 1, 8, 9, 5, 4, 6, 9, 7, 2, 9, 8, 1,
                                 3, 5, 7, 9]);

        let c = CscMatrix::new(5, 5, vec![1], vec![1], vec![0, 0, 1, 1, 1, 1]);

        assert_eq!(a / b, c);
    }

    #[test]
    fn test_csc_div_scalar() {
        let a = CscMatrix::new(5,
                               5,
                               vec![4, 2, 3, 5],
                               vec![1, 2, 0, 3],
                               vec![0, 0, 2, 2, 4, 4]);

        let b = CscMatrix::new(5,
                               5,
                               vec![2, 1, 1, 2],
                               vec![1, 2, 0, 3],
                               vec![0, 0, 2, 2, 4, 4]);

        assert_eq!(a / 2, b);
    }

    #[test]
    fn test_csc_div_vector() {
        let a = CscMatrix::new(5,
                               5,
                               vec![4, 2, 3, 5],
                               vec![1, 2, 0, 3],
                               vec![0, 0, 2, 2, 4, 4]);

        let b = Vector::new(vec![1, 4, 2, 8, 5]);

        let c = CscMatrix::new(5, 5, vec![1, 1, 3], vec![1, 2, 0], vec![0, 0, 2, 2, 3, 3]);

        assert_eq!(a / b, c);
    }
}
