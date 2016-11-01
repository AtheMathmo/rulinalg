//! CSC matrix multiplication module.
//!
//! # Examples
//!
//! ```
//! use rulinalg::matrix::Matrix;
//! use rulinalg::sparse_matrix::{CompressedMatrix, CscMatrix};
//! use rulinalg::vector::Vector;
//!
//! let a = CscMatrix::new(5, 4, vec![1, 2, 3], vec![0, 0, 0], vec![0, 1, 2, 3, 3]);
//! let b = Matrix::new(4, 4, (1..17).collect::<Vec<i32>>());
//! let c = Vector::new(vec![1, 2, 3, 4, 5]);
//!
//! let _ = a.clone() * 10;
//! let _ = a.clone() * b;
//! let _ = a * c;
//! ```

use std::ops::Mul;

use libnum::{One, Zero};

use matrix::{BaseMatrix, Matrix};
use vector::Vector;
use sparse_matrix::{CompressedMatrix, CscMatrix, SparseMatrix};

/// CSC matrix multiplied by dense matrix.
impl<T> Mul<Matrix<T>> for CscMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn mul(self, matrix: Matrix<T>) -> CscMatrix<T> {
        &self * &matrix
    }
}
/// CSC matrix multiplied by dense matrix.
impl<'a, T> Mul<&'a Matrix<T>> for CscMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn mul(self, matrix: &Matrix<T>) -> CscMatrix<T> {
        &self * matrix
    }
}
/// CSC matrix multiplied by dense matrix.
impl<'a, T> Mul<Matrix<T>> for &'a CscMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn mul(self, matrix: Matrix<T>) -> CscMatrix<T> {
        self * &matrix
    }
}
/// CSC matrix multiplied by dense matrix.
impl<'a, 'b, T> Mul<&'b Matrix<T>> for &'a CscMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn mul(self, matrix: &Matrix<T>) -> CscMatrix<T> {
        assert!(self.cols() == matrix.rows(),
                "The columns of the dense matrix must be equal the rows of the CSC matrix");

        let nnz = self.nnz();
        let mut data: Vec<T> = Vec::with_capacity(nnz);
        let mut indices: Vec<usize> = Vec::with_capacity(nnz);
        let mut ptrs: Vec<usize> = Vec::with_capacity(self.cols() + 1);

        ptrs.push(0);

        for dense_col_idx in 0..matrix.cols() {
            let mut new_col_data = vec![T::zero(); matrix.cols()];

            for (col_idx, (col_rows, col_data)) in self.iter_linear().enumerate() {
                for idx in 0..col_data.len() {
                    new_col_data[col_rows[idx]] =
                        new_col_data[col_rows[idx]] +
                        col_data[idx] * unsafe { *matrix.get_unchecked([col_idx, dense_col_idx]) };
                }
            }

            for (idx, col_data) in new_col_data.iter().enumerate() {
                if !col_data.is_zero() {
                    data.push(*col_data);
                    indices.push(idx);
                }
            }

            ptrs.push(data.len());
        }

        CscMatrix::new(self.rows(), matrix.cols(), data, indices, ptrs)
    }
}

/// CSC matrix multiplied by dense vector.
impl<T> Mul<Vector<T>> for CscMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = Vector<T>;

    fn mul(self, vector: Vector<T>) -> Vector<T> {
        &self * &vector
    }
}
/// / CSC matrix multiplied by dense vector.
impl<'a, T> Mul<&'a Vector<T>> for CscMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = Vector<T>;

    fn mul(self, vector: &Vector<T>) -> Vector<T> {
        &self * vector
    }
}
/// CSC matrix multiplied by dense vector.
impl<'a, T> Mul<Vector<T>> for &'a CscMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = Vector<T>;

    fn mul(self, vector: Vector<T>) -> Vector<T> {
        self * &vector
    }
}
/// CSC matrix multiplied by dense vector.
impl<'a, 'b, T> Mul<&'b Vector<T>> for &'a CscMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = Vector<T>;

    fn mul(self, vector: &Vector<T>) -> Vector<T> {
        assert!(self.rows() == vector.size(),
                "The size of the vector must be equal the size of the CSC matrix rows");

        let vector_data = vector.data();
        let mut data: Vec<T> = vec![T::zero(); self.cols()];

        for (col_idx, (col_rows, col_data)) in self.iter_linear().enumerate() {
            for idx in 0..col_data.len() {
                data[col_rows[idx]] = data[col_rows[idx]] + col_data[idx] * vector_data[col_idx];
            }
        }

        Vector::new(data)
    }
}


/// CSC matrix multiplied by scalar.
impl<T> Mul<T> for CscMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn mul(self, scalar: T) -> CscMatrix<T> {
        self * &scalar
    }
}
/// CSC matrix multiplied by scalar.
impl<'a, T> Mul<&'a T> for CscMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn mul(self, scalar: &T) -> CscMatrix<T> {
        &self * scalar
    }
}
/// CSC matrix multiplied by scalar.
impl<'a, T> Mul<T> for &'a CscMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn mul(self, scalar: T) -> CscMatrix<T> {
        self * &scalar
    }
}
/// CSC matrix multiplied by scalar.
impl<'a, 'b, T> Mul<&'b T> for &'a CscMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = CscMatrix<T>;

    fn mul(self, scalar: &T) -> CscMatrix<T> {
        let nnz = self.nnz();
        let mut data: Vec<T> = Vec::with_capacity(nnz);
        let mut indices: Vec<usize> = Vec::with_capacity(nnz);
        let mut ptrs: Vec<usize> = Vec::with_capacity(self.cols() + 1);

        ptrs.push(0);

        for (col_rows, col_data) in self.iter_linear() {
            for idx in 0..col_data.len() {
                let value = col_data[idx] * *scalar;

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
    fn test_csc_mul_matrix() {
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
                               vec![27, 32, 16, 45, 21, 16, 8, 35, 6, 20, 10, 10, 27, 4, 2, 45,
                                    24, 4, 2, 40],
                               vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                               vec![0, 4, 8, 12, 16, 20]);

        assert_eq!(a * b, c);
    }

    #[test]
    fn test_csc_mul_scalar() {
        let a = CscMatrix::new(5,
                               5,
                               vec![4, 2, 3, 5],
                               vec![1, 2, 0, 3],
                               vec![0, 0, 2, 2, 4, 4]);

        let b = CscMatrix::new(5,
                               5,
                               vec![8, 4, 6, 10],
                               vec![1, 2, 0, 3],
                               vec![0, 0, 2, 2, 4, 4]);

        assert_eq!(a * 2, b);
    }

    #[test]
    fn test_csc_mul_vector() {
        let a = CscMatrix::new(5,
                               5,
                               vec![4, 2, 3, 5],
                               vec![1, 2, 0, 3],
                               vec![0, 0, 2, 2, 4, 4]);

        let b = Vector::new(vec![1, 4, 2, 8, 5]);

        let c = Vector::new(vec![24, 16, 8, 40, 0]);

        assert_eq!(a * b, c);
    }
}
