//! CSR matrix multiplication module.
//!
//! # Examples
//!
//! ```
//! use rulinalg::matrix::Matrix;
//! use rulinalg::sparse_matrix::{CompressedMatrix, CsrMatrix};
//! use rulinalg::vector::Vector;
//!
//! let a = CsrMatrix::new(5, 4, vec![1, 2, 3], vec![0, 0, 0], vec![0, 1, 2, 3, 3, 3]);
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
use sparse_matrix::{CompressedMatrix, CsrMatrix, SparseMatrix};

/// csr matrix multiplied by dense matrix.
impl<T> Mul<Matrix<T>> for CsrMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = CsrMatrix<T>;

    fn mul(self, matrix: Matrix<T>) -> CsrMatrix<T> {
        &self * &matrix
    }
}
/// csr matrix multiplied by dense matrix.
impl<'a, T> Mul<&'a Matrix<T>> for CsrMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = CsrMatrix<T>;

    fn mul(self, matrix: &Matrix<T>) -> CsrMatrix<T> {
        &self * matrix
    }
}
/// csr matrix multiplied by dense matrix.
impl<'a, T> Mul<Matrix<T>> for &'a CsrMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = CsrMatrix<T>;

    fn mul(self, matrix: Matrix<T>) -> CsrMatrix<T> {
        self * &matrix
    }
}
/// csr matrix multiplied by dense matrix.
impl<'a, 'b, T> Mul<&'b Matrix<T>> for &'a CsrMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = CsrMatrix<T>;

    fn mul(self, matrix: &Matrix<T>) -> CsrMatrix<T> {
        assert!(self.cols() == matrix.rows(),
                "The columns of the dense matrix must be equal the rows of the csr matrix");

        let nnz = self.nnz();
        let mut data: Vec<T> = Vec::with_capacity(nnz);
        let mut indices: Vec<usize> = Vec::with_capacity(nnz);
        let mut ptrs: Vec<usize> = Vec::with_capacity(self.cols() + 1);

        ptrs.push(0);

        for (row_cols, row_data) in self.iter_linear() {
            for dense_col_idx in 0..matrix.cols() {
                let mut value = T::zero();

                for idx in 0..row_data.len() {
                    value = value +
                            row_data[idx] *
                            unsafe { *matrix.get_unchecked([row_cols[idx], dense_col_idx]) };
                }

                if !value.is_zero() {
                    data.push(value);
                    indices.push(dense_col_idx);
                }
            }

            ptrs.push(data.len());
        }

        CsrMatrix::new(self.rows(), matrix.cols(), data, indices, ptrs)
    }
}

/// csr matrix multiplied by dense vector.
impl<T> Mul<Vector<T>> for CsrMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = Vector<T>;

    fn mul(self, vector: Vector<T>) -> Vector<T> {
        &self * &vector
    }
}
/// / csr matrix multiplied by dense vector.
impl<'a, T> Mul<&'a Vector<T>> for CsrMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = Vector<T>;

    fn mul(self, vector: &Vector<T>) -> Vector<T> {
        &self * vector
    }
}
/// csr matrix multiplied by dense vector.
impl<'a, T> Mul<Vector<T>> for &'a CsrMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = Vector<T>;

    fn mul(self, vector: Vector<T>) -> Vector<T> {
        self * &vector
    }
}
/// csr matrix multiplied by dense vector.
impl<'a, 'b, T> Mul<&'b Vector<T>> for &'a CsrMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = Vector<T>;

    fn mul(self, vector: &Vector<T>) -> Vector<T> {
        assert!(self.rows() == vector.size(),
                "The size of the vector must be equal the size of the csr matrix rows");

        let vector_data = vector.data();
        let mut data: Vec<T> = Vec::with_capacity(self.cols());

        for (row_cols, row_data) in self.iter_linear() {
            let mut value = T::zero();

            for idx in 0..row_data.len() {
                value = value + row_data[idx] * vector_data[row_cols[idx]];
            }

            data.push(value);
        }

        Vector::new(data)
    }
}


/// csr matrix multiplied by scalar.
impl<T> Mul<T> for CsrMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = CsrMatrix<T>;

    fn mul(self, scalar: T) -> CsrMatrix<T> {
        self * &scalar
    }
}
/// csr matrix multiplied by scalar.
impl<'a, T> Mul<&'a T> for CsrMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = CsrMatrix<T>;

    fn mul(self, scalar: &T) -> CsrMatrix<T> {
        &self * scalar
    }
}
/// csr matrix multiplied by scalar.
impl<'a, T> Mul<T> for &'a CsrMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = CsrMatrix<T>;

    fn mul(self, scalar: T) -> CsrMatrix<T> {
        self * &scalar
    }
}
/// csr matrix multiplied by scalar.
impl<'a, 'b, T> Mul<&'b T> for &'a CsrMatrix<T>
    where T: Copy + Mul<T, Output = T> + One + Zero
{
    type Output = CsrMatrix<T>;

    fn mul(self, scalar: &T) -> CsrMatrix<T> {
        let nnz = self.nnz();
        let mut data: Vec<T> = Vec::with_capacity(nnz);
        let mut indices: Vec<usize> = Vec::with_capacity(nnz);
        let mut ptrs: Vec<usize> = Vec::with_capacity(self.cols() + 1);

        ptrs.push(0);

        for (row_cols, row_data) in self.iter_linear() {
            for idx in 0..row_data.len() {
                let value = row_data[idx] * *scalar;

                // Check if value is zero truncated
                if !value.is_zero() {
                    data.push(value);
                    indices.push(row_cols[idx]);
                }
            }

            ptrs.push(data.len());
        }

        CsrMatrix::new(self.rows(), self.cols(), data, indices, ptrs)
    }
}

#[cfg(test)]
mod tests {
    use matrix::Matrix;
    use sparse_matrix::{CompressedMatrix, CsrMatrix};
    use vector::Vector;

    #[test]
    fn test_csr_mul_matrix() {
        let a = CsrMatrix::new(5,
                               5,
                               vec![4, 2, 3, 5],
                               vec![1, 2, 0, 3],
                               vec![0, 0, 2, 2, 4, 4]);

        let b = Matrix::new(5,
                            5,
                            vec![6, 5, 7, 9, 2, 8, 4, 5, 1, 1, 8, 9, 5, 4, 6, 9, 7, 2, 9, 8, 1,
                                 3, 5, 7, 9]);

        let c = CsrMatrix::new(5,
                               5,
                               vec![48, 34, 30, 12, 16, 63, 50, 31, 72, 46],
                               vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                               vec![0, 0, 5, 5, 10, 10]);

        assert_eq!(a * b, c);
    }

    #[test]
    fn test_csr_mul_scalar() {
        let a = CsrMatrix::new(5,
                               5,
                               vec![4, 2, 3, 5],
                               vec![1, 2, 0, 3],
                               vec![0, 0, 2, 2, 4, 4]);

        let b = CsrMatrix::new(5,
                               5,
                               vec![8, 4, 6, 10],
                               vec![1, 2, 0, 3],
                               vec![0, 0, 2, 2, 4, 4]);

        assert_eq!(a * 2, b);
    }

    #[test]
    fn test_csr_mul_vector() {
        let a = CsrMatrix::new(5,
                               5,
                               vec![4, 2, 3, 5],
                               vec![1, 2, 0, 3],
                               vec![0, 0, 2, 2, 4, 4]);

        let b = Vector::new(vec![1, 4, 2, 8, 5]);

        let c = Vector::new(vec![0, 20, 0, 43, 0]);

        assert_eq!(a * b, c);
    }
}
