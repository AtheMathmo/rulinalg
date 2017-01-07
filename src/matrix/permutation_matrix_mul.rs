use matrix::{PermutationMatrix, Matrix,
             MatrixSlice, MatrixSliceMut,
             BaseMatrix, BaseMatrixMut};
use vector::Vector;
use utils::Permutation;

use libnum::Zero;

use std::ops::Mul;

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
    use matrix::{BaseMatrix, BaseMatrixMut};
    use matrix::PermutationMatrix;

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