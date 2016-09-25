use super::CscMatrix;

use std::ops::Mul;

/// Multiplies CSC matrix by scalar.
impl<T: Copy + Mul<T, Output = T>> Mul<T> for CscMatrix<T> {
    type Output = CscMatrix<T>;

    fn mul(self, scalar: T) -> CscMatrix<T> {
        self * &scalar
    }
}

/// Multiplies CSC matrix by scalar.
impl<'a, T: Copy + Mul<T, Output = T>> Mul<&'a T> for CscMatrix<T> {
    type Output = CscMatrix<T>;

    fn mul(mut self, scalar: &T) -> CscMatrix<T> {
        for value in &mut self.values {
            *value = (*value) * (*scalar);
        }

        self
    }
}
