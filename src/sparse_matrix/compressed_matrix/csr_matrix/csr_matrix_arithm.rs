use super::CsrMatrix;

use std::ops::Mul;

/// Multiplies CSR matrix by scalar.
impl<T: Copy + Mul<T, Output = T>> Mul<T> for CsrMatrix<T> {
    type Output = CsrMatrix<T>;

    fn mul(self, scalar: T) -> CsrMatrix<T> {
        self * &scalar
    }
}

/// Multiplies CSR matrix by scalar.
impl<'a, T: Copy + Mul<T, Output = T>> Mul<&'a T> for CsrMatrix<T> {
    type Output = CsrMatrix<T>;

    fn mul(mut self, scalar: &T) -> CsrMatrix<T> {
        for value in &mut self.values {
            *value = (*value) * (*scalar);
        }

        self
    }
}
