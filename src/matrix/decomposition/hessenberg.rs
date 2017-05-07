use matrix::{Matrix, BaseMatrix, BaseMatrixMut, MatrixSlice, MatrixSliceMut};
use matrix::decomposition::{Decomposition, HouseholderReflection,
                            HouseholderComposition};
use matrix::decomposition::householder;
use error::{Error, ErrorKind};
use vector::Vector;

use std::any::Any;

use libnum::{Zero, Float};

/// Result of unpacking a
/// [Hessenberg decomposition](struct.HessenbergDecomposition.html).
#[derive(Debug, Clone)]
pub struct QH<T> {
    /// The orthogonal factor `Q` in the decomposition.
    pub q: Matrix<T>,
    /// The upper Hessenberg factor `H` in the decomposition.
    pub h: Matrix<T>
}

/// Upper Hessenberg decomposition of real square matrices.
///
/// Given any real square `m x m` matrix `A`, there exists an orthogonal
/// `m x m` matrix `Q` and an `m x m` *upper* Hessenberg matrix `H` such that
///
/// ```text
/// A = Q H Qᵀ.                                         (1)
/// ```
///
/// An Upper Hessenberg matrix `H` is characterized by the property that
/// `H[i, j] = 0` for any `i > j + 1`. For example, for a 4x4 matrix `H`, this
/// means that `H` has a the following pattern of zeros and (possibly)
/// non-zeros `x`:
///
/// ```text
/// [ x x x x ]
/// [ x x x x ]
/// [ 0 x x x ]
/// [ 0 0 x x ]
/// ```
///
/// Hessenberg matrices are of particular interest because their eigenvalues
/// and eigenvectors can be computed much more efficiently than for a general
/// non-symmetric matrix, and because the Hessenberg matrix `H` corresponding
/// to any matrix `A` is unitarily similar to `A`, the eigenvalues of `H`
/// coincide with the eigenvalues of `A`.
///
/// As in the case of [HouseholderQr](struct.HouseholderQr.html),
/// the `Q` factor is internally stored implicitly in terms of a sequence of
/// Householder transformations. The implicit `Q` matrix can be accessed by calling
/// the `.q()` method on the `HessenbergDecomposition` instance.
///
/// # Examples
///
/// ```rust
/// # #[macro_use] extern crate rulinalg; fn main() {
/// use rulinalg::matrix::decomposition::{Decomposition, HessenbergDecomposition, QH};
///
/// let x = matrix![5.0, 2.0, 0.0, -2.0;
///                 3.0, 3.0, 4.0,  3.0;
///                -3.0, 1.0, 2.0,  1.0;
///                 2.0, 3.0, 2.0, -3.0];
///
/// // We only clone `x` here so that we can compare for equality later
/// let QH { q, h } = HessenbergDecomposition::decompose(x.clone()).unpack();
///
/// assert_matrix_eq!(x, &q * h * q.transpose(), comp = float, ulp = 100);
/// # }
/// ```
///
/// # Internal storage format
///
/// The internal storage format is very similar to that of
/// [HouseholderQr](struct.HouseholderQr.html). Each Householder reflector
/// is stored compactly in the elements of the matrix which correspond
/// to zeros in the Heissenberg factor `H`. In addition, a vector of length
/// `n - 1` is allocated to hold the multipliers for the Householder vectors.
#[derive(Debug, Clone)]
pub struct HessenbergDecomposition<T> {
    qh: Matrix<T>,
    tau: Vec<T>
}

impl<T> HessenbergDecomposition<T> where T: Float {
    /// Computes the upper Hessenberg decomposition
    /// for a given square matrix.
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    pub fn decompose(matrix: Matrix<T>) -> HessenbergDecomposition<T> {
        assert!(matrix.rows() == matrix.cols(), "Matrix must be square for Hessenberg decomposition.");
        let n = matrix.rows();

        // n - 1 is the number of elements along the sub-1 diagonal
        // (which characterizes the Hessenberg matrix)
        let n1 = n.saturating_sub(1);

        let mut qh = matrix;
        let mut tau = vec![T::zero(); n1];

        let mut buffer = vec![T::zero(); n1];
        let mut multiply_buffer = vec![T::zero(); n];

        for j in 0 .. n1 {
            buffer.truncate(n1 - j);

            // Compute the Householder matrix Q_j and left-apply it to the
            // bottom right corner (we don't need to consider the columns
            // before j, since they are implicitly zero)
            let house = {
                let mut bottom_right = qh.sub_slice_mut([j + 1, j], n1 - j, n - j);
                bottom_right.col(0).clone_into_slice(&mut buffer);

                let house = HouseholderReflection::compute(Vector::new(buffer));
                house.buffered_left_multiply_into(&mut bottom_right, &mut multiply_buffer);
                house.store_in_col(&mut bottom_right.col_mut(0));
                house
            };

            // Apply Q_j to the trailing columns
            let mut trailing_cols = qh.sub_slice_mut([0, j + 1], n, n1 - j);
            house.buffered_right_multiply_into(&mut trailing_cols, &mut multiply_buffer);

            tau[j] = house.tau();
            buffer = house.into_vector().into_vec();

        }

        HessenbergDecomposition {
            qh: qh,
            tau: tau
        }
    }

    /// Returns the orthogonal factor Q as an instance
    /// of a [HouseholderComposition](struct.HouseholderComposition.html)
    /// operator.
    pub fn q(&self) -> HouseholderComposition<T> {
        householder::create_composition(&self.qh, &self.tau, 1)
    }
}

impl<T> Decomposition for HessenbergDecomposition<T> where T: Float {
    type Factors = QH<T>;

    fn unpack(self) -> Self::Factors {
        let q = self.q().first_k_columns(self.qh.rows());
        let mut h = self.qh;
        nullify_lower_portion(&mut h);

        QH {
            q: q,
            h: h
        }
    }
}

/// Makes the portion of the matrix which is not part of the
/// Upper Hessenberg part identically zero.
fn nullify_lower_portion<T>(matrix: &mut Matrix<T>) where T: Zero {
    for (i, mut row) in matrix.row_iter_mut().enumerate().skip(1) {
        for element in row.raw_slice_mut().iter_mut().take(i - 1) {
            *element = T::zero();
        }
    }
}

impl<T: Any + Float> Matrix<T> {
    /// Returns H, where H is the upper hessenberg form.
    ///
    /// If the transformation matrix is also required, you should
    /// use `upper_hess_decomp`.
    ///
    /// Note: This function is deprecated and will be removed in a future
    /// release. See
    /// [HessenbergDecomposition](decomposition/struct.HessenbergDecomposition.html)
    /// for its replacement.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::matrix::Matrix;
    ///
    /// let a = matrix![2., 0., 1., 1.;
    ///                 2., 0., 1., 2.;
    ///                 1., 2., 0., 0.;
    ///                 2., 0., 1., 1.];
    /// let h = a.upper_hessenberg();
    ///
    /// println!("{:}", h.expect("Could not get upper Hessenberg form."));
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    ///
    /// # Failures
    ///
    /// - The matrix cannot be reduced to upper hessenberg form.
    #[deprecated]
    pub fn upper_hessenberg(mut self) -> Result<Matrix<T>, Error> {
        let n = self.rows;
        assert!(n == self.cols,
                "Matrix must be square to produce upper hessenberg.");

        for i in 0..n - 2 {
            let h_holder_vec: Matrix<T>;
            {
                let lower_slice = MatrixSlice::from_matrix(&self, [i + 1, i], n - i - 1, 1);
                // Try to get the house holder transform - else map error and pass up.
                h_holder_vec = try!(Matrix::make_householder_vec(&lower_slice.iter()
                        .cloned()
                        .collect::<Vec<_>>())
                    .map_err(|_| {
                        Error::new(ErrorKind::DecompFailure,
                                   "Cannot compute upper Hessenberg form.")
                    }));
            }

            {
                // Apply holder on the left
                let mut block =
                    MatrixSliceMut::from_matrix(&mut self, [i + 1, i], n - i - 1, n - i);
                block -= &h_holder_vec * (h_holder_vec.transpose() * &block) *
                         (T::one() + T::one());
            }

            {
                // Apply holder on the right
                let mut block = MatrixSliceMut::from_matrix(&mut self, [0, i + 1], n, n - i - 1);
                block -= (&block * &h_holder_vec) * h_holder_vec.transpose() *
                         (T::one() + T::one());
            }

        }

        // Enforce upper hessenberg
        for i in 0..self.cols - 2 {
            for j in i + 2..self.rows {
                unsafe {
                    *self.get_unchecked_mut([j, i]) = T::zero();
                }
            }
        }

        Ok(self)
    }

    /// Returns (U,H), where H is the upper hessenberg form
    /// and U is the unitary transform matrix.
    ///
    /// Note: The current transform matrix seems broken...
    ///
    /// Note: This function is deprecated and will be removed in a future
    /// release. See
    /// [HessenbergDecomposition](decomposition/struct.HessenbergDecomposition.html)
    /// for its replacement.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::matrix::BaseMatrix;
    ///
    /// let a = matrix![1., 2., 3.;
    ///                 4., 5., 6.;
    ///                 7., 8., 9.];
    ///
    /// // u is the transform, h is the upper hessenberg form.
    /// let (u, h) = a.clone().upper_hess_decomp().expect("This matrix should decompose!");
    ///
    /// assert_matrix_eq!(h, u.transpose() * a * u, comp = abs, tol = 1e-12);
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    ///
    /// # Failures
    ///
    /// - The matrix cannot be reduced to upper hessenberg form.
    #[deprecated]
    pub fn upper_hess_decomp(self) -> Result<(Matrix<T>, Matrix<T>), Error> {
        let n = self.rows;
        assert!(n == self.cols,
                "Matrix must be square to produce upper hessenberg.");

        // First we form the transformation.
        let mut transform = Matrix::identity(n);

        for i in (0..n - 2).rev() {
            let h_holder_vec: Matrix<T>;
            {
                let lower_slice = MatrixSlice::from_matrix(&self, [i + 1, i], n - i - 1, 1);
                h_holder_vec = try!(Matrix::make_householder_vec(&lower_slice.iter()
                        .cloned()
                        .collect::<Vec<_>>())
                    .map_err(|_| {
                        Error::new(ErrorKind::DecompFailure, "Could not compute eigenvalues.")
                    }));
            }

            let mut trans_block =
                MatrixSliceMut::from_matrix(&mut transform, [i + 1, i + 1], n - i - 1, n - i - 1);
            trans_block -= &h_holder_vec * (h_holder_vec.transpose() * &trans_block) *
                           (T::one() + T::one());
        }

        // Now we reduce to upper hessenberg
        #[allow(deprecated)]
        Ok((transform, try!(self.upper_hessenberg())))
    }
}

#[cfg(test)]
mod tests {
    use super::QH;
    use super::HessenbergDecomposition;

    use matrix::{BaseMatrix, Matrix};
    use matrix::decomposition::Decomposition;
    use testsupport::is_upper_hessenberg;

    #[test]
    #[should_panic]
    #[allow(deprecated)]
    fn test_non_square_upper_hessenberg() {
        let a: Matrix<f64> = Matrix::ones(2, 3);

        let _ = a.upper_hessenberg();
    }

    #[test]
    #[should_panic]
    #[allow(deprecated)]
    fn test_non_square_upper_hess_decomp() {
        let a: Matrix<f64> = Matrix::ones(2, 3);

        let _ = a.upper_hess_decomp();
    }

    fn verify_hessenberg(x: Matrix<f64>) {
        let QH {ref h, ref q } = HessenbergDecomposition::decompose(x.clone()).unpack();

        assert!(is_upper_hessenberg(h));

        let identity = Matrix::identity(h.rows());

        // Orthogonality. For simplicity, we use a fixed absolute tolerance
        // and expect elements to be in the vicinity of 1.0
        assert_matrix_eq!(q.transpose() * q, identity, comp = abs, tol = 1e-14);
        assert_matrix_eq!(q * q.transpose(), identity, comp = abs, tol = 1e-14);

        // Reconstruction
        assert_matrix_eq!(q * h * q.transpose(), x, comp = abs, tol = 1e-14);
    }

    #[test]
    fn hessenberg_decomposition() {
        {
            let x: Matrix<f64> = matrix![];
            verify_hessenberg(x);
        }

        {
            let x = matrix![3.0];
            verify_hessenberg(x);
        }

        {
            let x = matrix![3.0,  2.0;
                            5.0,  1.0];
            verify_hessenberg(x);
        }

        {
            let x = matrix![3.0,  2.0, -5.0;
                            5.0,  1.0,  2.0;
                            3.0, -2.0,  0.0];
            verify_hessenberg(x);
        }

        {
            let x = matrix![3.0,  2.0, -5.0,  2.0;
                            5.0,  1.0,  2.0, -2.0;
                            3.0, -2.0,  0.0,  0.0;
                            4.0,  3.0, -1.0,  3.0];
            verify_hessenberg(x);
        }

        {
            let x = matrix![1.0,  3.0, -5.0,  2.0, 2.0;
                            0.0,  0.0,  5.0, -1.0, 4.0;
                            3.0, -4.0,  0.0,  0.0, 5.0;
                            0.0,  3.0, -1.0,  3.0, 1.0;
                            2.0, -4.0,  3.0,  2.0, 3.0];
            verify_hessenberg(x);
        }
    }

}
