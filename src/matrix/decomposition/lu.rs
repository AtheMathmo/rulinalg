use matrix::{Matrix, BaseMatrixMut};
use error::{Error, ErrorKind};

use std::any::Any;

use libnum::Float;

use matrix::decomposition::Decomposition;

/// TODO: docs
pub struct LUP<T> {
    pub l: Matrix<T>,
    pub u: Matrix<T>,
    pub p: Matrix<T>
}

/// TODO: Docs
pub struct PartialPivLu<T> {
    // For now, we store the full matrices, but
    // we can improve this by storing the decomposition
    // in the input matrix such that L and U can be stored
    // in the space of a single matrix
    lup: LUP<T>
}

impl<T> Decomposition for PartialPivLu<T> {
    type Factors = LUP<T>;

    fn unpack(self) -> LUP<T> {
        self.lup
    }
}

impl<T: 'static + Float> PartialPivLu<T> {
    /// TODO
    fn decompose(matrix: Matrix<T>) -> Result<Self, Error> {
        matrix.lup_decomp().map(|(l, u, p)|
            PartialPivLu {
                lup: LUP {
                    l: l,
                    u: u,
                    p: p
                }
            }
        )
    }
}


impl<T> Matrix<T> where T: Any + Float
{
    /// Computes L, U, and P for LUP decomposition.
    ///
    /// Returns L,U, and P respectively.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::matrix::Matrix;
    ///
    /// let a = matrix![1.0, 2.0, 0.0;
    ///                 0.0, 3.0, 4.0;
    ///                 5.0, 1.0, 2.0];
    ///
    /// let (l, u, p) = a.lup_decomp().expect("This matrix should decompose!");
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// - Matrix is not square.
    ///
    /// # Failures
    ///
    /// - Matrix cannot be LUP decomposed.
    pub fn lup_decomp(self) -> Result<(Matrix<T>, Matrix<T>, Matrix<T>), Error> {
        let n = self.cols;
        assert!(self.rows == n, "Matrix must be square for LUP decomposition.");
        let mut l = Matrix::<T>::zeros(n, n);
        let mut u = self;
        let mut p = Matrix::<T>::identity(n);

        for index in 0..n {
            let mut curr_max_idx = index;
            let mut curr_max = u[[curr_max_idx, curr_max_idx]];

            for i in (curr_max_idx+1)..n {
                if u[[i, index]].abs() > curr_max.abs() {
                    curr_max = u[[i, index]];
                    curr_max_idx = i;
                }
            }
            if curr_max.abs() < T::epsilon() {
                return Err(Error::new(ErrorKind::DivByZero,
                    "Singular matrix found in LUP decomposition. \
                    A value in the diagonal of U == 0.0."));
            }

            if curr_max_idx != index {
                l.swap_rows(index, curr_max_idx);
                u.swap_rows(index, curr_max_idx);
                p.swap_rows(index, curr_max_idx);
            }
            l[[index, index]] = T::one();
            for i in (index+1)..n {
                let mult = u[[i, index]]/curr_max;
                l[[i, index]] = mult;
                u[[i, index]] = T::zero();
                for j in (index+1)..n {
                    u[[i, j]] = u[[i,j]] - mult*u[[index, j]];
                }
            }
        }
        Ok((l, u, p))
    }
}

#[cfg(test)]
mod tests {
    use matrix::{Matrix, BaseMatrix};
    use matrix::PermutationMatrix;
    use testsupport::{is_lower_triangular, is_upper_triangular};

    use super::{PartialPivLu, LUP};
    use matrix::decomposition::Decomposition;

    #[test]
    #[should_panic]
    fn test_non_square_lup_decomp() {
        let a: Matrix<f64> = Matrix::ones(2, 3);

        let _ = a.lup_decomp();
    }

    #[test]
    fn test_lup_decomp() {
        use error::ErrorKind;
        let a: Matrix<f64> = matrix![
            1., 2., 3., 4.;
            0., 0., 0., 0.;
            0., 0., 0., 0.;
            0., 0., 0., 0.
        ];

        match a.lup_decomp() {
            Err(e) => assert!(*e.kind() == ErrorKind::DivByZero),
            Ok(_) => panic!()
        }
    }

    #[test]
    fn lup_decompose_arbitrary() {
        // Since the LUP decomposition is not in general unique,
        // we can not test against factors directly, but
        // instead we must rely on the fact that the
        // matrices P, L and U together construct the
        // original matrix
        let x = matrix![ -3.0,   0.0,   4.0,   1.0;
                        -12.0,   5.0,  17.0,   1.0;
                         15.0,   0.0, -18.0,  -5.0;
                          6.0,  20.0, -10.0, -15.0 ];

        let LUP { l, u, p } = PartialPivLu::decompose(x.clone())
                                           .unwrap()
                                           .unpack();
        let y = p.transpose() * &l * &u;
        assert_matrix_eq!(x, y, comp = float);
        assert!(is_lower_triangular(&l));
        assert!(is_upper_triangular(&u));
    }
}
