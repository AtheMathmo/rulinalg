use matrix::{Matrix, BaseMatrix, BaseMatrixMut};
use error::{Error, ErrorKind};

use std::any::Any;

use libnum::Float;

fn lup_decomp<T>(size:usize, l:&mut Matrix<T>, u:&mut Matrix<T>, p:&mut Matrix<T>) -> Result<(), Error> where T: Any + Float {
    let n = l.rows();
    if size == 0 {
        Ok(())
    } else {
        let mut curr_max_idx = n-size;
        let mut curr_max = u[[curr_max_idx, curr_max_idx]];

        for i in curr_max_idx..n {
            if u[[i, n-size]].abs() > curr_max.abs() {
                curr_max = u[[i, n-size]];
                curr_max_idx = i;
            }
        }
        if curr_max.abs() < T::epsilon() {
            return Err(Error::new(ErrorKind::DivByZero,
                "Singular matrix found in LUP decomposition. \
                A value in the diagonal of U == 0.0."));
        }

        if curr_max_idx != n-size {
            l.swap_rows(n-size, curr_max_idx);
            u.swap_rows(n-size, curr_max_idx);
            p.swap_rows(n-size, curr_max_idx);
        }
        l[[n-size, n-size]] = T::one();
        for i in (n-size+1)..n {
            let mult = u[[i, n-size]]/curr_max;
            l[[i, n-size]] = mult;
            u[[i, n-size]] = T::zero();
            for j in (n-size+1)..n {
                u[[i, j]] = u[[i,j]] - mult*u[[n-size, j]];
            }
        }
        lup_decomp(size-1, l, u, p)
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
    /// use rulinalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(3,3, vec![1.0,2.0,0.0,
    ///                               0.0,3.0,4.0,
    ///                               5.0, 1.0, 2.0]);
    ///
    /// let (l,u,p) = a.lup_decomp().expect("This matrix should decompose!");
    /// ```
    ///
    /// # Panics
    ///
    /// - Matrix is not square.
    ///
    /// # Failures
    ///
    /// - Matrix cannot be LUP decomposed.
    pub fn lup_decomp(&self) -> Result<(Matrix<T>, Matrix<T>, Matrix<T>), Error> {
        let n = self.cols;
        let mut l = Matrix::<T>::zeros(n, n);
        let mut u = self.clone();
        let mut p = Matrix::<T>::identity(n);
        match lup_decomp(n, &mut l, &mut u, &mut p) {
            Ok(()) => Ok((l, u, p)),
            Err(e) => Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use matrix::Matrix;

    #[test]
    #[should_panic]
    fn test_non_square_lup_decomp() {
        let a: Matrix<f64> = Matrix::ones(2, 3);

        let _ = a.lup_decomp();
    }

    #[test]
    fn test_lup_decomp() {
        use error::ErrorKind;
        let a: Matrix<f64> = matrix!(
            1., 2., 3., 4.;
            0., 0., 0., 0.;
            0., 0., 0., 0.;
            0., 0., 0., 0.
        );

        match a.lup_decomp() {
            Err(e) => assert!(*e.kind() == ErrorKind::DivByZero),
            Ok(_) => panic!()
        }
    }
}
