//! Matrix Decompositions
//!
//! References:
//! 1. [On Matrix Balancing and EigenVector computation]
//! (http://arxiv.org/pdf/1401.5766v1.pdf), James, Langou and Lowery
//!
//! 2. [The QR algorithm for eigen decomposition]
//! (http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter4.pdf)
//!
//! 3. [Computation of the SVD]
//! (http://www.cs.utexas.edu/users/inderjit/public_papers/HLA_SVD.pdf)

use std::any::Any;
use std::cmp;
use std::ops::{Mul, Add, Div, Sub, Neg};
use std::slice;

use matrix::{Matrix, MatrixSlice, MatrixSliceMut, BaseMatrix, BaseMatrixMut};
use vector::Vector;
use Metric;
use utils;
use error::{Error, ErrorKind};

use libnum::{One, Zero, Float, Signed};
use libnum::{cast, abs};
use epsilon::MachineEpsilon;

impl<T> Matrix<T>
    where T: Any + Float
{
    /// Cholesky decomposition
    ///
    /// Returns the cholesky decomposition of a positive definite matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let m = Matrix::new(3,3, vec![1.0,0.5,0.5,0.5,1.0,0.5,0.5,0.5,1.0]);
    ///
    /// let l = m.cholesky();
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    ///
    /// # Failures
    ///
    /// - Matrix is not positive definite.
    pub fn cholesky(&self) -> Result<Matrix<T>, Error> {
        assert!(self.rows == self.cols,
                "Matrix must be square for Cholesky decomposition.");

        let mut new_data = Vec::<T>::with_capacity(self.rows() * self.cols());

        for i in 0..self.rows() {

            for j in 0..self.cols() {

                if j > i {
                    new_data.push(T::zero());
                    continue;
                }

                let mut sum = T::zero();
                for k in 0..j {
                    sum = sum + (new_data[i * self.cols() + k] * new_data[j * self.cols() + k]);
                }

                if j == i {
                    new_data.push((self[[i, i]] - sum).sqrt());
                } else {
                    let p = (self[[i, j]] - sum) / new_data[j * self.cols + j];

                    if !p.is_finite() {
                        return Err(Error::new(ErrorKind::DecompFailure,
                                              "Matrix is not positive definite."));
                    } else {

                    }
                    new_data.push(p);
                }
            }
        }

        Ok(Matrix {
            rows: self.rows(),
            cols: self.cols(),
            data: new_data,
        })
    }

    /// Compute the cos and sin values for the givens rotation.
    ///
    /// Returns a tuple (c, s).
    fn givens_rot(a: T, b: T) -> (T, T) {
        let r = a.hypot(b);

        (a / r, -b / r)
    }

    fn make_householder(column: &[T]) -> Result<Matrix<T>, Error> {
        let size = column.len();

        if size == 0 {
            return Err(Error::new(ErrorKind::InvalidArg,
                                  "Column for householder transform cannot be empty."));
        }

        let denom = column[0] + column[0].signum() * utils::dot(column, column).sqrt();

        if denom == T::zero() {
            return Err(Error::new(ErrorKind::DecompFailure,
                                  "Cannot produce househoulder transform from column as first \
                                   entry is 0."));
        }

        let mut v = column.into_iter().map(|&x| x / denom).collect::<Vec<T>>();
        // Ensure first element is fixed to 1.
        v[0] = T::one();
        let v = Vector::new(v);
        let v_norm_sq = v.dot(&v);

        let v_vert = Matrix::new(size, 1, v.data().clone());
        let v_hor = Matrix::new(1, size, v.into_vec());
        Ok(Matrix::<T>::identity(size) - (v_vert * v_hor) * ((T::one() + T::one()) / v_norm_sq))
    }

    fn make_householder_vec(column: &[T]) -> Result<Matrix<T>, Error> {
        let size = column.len();

        if size == 0 {
            return Err(Error::new(ErrorKind::InvalidArg,
                                  "Column for householder transform cannot be empty."));
        }

        let denom = column[0] + column[0].signum() * utils::dot(column, column).sqrt();

        if denom == T::zero() {
            return Err(Error::new(ErrorKind::DecompFailure,
                                  "Cannot produce househoulder transform from column as first \
                                   entry is 0."));
        }

        let mut v = column.into_iter().map(|&x| x / denom).collect::<Vec<T>>();
        // Ensure first element is fixed to 1.
        v[0] = T::one();
        let v = Matrix::new(size, 1, v);

        Ok(&v / v.norm())
    }

    /// Compute the QR decomposition of the matrix.
    ///
    /// Returns the tuple (Q,R).
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let m = Matrix::new(3,3, vec![1.0,0.5,0.5,0.5,1.0,0.5,0.5,0.5,1.0]);
    ///
    /// let (q, r) = m.qr_decomp().unwrap();
    /// ```
    ///
    /// # Failures
    ///
    /// - Cannot compute the QR decomposition.
    pub fn qr_decomp(self) -> Result<(Matrix<T>, Matrix<T>), Error> {
        let m = self.rows();
        let n = self.cols();

        let mut q = Matrix::<T>::identity(m);
        let mut r = self;

        for i in 0..(n - ((m == n) as usize)) {
            let holder_transform: Result<Matrix<T>, Error>;
            {
                let lower_slice = MatrixSlice::from_matrix(&r, [i, i], m - i, 1);
                holder_transform =
                    Matrix::make_householder(&lower_slice.iter().cloned().collect::<Vec<_>>());
            }

            if !holder_transform.is_ok() {
                return Err(Error::new(ErrorKind::DecompFailure,
                                      "Cannot compute QR decomposition."));
            } else {
                let mut holder_data = holder_transform.unwrap().into_vec();

                // This bit is inefficient
                // using for now as we'll swap to lapack eventually.
                let mut h_full_data = Vec::with_capacity(m * m);

                for j in 0..m {
                    let mut row_data: Vec<T>;
                    if j < i {
                        row_data = vec![T::zero(); m];
                        row_data[j] = T::one();
                        h_full_data.extend(row_data);
                    } else {
                        row_data = vec![T::zero(); i];
                        h_full_data.extend(row_data);
                        h_full_data.extend(holder_data.drain(..m - i));
                    }
                }

                let h = Matrix::new(m, m, h_full_data);

                q = q * &h;
                r = h * &r;
            }
        }

        Ok((q, r))
    }

    /// Converts matrix to bidiagonal form
    ///
    /// Returns (B, U, V), where B is bidiagonal and `self = U B V_T`.
    ///
    /// Note that if `self` has `self.rows() > self.cols()` the matrix will
    /// be transposed and then reduced - this will lead to a sub-diagonal instead
    /// of super-diagonal.
    ///
    /// # Failures
    ///
    /// - The matrix cannot be reduced to bidiagonal form.
    pub fn bidiagonal_decomp(mut self) -> Result<(Matrix<T>, Matrix<T>, Matrix<T>), Error> {
        let mut flipped = false;

        if self.rows < self.cols {
            flipped = true;
            self = self.transpose()
        }

        let m = self.rows;
        let n = self.cols;

        let mut u = Matrix::identity(m);
        let mut v = Matrix::identity(n);

        for k in 0..n {
            let h_holder: Matrix<T>;
            {
                let lower_slice = MatrixSlice::from_matrix(&self, [k, k], m - k, 1);
                h_holder = try!(Matrix::make_householder(&lower_slice.iter()
                        .cloned()
                        .collect::<Vec<_>>())
                    .map_err(|_| {
                        Error::new(ErrorKind::DecompFailure, "Cannot compute bidiagonal form.")
                    }));
            }

            {
                // Apply householder on the left to kill under diag.
                let lower_self_block = MatrixSliceMut::from_matrix(&mut self, [k, k], m - k, n - k);
                let transformed_self = &h_holder * &lower_self_block;
                lower_self_block.set_to(transformed_self.as_slice());
                let lower_u_block = MatrixSliceMut::from_matrix(&mut u, [0, k], m, m - k);
                let transformed_u = &lower_u_block * h_holder;
                lower_u_block.set_to(transformed_u.as_slice());
            }

            if k < n - 2 {
                let row: &[T];
                unsafe {
                    // Get the kth row from column k+1 to end.
                    row = slice::from_raw_parts(self.data
                                                    .as_ptr()
                                                    .offset((k * self.cols + k + 1) as isize),
                                                n - k - 1);
                }

                let row_h_holder = try!(Matrix::make_householder(row).map_err(|_| {
                    Error::new(ErrorKind::DecompFailure, "Cannot compute bidiagonal form.")
                }));

                {
                    // Apply householder on the right to kill right of super diag.
                    let lower_self_block =
                        MatrixSliceMut::from_matrix(&mut self, [k, k + 1], m - k, n - k - 1);

                    let transformed_self = &lower_self_block * &row_h_holder;
                    lower_self_block.set_to(transformed_self.as_slice());
                    let lower_v_block =
                        MatrixSliceMut::from_matrix(&mut v, [0, k + 1], n, n - k - 1);
                    let transformed_v = &lower_v_block * row_h_holder;
                    lower_v_block.set_to(transformed_v.as_slice());

                }
            }
        }

        // Trim off the zerod blocks.
        self.data.truncate(n * n);
        self.rows = n;
        u = MatrixSlice::from_matrix(&u, [0, 0], m, n).into_matrix();

        if flipped {
            Ok((self.transpose(), v, u))
        } else {
            Ok((self, u, v))
        }

    }
}

/// Ensures that all singular values in the given singular value decomposition
/// are non-negative, making necessary corrections to the singular vectors.
///
/// The SVD is represented by matrices `(b, u, v)`, where `b` is the diagonal matrix
/// containing the singular values, `u` is the matrix of left singular vectors
/// and v is the matrix of right singular vectors.
fn correct_svd_signs<T>(mut b: Matrix<T>,
                        mut u: Matrix<T>,
                        mut v: Matrix<T>)
                        -> (Matrix<T>, Matrix<T>, Matrix<T>)
    where T: Any + Float + Signed
{

    // When correcting the signs of the singular vectors, we can choose
    // to correct EITHER u or v. We make the choice depending on which matrix has the
    // least number of rows. Later we will need to multiply all elements in columns by
    // -1, which might be significantly faster in corner cases if we pick the matrix
    // with the least amount of rows.
    {
        let ref mut shortest_matrix = if u.rows() <= v.rows() { &mut u }
                                      else { &mut v };
        let column_length = shortest_matrix.rows();
        let num_singular_values = cmp::min(b.rows(), b.cols());

        for i in 0 .. num_singular_values {
            if b[[i, i]] < T::zero() {
                // Swap sign of singular value and column in u
                b[[i, i]] = b[[i, i]].abs();

                // Access the column as a slice and flip sign
                let mut column = shortest_matrix.sub_slice_mut([0, i], column_length, 1);
                column *= -T::one();
            }
        }
    }
    (b, u, v)
}

fn sort_svd<T>(mut b: Matrix<T>,
               mut u: Matrix<T>,
               mut v: Matrix<T>)
               -> (Matrix<T>, Matrix<T>, Matrix<T>)
    where T: Any + Float + Signed
{

    assert!(u.cols() == b.cols() && b.cols() == v.cols());

    // This unfortunately incurs two allocations since we have no (simple)
    // way to iterate over a matrix diagonal, only to copy it into a new Vector
    let mut indexed_sorted_values: Vec<_> = b.diag().into_vec()
        .into_iter()
        .enumerate()
        .collect();

    // Sorting a vector of indices simultaneously with the singular values
    // gives us a mapping between old and new (final) column indices.
    indexed_sorted_values.sort_by(|&(_, ref x), &(_, ref y)|
        x.partial_cmp(y).expect("All singular values should be finite, and thus sortable.")
         .reverse()
    );

    // Set the diagonal elements of the singular value matrix
    for (i, &(_, value)) in indexed_sorted_values.iter().enumerate() {
        b[[i, i]] = value;
    }

    // Assuming N columns, the simultaneous sorting of indices and singular values yields
    // a set of N (i, j) pairs which correspond to columns which must be swapped. However,
    // for any (i, j) in this set, there is also (j, i). Keeping both of these would make us
    // swap the columns back and forth, so we must remove the duplicates. We can avoid
    // any further sorting or hashsets or similar by noting that we can simply
    // remove any (i, j) for which j >= i. This also removes (i, i) pairs,
    // i.e. columns that don't need to be swapped.
    let swappable_pairs = indexed_sorted_values.into_iter()
        .enumerate()
        .map(|(new_index, (old_index, _))| (old_index, new_index))
        .filter(|&(old_index, new_index)| old_index < new_index);

    for (old_index, new_index) in swappable_pairs {
        u.swap_cols(old_index, new_index);
        v.swap_cols(old_index, new_index);
    }

    (b, u, v)
}

impl<T: Any + Float + Signed + MachineEpsilon> Matrix<T> {
    /// Singular Value Decomposition
    ///
    /// Computes the SVD using the Golub-Reinsch algorithm.
    ///
    /// Returns Σ, U, V, such that `self` = U Σ V<sup>T</sup>. Σ is a diagonal matrix whose elements
    /// correspond to the non-negative singular values of the matrix. The singular values are ordered in
    /// non-increasing order. U and V have orthonormal columns, and each column represents the
    /// left and right singular vectors for the corresponding singular value in Σ, respectively.
    ///
    /// If `self` has M rows and N columns, the dimensions of the returned matrices
    /// are as follows.
    ///
    /// If M >= N:
    ///
    /// - `Σ`: N x N
    /// - `U`: M x N
    /// - `V`: N x N
    ///
    /// If M < N:
    ///
    /// - `Σ`: M x M
    /// - `U`: M x M
    /// - `V`: N x M
    ///
    /// Note: This version of the SVD is sometimes referred to as the 'economy SVD'.
    ///
    /// # Failures
    ///
    /// This function may fail in some cases. The current decomposition whilst being
    /// efficient is fairly basic. Hopefully the algorithm can be made not to fail in the near future.
    pub fn svd(self) -> Result<(Matrix<T>, Matrix<T>, Matrix<T>), Error> {
        let (b, u, v) = try!(self.svd_unordered());
        Ok(sort_svd(b, u, v))
    }

    fn svd_unordered(self) -> Result<(Matrix<T>, Matrix<T>, Matrix<T>), Error> {
        let (b, u, v) = try!(self.svd_golub_reinsch());

        // The Golub-Reinsch implementation sometimes spits out negative singular values,
        // so we need to correct these.
        Ok(correct_svd_signs(b, u, v))
    }

    fn svd_golub_reinsch(mut self) -> Result<(Matrix<T>, Matrix<T>, Matrix<T>), Error> {
        let mut flipped = false;

        // The algorithm assumes rows > cols. If this is not the case we transpose and fix later.
        if self.cols > self.rows {
            self = self.transpose();
            flipped = true;
        }

        let eps = T::from(3.0).unwrap() * T::epsilon();
        let n = self.cols;

        // Get the bidiagonal decomposition
        let (mut b, mut u, mut v) = try!(self.bidiagonal_decomp()
            .map_err(|_| Error::new(ErrorKind::DecompFailure, "Could not compute SVD.")));

        loop {
            // Values to count the size of lower diagonal block
            let mut q = 0;
            let mut on_lower = true;

            // Values to count top block
            let mut p = 0;
            let mut on_middle = false;

            // Iterate through and hard set the super diag if converged
            for i in (0..n - 1).rev() {
                let (b_ii, b_sup_diag, diag_abs_sum): (T, T, T);
                unsafe {
                    b_ii = *b.get_unchecked([i, i]);
                    b_sup_diag = b.get_unchecked([i, i + 1]).abs();
                    diag_abs_sum = eps * (b_ii.abs() + b.get_unchecked([i + 1, i + 1]).abs());
                }
                if b_sup_diag <= diag_abs_sum {
                    // Adjust q or p to define boundaries of sup-diagonal box
                    if on_lower {
                        q += 1;
                    } else if on_middle {
                        on_middle = false;
                        p = i + 1;
                    }
                    unsafe {
                        *b.get_unchecked_mut([i, i + 1]) = T::zero();
                    }
                } else {
                    if on_lower {
                        // No longer on the lower diagonal
                        on_middle = true;
                        on_lower = false;
                    }
                }
            }

            // We have converged!
            if q == n - 1 {
                break;
            }

            // Zero off diagonals if needed.
            for i in p..n - q - 1 {
                let (b_ii, b_sup_diag): (T, T);
                unsafe {
                    b_ii = *b.get_unchecked([i, i]);
                    b_sup_diag = *b.get_unchecked([i, i + 1]);
                }

                if b_ii.abs() < eps {
                    let (c, s) = Matrix::<T>::givens_rot(b_ii, b_sup_diag);
                    let givens = Matrix::new(2, 2, vec![c, s, -s, c]);
                    let b_i = MatrixSliceMut::from_matrix(&mut b, [i, i], 1, 2);
                    let zerod_line = &b_i * givens;

                    b_i.set_to(zerod_line.as_slice());
                }
            }

            // Apply Golub-Kahan svd step
            unsafe {
                try!(Matrix::<T>::golub_kahan_svd_step(&mut b, &mut u, &mut v, p, q)
                    .map_err(|_| Error::new(ErrorKind::DecompFailure, "Could not compute SVD.")));
            }
        }

        if flipped {
            Ok((b.transpose(), v, u))
        } else {
            Ok((b, u, v))
        }

    }

    /// This function is unsafe as it makes assumptions about the dimensions
    /// of the inputs matrices and does not check them. As a result if misused
    /// this function can call `get_unchecked` on invalid indices.
    unsafe fn golub_kahan_svd_step(b: &mut Matrix<T>,
                                   u: &mut Matrix<T>,
                                   v: &mut Matrix<T>,
                                   p: usize,
                                   q: usize)
                                   -> Result<(), Error> {
        let n = b.rows();

        // C is the lower, right 2x2 square of aTa, where a is the
        // middle block of b (between p and n-q).
        //
        // Computed as xTx + yTy, where y is the bottom 2x2 block of a
        // and x are the two columns above it within a.
        let c: Matrix<T>;
        {
            let y = MatrixSlice::from_matrix(&b, [n - q - 2, n - q - 2], 2, 2).into_matrix();
            if n - q - p - 2 > 0 {
                let x = MatrixSlice::from_matrix(&b, [p, n - q - 2], n - q - p - 2, 2);
                c = x.into_matrix().transpose() * x + y.transpose() * y;
            } else {
                c = y.transpose() * y;
            }
        }

        let c_eigs = try!(c.eigenvalues());

        // Choose eigenvalue closes to c[1,1].
        let lambda: T;
        if (c_eigs[0] - *c.get_unchecked([1, 1])).abs() <
           (c_eigs[1] - *c.get_unchecked([1, 1])).abs() {
            lambda = c_eigs[0];
        } else {
            lambda = c_eigs[1];
        }

        let b_pp = *b.get_unchecked([p, p]);
        let mut alpha = (b_pp * b_pp) - lambda;
        let mut beta = b_pp * *b.get_unchecked([p, p + 1]);
        for k in p..n - q - 1 {
            // Givens rot on columns k and k + 1
            let (c, s) = Matrix::<T>::givens_rot(alpha, beta);
            let givens_mat = Matrix::new(2, 2, vec![c, s, -s, c]);

            {
                // Pick the rows from b to be zerod.
                let b_block = MatrixSliceMut::from_matrix(b,
                                                          [k.saturating_sub(1), k],
                                                          cmp::min(3, n - k.saturating_sub(1)),
                                                          2);
                let transformed = &b_block * &givens_mat;
                b_block.set_to(transformed.as_slice());

                let v_block = MatrixSliceMut::from_matrix(v, [0, k], n, 2);
                let transformed = &v_block * &givens_mat;
                v_block.set_to(transformed.as_slice());
            }

            alpha = *b.get_unchecked([k, k]);
            beta = *b.get_unchecked([k + 1, k]);

            let (c, s) = Matrix::<T>::givens_rot(alpha, beta);
            let givens_mat = Matrix::new(2, 2, vec![c, -s, s, c]);

            {
                // Pick the columns from b to be zerod.
                let b_block = MatrixSliceMut::from_matrix(b, [k, k], 2, cmp::min(3, n - k));
                let transformed = &givens_mat * &b_block;
                b_block.set_to(transformed.as_slice());

                let m = u.rows();
                let u_block = MatrixSliceMut::from_matrix(u, [0, k], m, 2);
                let transformed = &u_block * givens_mat.transpose();
                u_block.set_to(transformed.as_slice());
            }

            if k + 2 < n - q {
                alpha = *b.get_unchecked([k, k + 1]);
                beta = *b.get_unchecked([k, k + 2]);
            }
        }
        Ok(())
    }

    /// Returns H, where H is the upper hessenberg form.
    ///
    /// If the transformation matrix is also required, you should
    /// use `upper_hess_decomp`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(4,4,vec![2.,0.,1.,1.,2.,0.,1.,2.,1.,2.,0.,0.,2.,0.,1.,1.]);
    /// let h = a.upper_hessenberg();
    ///
    /// println!("{:?}", h.expect("Could not get upper Hessenberg form.").data());
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    ///
    /// # Failures
    ///
    /// - The matrix cannot be reduced to upper hessenberg form.
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
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, BaseMatrix};
    ///
    /// let a = Matrix::new(3,3,vec![1.,2.,3.,4.,5.,6.,7.,8.,9.]);
    ///
    /// // u is the transform, h is the upper hessenberg form.
    /// let (u,h) = a.clone().upper_hess_decomp().expect("This matrix should decompose!");
    ///
    /// println!("The hess : {:?}", h.data());
    /// println!("Manual hess : {:?}", (u.transpose() * a * u).data());
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    ///
    /// # Failures
    ///
    /// - The matrix cannot be reduced to upper hessenberg form.
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
        Ok((transform, try!(self.upper_hessenberg())))
    }

    fn balance_matrix(&mut self) {
        let n = self.rows();
        let radix = T::one() + T::one();

        debug_assert!(n == self.cols(),
                      "Matrix must be square to produce balance matrix.");

        let mut d = Matrix::<T>::identity(n);
        let mut converged = false;

        while !converged {
            converged = true;

            for i in 0..n {
                let mut c = self.select_cols(&[i]).norm();
                let mut r = self.select_rows(&[i]).norm();

                let s = c * c + r * r;
                let mut f = T::one();

                while c < r / radix {
                    c = c * radix;
                    r = r / radix;
                    f = f * radix;
                }

                while c >= r * radix {
                    c = c / radix;
                    r = r * radix;
                    f = f / radix;
                }

                if (c * c + r * r) < cast::<f64, T>(0.95).unwrap() * s {
                    converged = false;
                    d.data[i * (self.cols + 1)] = f * d.data[i * (self.cols + 1)];

                    for j in 0..n {
                        self.data[j * self.cols + i] = f * self.data[j * self.cols + i];
                        self.data[i * self.cols + j] = self.data[i * self.cols + j] / f;
                    }
                }
            }
        }
    }

    fn direct_2_by_2_eigenvalues(&self) -> Result<Vec<T>, Error> {
        // The characteristic polynomial of a 2x2 matrix A is
        // λ² − (a₁₁ + a₂₂)λ + (a₁₁a₂₂ − a₁₂a₂₁);
        // the quadratic formula suffices.
        let tr = self.data[0] + self.data[3];
        let det = self.data[0] * self.data[3] - self.data[1] * self.data[2];

        let two = T::one() + T::one();
        let four = two + two;

        let discr = tr * tr - four * det;

        if discr < T::zero() {
            Err(Error::new(ErrorKind::DecompFailure,
                           "Matrix has complex eigenvalues. Currently unsupported, sorry!"))
        } else {
            let discr_root = discr.sqrt();
            Ok(vec![(tr - discr_root) / two, (tr + discr_root) / two])
        }

    }

    fn francis_shift_eigenvalues(&self) -> Result<Vec<T>, Error> {
        let n = self.rows();
        debug_assert!(n > 2,
                      "Francis shift only works on matrices greater than 2x2.");
        debug_assert!(n == self.cols, "Matrix must be square for Francis shift.");

        let mut h = try!(self.clone()
            .upper_hessenberg()
            .map_err(|_| Error::new(ErrorKind::DecompFailure, "Could not compute eigenvalues.")));
        h.balance_matrix();

        // The final index of the active matrix
        let mut p = n - 1;

        let eps = cast::<f64, T>(1e-20).expect("Failed to cast value for convergence check.");

        while p > 1 {
            let q = p - 1;
            let s = h[[q, q]] + h[[p, p]];
            let t = h[[q, q]] * h[[p, p]] - h[[q, p]] * h[[p, q]];

            let mut x = h[[0, 0]] * h[[0, 0]] + h[[0, 1]] * h[[1, 0]] - h[[0, 0]] * s + t;
            let mut y = h[[1, 0]] * (h[[0, 0]] + h[[1, 1]] - s);
            let mut z = h[[1, 0]] * h[[2, 1]];

            for k in 0..p - 1 {
                let r = cmp::max(1, k) - 1;

                let householder = try!(Matrix::make_householder(&[x, y, z]).map_err(|_| {
                    Error::new(ErrorKind::DecompFailure, "Could not compute eigenvalues.")
                }));

                {
                    // Apply householder transformation to block (on the left)
                    let h_block = MatrixSliceMut::from_matrix(&mut h, [k, r], 3, n - r);
                    let transformed = &householder * &h_block;
                    h_block.set_to(transformed.as_slice());
                }

                let r = cmp::min(k + 4, p + 1);

                {
                    // Apply householder transformation to the block (on the right)
                    let h_block = MatrixSliceMut::from_matrix(&mut h, [0, k], r, 3);
                    let transformed = &h_block * householder.transpose();
                    h_block.set_to(transformed.as_slice());
                }

                x = h[[k + 1, k]];
                y = h[[k + 2, k]];

                if k < p - 2 {
                    z = h[[k + 3, k]];
                }
            }

            let (c, s) = Matrix::givens_rot(x, y);
            let givens_mat = Matrix::new(2, 2, vec![c, -s, s, c]);

            {
                // Apply Givens rotation to the block (on the left)
                let h_block = MatrixSliceMut::from_matrix(&mut h, [q, p - 2], 2, n - p + 2);
                let transformed = &givens_mat * &h_block;
                h_block.set_to(transformed.as_slice());
            }

            {
                // Apply Givens rotation to block (on the right)
                let h_block = MatrixSliceMut::from_matrix(&mut h, [0, q], p + 1, 2);
                let transformed = &h_block * givens_mat.transpose();
                h_block.set_to(transformed.as_slice());
            }

            // Check for convergence
            if abs(h[[p, q]]) < eps * (abs(h[[q, q]]) + abs(h[[p, p]])) {
                h.data[p * h.cols + q] = T::zero();
                p -= 1;
            } else if abs(h[[p - 1, q - 1]]) < eps * (abs(h[[q - 1, q - 1]]) + abs(h[[q, q]])) {
                h.data[(p - 1) * h.cols + q - 1] = T::zero();
                p -= 2;
            }
        }

        Ok(h.diag().into_vec())
    }

    /// Eigenvalues of a square matrix.
    ///
    /// Returns a Vec of eigenvalues.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(4,4, (1..17).map(|v| v as f64).collect::<Vec<f64>>());
    /// let e = a.eigenvalues().expect("We should be able to compute these eigenvalues!");
    /// println!("{:?}", e);
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    ///
    /// # Failures
    ///
    /// - Eigenvalues cannot be computed.
    pub fn eigenvalues(&self) -> Result<Vec<T>, Error> {
        let n = self.rows();
        assert!(n == self.cols,
                "Matrix must be square for eigenvalue computation.");

        match n {
            1 => Ok(vec![self.data[0]]),
            2 => self.direct_2_by_2_eigenvalues(),
            _ => self.francis_shift_eigenvalues(),
        }
    }

    fn direct_2_by_2_eigendecomp(&self) -> Result<(Vec<T>, Matrix<T>), Error> {
        let eigenvalues = try!(self.eigenvalues());
        // Thanks to
        // http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/index.html
        // for this characterization—
        if self.data[2] != T::zero() {
            let decomp_data = vec![eigenvalues[0] - self.data[3],
                                   eigenvalues[1] - self.data[3],
                                   self.data[2],
                                   self.data[2]];
            Ok((eigenvalues, Matrix::new(2, 2, decomp_data)))
        } else if self.data[1] != T::zero() {
            let decomp_data = vec![self.data[1],
                                   self.data[1],
                                   eigenvalues[0] - self.data[0],
                                   eigenvalues[1] - self.data[0]];
            Ok((eigenvalues, Matrix::new(2, 2, decomp_data)))
        } else {
            Ok((eigenvalues, Matrix::new(2, 2, vec![T::one(), T::zero(), T::zero(), T::one()])))
        }
    }

    fn francis_shift_eigendecomp(&self) -> Result<(Vec<T>, Matrix<T>), Error> {
        let n = self.rows();
        debug_assert!(n > 2,
                      "Francis shift only works on matrices greater than 2x2.");
        debug_assert!(n == self.cols, "Matrix must be square for Francis shift.");

        let (u, mut h) = try!(self.clone().upper_hess_decomp().map_err(|_| {
            Error::new(ErrorKind::DecompFailure,
                       "Could not compute eigen decomposition.")
        }));
        h.balance_matrix();
        let mut transformation = Matrix::identity(n);

        // The final index of the active matrix
        let mut p = n - 1;

        let eps = cast::<f64, T>(1e-20).expect("Failed to cast value for convergence check.");

        while p > 1 {
            let q = p - 1;
            let s = h[[q, q]] + h[[p, p]];
            let t = h[[q, q]] * h[[p, p]] - h[[q, p]] * h[[p, q]];

            let mut x = h[[0, 0]] * h[[0, 0]] + h[[0, 1]] * h[[1, 0]] - h[[0, 0]] * s + t;
            let mut y = h[[1, 0]] * (h[[0, 0]] + h[[1, 1]] - s);
            let mut z = h[[1, 0]] * h[[2, 1]];

            for k in 0..p - 1 {
                let r = cmp::max(1, k) - 1;

                let householder = try!(Matrix::make_householder(&[x, y, z]).map_err(|_| {
                    Error::new(ErrorKind::DecompFailure,
                               "Could not compute eigen decomposition.")
                }));

                {
                    // Apply householder transformation to block (on the left)
                    let h_block = MatrixSliceMut::from_matrix(&mut h, [k, r], 3, n - r);
                    let transformed = &householder * &h_block;
                    h_block.set_to(transformed.as_slice());
                }

                let r = cmp::min(k + 4, p + 1);

                {
                    // Apply householder transformation to the block (on the right)
                    let h_block = MatrixSliceMut::from_matrix(&mut h, [0, k], r, 3);
                    let transformed = &h_block * householder.transpose();
                    h_block.set_to(transformed.as_slice());
                }

                {
                    // Update the transformation matrix
                    let trans_block =
                        MatrixSliceMut::from_matrix(&mut transformation, [0, k], n, 3);
                    let transformed = &trans_block * householder.transpose();
                    trans_block.set_to(transformed.as_slice());
                }

                x = h[[k + 1, k]];
                y = h[[k + 2, k]];

                if k < p - 2 {
                    z = h[[k + 3, k]];
                }
            }

            let (c, s) = Matrix::givens_rot(x, y);
            let givens_mat = Matrix::new(2, 2, vec![c, -s, s, c]);

            {
                // Apply Givens rotation to the block (on the left)
                let h_block = MatrixSliceMut::from_matrix(&mut h, [q, p - 2], 2, n - p + 2);
                let transformed = &givens_mat * &h_block;
                h_block.set_to(transformed.as_slice());
            }

            {
                // Apply Givens rotation to block (on the right)
                let h_block = MatrixSliceMut::from_matrix(&mut h, [0, q], p + 1, 2);
                let transformed = &h_block * givens_mat.transpose();
                h_block.set_to(transformed.as_slice());
            }

            {
                // Update the transformation matrix
                let trans_block = MatrixSliceMut::from_matrix(&mut transformation, [0, q], n, 2);
                let transformed = &trans_block * givens_mat.transpose();
                trans_block.set_to(transformed.as_slice());
            }

            // Check for convergence
            if abs(h[[p, q]]) < eps * (abs(h[[q, q]]) + abs(h[[p, p]])) {
                h.data[p * h.cols + q] = T::zero();
                p -= 1;
            } else if abs(h[[p - 1, q - 1]]) < eps * (abs(h[[q - 1, q - 1]]) + abs(h[[q, q]])) {
                h.data[(p - 1) * h.cols + q - 1] = T::zero();
                p -= 2;
            }
        }

        Ok((h.diag().into_vec(), u * transformation))
    }

    /// Eigendecomposition of a square matrix.
    ///
    /// Returns a Vec of eigenvalues, and a matrix with eigenvectors as the columns.
    ///
    /// The eigenvectors are only gauranteed to be correct if the matrix is real-symmetric.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(3,3,vec![3.,2.,4.,2.,0.,2.,4.,2.,3.]);
    ///
    /// let (e, m) = a.eigendecomp().expect("We should be able to compute this eigendecomp!");
    /// println!("{:?}", e);
    /// println!("{:?}", m.data());
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    ///
    /// # Failures
    ///
    /// - The eigen decomposition can not be computed.
    pub fn eigendecomp(&self) -> Result<(Vec<T>, Matrix<T>), Error> {
        let n = self.rows();
        assert!(n == self.cols, "Matrix must be square for eigendecomp.");

        match n {
            1 => Ok((vec![self.data[0]], Matrix::new(1, 1, vec![T::one()]))),
            2 => self.direct_2_by_2_eigendecomp(),
            _ => self.francis_shift_eigendecomp(),
        }
    }
}


impl<T> Matrix<T> where T: Any + Copy + One + Zero + Neg<Output=T> +
                           Add<T, Output=T> + Mul<T, Output=T> +
                           Sub<T, Output=T> + Div<T, Output=T> +
                           PartialOrd {

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
        assert!(self.rows == n, "Matrix must be square for LUP decomposition.");

        let mut l = Matrix::<T>::zeros(n, n);
        let mut u = Matrix::<T>::zeros(n, n);

        let mt = self.transpose();

        let mut p = Matrix::<T>::identity(n);

// Compute the permutation matrix
        for i in 0..n {
            let (row,_) = utils::argmax(&mt.data[i*(n+1)..(i+1)*n]);

            if row != 0 {
                for j in 0..n {
                    p.data.swap(i*n + j, row*n+j)
                }
            }
        }

        let a_2 = &p * self;

        for i in 0..n {
            l.data[i*(n+1)] = T::one();

            for j in 0..i+1 {
                let mut s1 = T::zero();

                for k in 0..j {
                    s1 = s1 + l.data[j*n + k] * u.data[k*n + i];
                }

                u.data[j*n + i] = a_2[[j,i]] - s1;
            }

            for j in i..n {
                let mut s2 = T::zero();

                for k in 0..i {
                    s2 = s2 + l.data[j*n + k] * u.data[k*n + i];
                }

                let denom = u[[i,i]];

                if denom == T::zero() {
                    return Err(Error::new(ErrorKind::DivByZero,
                        "Singular matrix found in LUP decomposition. \
                        A value in the diagonal of U == 0.0."));
                }
                l.data[j*n + i] = (a_2[[j,i]] - s2) / denom;
            }

        }

        Ok((l,u,p))
    }
}



#[cfg(test)]
mod tests {
    use matrix::{Matrix, BaseMatrix};
    use vector::Vector;
    use super::sort_svd;

    fn validate_bidiag(mat: &Matrix<f64>,
                       b: &Matrix<f64>,
                       u: &Matrix<f64>,
                       v: &Matrix<f64>,
                       upper: bool) {
        for (idx, row) in b.iter_rows().enumerate() {
            let pair_start = if upper {
                idx
            } else {
                idx.saturating_sub(1)
            };
            assert!(!row.iter().take(pair_start).any(|&x| x > 1e-10));
            assert!(!row.iter().skip(pair_start + 2).any(|&x| x > 1e-10));
        }

        let recovered = u * b * v.transpose();

        assert_eq!(recovered.rows(), mat.rows());
        assert_eq!(recovered.cols(), mat.cols());

        assert!(!mat.data()
            .iter()
            .zip(recovered.data().iter())
            .any(|(&x, &y)| (x - y).abs() > 1e-10));
    }

    #[test]
    fn test_bidiagonal_square() {
        let mat = Matrix::new(5,
                              5,
                              vec![1f64, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 1.0, 2.0, 1.0, 3.0, 1.0,
                                   7.0, 1.0, 1.0, 4.0, 2.0, 1.0, -1.0, 3.0, 5.0, 1.0, 1.0, 3.0,
                                   2.0]);
        let (b, u, v) = mat.clone().bidiagonal_decomp().unwrap();
        validate_bidiag(&mat, &b, &u, &v, true);
    }

    #[test]
    fn test_bidiagonal_non_square() {
        let mat = Matrix::new(5,
                              3,
                              vec![1f64, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 1.0, 2.0, 1.0, 3.0, 1.0,
                                   7.0, 1.0, 1.0]);
        let (b, u, v) = mat.clone().bidiagonal_decomp().unwrap();
        validate_bidiag(&mat, &b, &u, &v, true);

        let mat = Matrix::new(3,
                              5,
                              vec![1f64, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 1.0, 2.0, 1.0, 3.0, 1.0,
                                   7.0, 1.0, 1.0]);
        let (b, u, v) = mat.clone().bidiagonal_decomp().unwrap();
        validate_bidiag(&mat, &b, &u, &v, false);
    }

    fn validate_svd(mat: &Matrix<f64>, b: &Matrix<f64>, u: &Matrix<f64>, v: &Matrix<f64>) {
        // b is diagonal (the singular values)
        for (idx, row) in b.iter_rows().enumerate() {
            assert!(!row.iter().take(idx).any(|&x| x > 1e-10));
            assert!(!row.iter().skip(idx + 1).any(|&x| x > 1e-10));
            // Assert non-negativity of diagonal elements
            assert!(row[idx] >= 0.0);
        }

        let recovered = u * b * v.transpose();

        assert_eq!(recovered.rows(), mat.rows());
        assert_eq!(recovered.cols(), mat.cols());

        assert!(!mat.data()
            .iter()
            .zip(recovered.data().iter())
            .any(|(&x, &y)| (x - y).abs() > 1e-10));

        // The transposition is due to the fact that there does not exist
        // any column iterators at the moment, and we need to simultaneously iterate
        // over the columns. Once they do exist, we should rewrite
        // the below iterators to use iter_cols() or whatever instead.
        let ref u_transposed = u.transpose();
        let ref v_transposed = v.transpose();
        let ref mat_transposed = mat.transpose();

        let mut singular_triplets = u_transposed.iter_rows().zip(b.diag().into_iter()).zip(v_transposed.iter_rows())
            // chained zipping results in nested tuple. Flatten it.
            .map(|((u_col, singular_value), v_col)| (Vector::new(u_col), singular_value, Vector::new(v_col)));

        assert!(singular_triplets.by_ref()
            // For a matrix M, each singular value σ and left and right singular vectors u and v respectively
            // satisfy M v = σ u, so we take the difference
            .map(|(ref u, sigma, ref v)| mat * v - u * sigma)
            .flat_map(|v| v.into_vec().into_iter())
            .all(|x| x.abs() < 1e-10));

        assert!(singular_triplets.by_ref()
            // For a matrix M, each singular value σ and left and right singular vectors u and v respectively
            // satisfy M_transposed u = σ v, so we take the difference
            .map(|(ref u, sigma, ref v)| mat_transposed * u - v * sigma)
            .flat_map(|v| v.into_vec().into_iter())
            .all(|x| x.abs() < 1e-10));
    }

    #[test]
    fn test_sort_svd() {
        let u = Matrix::new(2, 3, vec![1.0, 2.0, 3.0,
                                       4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 3, vec![4.0, 0.0, 0.0,
                                       0.0, 8.0, 0.0,
                                       0.0, 0.0, 2.0]);
        let v = Matrix::new(3, 3, vec![21.0, 22.0, 23.0,
                                       24.0, 25.0, 26.0,
                                       27.0, 28.0, 29.0]);
        let (b, u, v) = sort_svd(b, u, v);

        assert_eq!(b.data(), &vec![8.0, 0.0, 0.0,
                                  0.0, 4.0, 0.0,
                                  0.0, 0.0, 2.0]);
        assert_eq!(u.data(), &vec![2.0, 1.0, 3.0,
                                  5.0, 4.0, 6.0]);
        assert_eq!(v.data(), &vec![22.0, 21.0, 23.0,
                                  25.0, 24.0, 26.0,
                                  28.0, 27.0, 29.0]);

    }

    #[test]
    fn test_svd_tall_matrix() {
        // Note: This matrix is not arbitrary. It has been constructed specifically so that
        // the "natural" order of the singular values it not sorted by default.
        let mat = Matrix::new(5, 4,
                              vec![ 3.61833700244349288, -3.28382346228211697,  1.97968027781346501, -0.41869628192662156,
                                    3.96046289599926427,  0.70730060716580723, -2.80552479438772817, -1.45283286109873933,
                                    1.44435028724617442,  1.27749196276785826, -1.09858397535426366, -0.03159619816434689,
                                    1.13455445826500667,  0.81521390274755756,  3.99123446373437263, -2.83025703359666192,
                                   -3.30895752093770579, -0.04979044289857298,  3.03248594516832792,  3.85962479743330977]);
        let (b, u, v) = mat.clone().svd().unwrap();

        let expected_values = vec![8.0, 6.0, 4.0, 2.0];

        validate_svd(&mat, &b, &u, &v);

        // Assert the singular values are what we expect
        assert!(expected_values.iter()
            .zip(b.diag().data().iter())
            .all(|(expected, actual)| (expected - actual).abs() < 1e-14));
    }

    #[test]
    fn test_svd_short_matrix() {
        // Note: This matrix is not arbitrary. It has been constructed specifically so that
        // the "natural" order of the singular values it not sorted by default.
        let mat = Matrix::new(4, 5,
                              vec![ 3.61833700244349288,  3.96046289599926427,  1.44435028724617442,  1.13455445826500645, -3.30895752093770579,
                                   -3.28382346228211697,  0.70730060716580723,  1.27749196276785826,  0.81521390274755756, -0.04979044289857298,
                                    1.97968027781346545, -2.80552479438772817, -1.09858397535426366,  3.99123446373437263,  3.03248594516832792,
                                   -0.41869628192662156, -1.45283286109873933, -0.03159619816434689, -2.83025703359666192,  3.85962479743330977]);
        let (b, u, v) = mat.clone().svd().unwrap();

        let expected_values = vec![8.0, 6.0, 4.0, 2.0];

        validate_svd(&mat, &b, &u, &v);

        // Assert the singular values are what we expect
        assert!(expected_values.iter()
            .zip(b.diag().data().iter())
            .all(|(expected, actual)| (expected - actual).abs() < 1e-14));
    }

    #[test]
    fn test_svd_square_matrix() {
        let mat = Matrix::new(5, 5,
                              vec![1.0,  2.0,  3.0,  4.0,  5.0,
                                   2.0,  4.0,  1.0,  2.0,  1.0,
                                   3.0,  1.0,  7.0,  1.0,  1.0,
                                   4.0,  2.0,  1.0, -1.0,  3.0,
                                   5.0,  1.0,  1.0,  3.0,  2.0]);

        let expected_values = vec![ 12.1739747429271112,   5.2681047320525831,   4.4942269799769843,
                                     2.9279675877385123,   2.8758200827412224];

        let (b, u, v) = mat.clone().svd().unwrap();
        validate_svd(&mat, &b, &u, &v);

        // Assert the singular values are what we expect
        assert!(expected_values.iter()
            .zip(b.diag().data().iter())
            .all(|(expected, actual)| (expected - actual).abs() < 1e-12));
    }

    #[test]
    fn test_1_by_1_matrix_eigenvalues() {
        let a = Matrix::new(1, 1, vec![3.]);
        assert_eq!(vec![3.], a.eigenvalues().unwrap());
    }

    #[test]
    fn test_2_by_2_matrix_eigenvalues() {
        let a = Matrix::new(2, 2, vec![1., 2., 3., 4.]);
        // characteristic polynomial is λ² − 5λ − 2 = 0
        assert_eq!(vec![(5. - (33.0f32).sqrt()) / 2., (5. + (33.0f32).sqrt()) / 2.],
                   a.eigenvalues().unwrap());
    }

    #[test]
    fn test_2_by_2_matrix_zeros_eigenvalues() {
        let a = Matrix::new(2, 2, vec![0.; 4]);
        // characteristic polynomial is λ² = 0
        assert_eq!(vec![0.0, 0.0], a.eigenvalues().unwrap());
    }

    #[test]
    fn test_2_by_2_matrix_complex_eigenvalues() {
        // This test currently fails - complex eigenvalues would be nice though!
        let a = Matrix::new(2, 2, vec![1.0, -3.0, 1.0, 1.0]);
        // characteristic polynomial is λ² − λ + 4 = 0

        // Decomposition will fail
        assert!(a.eigenvalues().is_err());
    }

    #[test]
    fn test_2_by_2_matrix_eigendecomp() {
        let a = Matrix::new(2, 2, vec![20., 4., 20., 16.]);
        let (eigenvals, eigenvecs) = a.eigendecomp().unwrap();

        let lambda_1 = eigenvals[0];
        let lambda_2 = eigenvals[1];

        let v1 = Vector::new(vec![eigenvecs[[0, 0]], eigenvecs[[1, 0]]]);
        let v2 = Vector::new(vec![eigenvecs[[0, 1]], eigenvecs[[1, 1]]]);

        let epsilon = 0.00001;
        assert!((&a * &v1 - &v1 * lambda_1).into_vec().iter().all(|&c| c < epsilon));
        assert!((&a * &v2 - &v2 * lambda_2).into_vec().iter().all(|&c| c < epsilon));
    }

    #[test]
    fn test_3_by_3_eigenvals() {
        let a = Matrix::new(3, 3, vec![17f64, 22., 27., 22., 29., 36., 27., 36., 45.]);

        let eigs = a.eigenvalues().unwrap();

        let eig_1 = 90.4026;
        let eig_2 = 0.5973;
        let eig_3 = 0.0;

        assert!(eigs.iter().any(|x| (x - eig_1).abs() < 1e-4));
        assert!(eigs.iter().any(|x| (x - eig_2).abs() < 1e-4));
        assert!(eigs.iter().any(|x| (x - eig_3).abs() < 1e-4));
    }

    #[test]
    fn test_5_by_5_eigenvals() {
        let a = Matrix::new(5,
                            5,
                            vec![1f64, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 1.0, 2.0, 1.0, 3.0, 1.0,
                                 7.0, 1.0, 1.0, 4.0, 2.0, 1.0, -1.0, 3.0, 5.0, 1.0, 1.0, 3.0, 2.0]);

        let eigs = a.eigenvalues().unwrap();

        let eig_1 = 12.174;
        let eig_2 = 5.2681;
        let eig_3 = -4.4942;
        let eig_4 = 2.9279;
        let eig_5 = -2.8758;

        assert!(eigs.iter().any(|x| (x - eig_1).abs() < 1e-4));
        assert!(eigs.iter().any(|x| (x - eig_2).abs() < 1e-4));
        assert!(eigs.iter().any(|x| (x - eig_3).abs() < 1e-4));
        assert!(eigs.iter().any(|x| (x - eig_4).abs() < 1e-4));
        assert!(eigs.iter().any(|x| (x - eig_5).abs() < 1e-4));
    }

    #[test]
    #[should_panic]
    fn test_non_square_cholesky() {
        let a = Matrix::new(2, 3, vec![1.0; 6]);

        let _ = a.cholesky();
    }

    #[test]
    #[should_panic]
    fn test_non_square_upper_hessenberg() {
        let a = Matrix::new(2, 3, vec![1.0; 6]);

        let _ = a.upper_hessenberg();
    }

    #[test]
    #[should_panic]
    fn test_non_square_upper_hess_decomp() {
        let a = Matrix::new(2, 3, vec![1.0; 6]);

        let _ = a.upper_hess_decomp();
    }

    #[test]
    #[should_panic]
    fn test_non_square_eigenvalues() {
        let a = Matrix::new(2, 3, vec![1.0; 6]);

        let _ = a.eigenvalues();
    }

    #[test]
    #[should_panic]
    fn test_non_square_eigendecomp() {
        let a = Matrix::new(2, 3, vec![1.0; 6]);

        let _ = a.eigendecomp();
    }

    #[test]
    #[should_panic]
    fn test_non_square_lup_decomp() {
        let a = Matrix::new(2, 3, vec![1.0; 6]);

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
