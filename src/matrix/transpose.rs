//! Transposing

use super::{Matrix, MatrixSlice, MatrixSliceMut, Axes, BaseMatrix, BaseMatrixMut};
use std::cmp;
use std::ptr;
use utils;

#[cfg(target_pointer_width = "64")]
use extprim::u128::u128;

struct ReducedDivisor {
    _mul_coeff: usize,
    _shift_coeff: usize,
    y: usize,
}

impl ReducedDivisor {
    #[cfg(target_pointer_width = "32")]
    fn find_log_2(x: usize) -> usize {
        let a = 31 - x.leading_zeros() as usize;
        if !a.is_power_of_two() {
            (a + 1)
        } else {
            a
        }
    }

    #[cfg(target_pointer_width = "64")]
    fn find_log_2(x: usize) -> usize {
        let a = 63 - x.leading_zeros() as usize;
        if !a.is_power_of_two() {
            a + 1
        } else {
            a
        }
    }

    #[cfg(target_pointer_width = "32")]
    #[inline(always)]
    fn umulhi(x: usize, y: usize) -> usize {
        ((x as u64 * y as u64) >> 32) as usize
    }

    #[cfg(target_pointer_width = "64")]
    #[inline(always)]
    fn umulhi(x: usize, y: usize) -> usize {
        let a = x >> 32;
        let b = x & 0xffffffff;
        let c = y >> 32;
        let d = y & 0xffffffff;

        let lo = b.wrapping_mul(d);
        let (mid, carry) = a.wrapping_mul(d).overflowing_add(b.wrapping_mul(c));
        let mut hi = a.wrapping_mul(c);
        if carry {
            hi = hi.wrapping_add(1 << 32);
        }

        let (_, lo_carry) = lo.overflowing_add(mid << 32);

        if lo_carry {
            hi = hi.wrapping_add(1);
        }

        hi.wrapping_add(mid >> 32)

        // let hi: u64;
        // unsafe {
        //     asm!("
        //         movq $1, %rax
        //         mulq $2
        //         movq %rdx, $0
        //     "
        //     : "=r"(hi)
        //     : "r"(x), "r"(y)
        //     : "rax", "rdx");
        // }
        // hi as usize
    }

    #[cfg(target_pointer_width = "32")]
    fn find_divisor(denom: usize) -> Result<(usize, usize), &'static str> {
        if denom == 0 {
            Err("Cannot find reduced divisor for 0")
        } else if denom == 1 {
            Ok((0, 0))
        } else {
            let p = 31 + ReducedDivisor::find_log_2(denom);
            let m = ((1u64 << p) + denom as u64 - 1) / denom as u64;

            Ok((m as usize, p - 32))
        }
    }

    #[cfg(target_pointer_width = "64")]
    fn find_divisor(denom: usize) -> Result<(usize, usize), &'static str> {
        if denom == 0 {
            Err("Cannot find reduced divisor for 0")
        } else if denom == 1 {
            Ok((0, 0))
        } else {
            let p = 63 + ReducedDivisor::find_log_2(denom);

            let m = ((u128::new(1u64) << p) + u128::new((denom - 1) as u64)) /
                    u128::new(denom as u64);
            Ok((m.low64() as usize, p - 64))
        }
    }

    pub fn new(y: usize) -> Result<ReducedDivisor, &'static str> {
        let (m, s) = try!(ReducedDivisor::find_divisor(y));
        Ok(ReducedDivisor {
            _mul_coeff: m,
            _shift_coeff: s,
            y: y,
        })
    }

    #[inline(always)]
    pub fn div(&self, x: usize) -> usize {
        if self._mul_coeff == 0 {
            x
        } else {
            ReducedDivisor::umulhi(x, self._mul_coeff) >> self._shift_coeff
        }
    }

    #[inline(always)]
    pub fn modulus(&self, x: usize) -> usize {
        if self._mul_coeff == 0 {
            0
        } else {
            x - self.div(x) * self.y
        }
    }

    #[inline(always)]
    pub fn divmod(&self, x: usize) -> (usize, usize) {
        if self.y == 1 {
            (x, 0)
        } else {
            let q = self.div(x);
            (q, x - q * self.y)
        }
    }
}

fn gcd(mut u: usize, mut v: usize) -> usize {
    while v != 0 {
        let curr_v = v;
        v = u % v;;
        u = curr_v;
    }

    u
}

fn mmi(a: usize, n: usize) -> Result<usize, &'static str> {
    if n < 1 {
        Err("Inverse doesn't exist for n <= 1")
    } else {
        let mut t = 0i32;
        let mut r = n;
        let mut newt = 1i32;
        let mut newr = a;

        while newr != 0 {
            let quotient = r / newr;

            let oldt = t;
            t = newt;
            newt = oldt - (quotient as i32 * newt);

            let oldr = r;
            r = newr;
            newr = oldr - quotient * newr;
        }

        if r > 1 {
            return Err("value a is not invertible mod n");
        }

        if t < 0 {
            t += n as i32;
        }

        Ok(t as usize)
    }
}

#[inline(always)]
fn gather_rot_col(i: usize, x: usize, red_m: &ReducedDivisor) -> usize {
    red_m.modulus(i + x)
}

#[inline(always)]
fn f_helper(i: usize,
            j: usize,
            rows: usize,
            cols: usize,
            c: usize,
            red_c: &ReducedDivisor)
            -> usize {
    if (i + c) - red_c.modulus(j) <= rows {
        j + i * (cols - 1)
    } else {
        j + i * (cols - 1) + rows
    }
}

#[inline(always)]
fn d_inverse(i: usize,
             j: usize,
             b: usize,
             a_inv: usize,
             rows: usize,
             cols: usize,
             c: usize,
             red_b: &ReducedDivisor,
             red_c: &ReducedDivisor)
             -> usize {
    let f_ij = f_helper(i, j, rows, cols, c, red_c);
    let (f_ij_div, f_ij_mod) = red_c.divmod(f_ij);
    (red_b.modulus((a_inv * f_ij_div)) + f_ij_mod * b)
}

#[inline(always)]
fn gather_shuffle_col(i: usize,
                      j: usize,
                      cols: usize,
                      red_a: &ReducedDivisor,
                      red_m: &ReducedDivisor)
                      -> usize {
    red_m.modulus((j + i * cols) - red_a.div(i))
}

impl<T: Copy> Matrix<T> {
    /// Transposes the matrix in place.
    ///
    /// Computes the transpose of the matrix without allocating memory
    /// for a new matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, BaseMatrix};
    ///
    /// let mut a = Matrix::new(2, 3, vec![0,1,2,3,4,5]);
    /// a.inplace_transpose();
    ///
    /// assert_eq!(a.rows(), 3);
    /// assert_eq!(a.cols(), 2);
    ///
    /// // Print the matrix which is not transposed
    /// println!("{}", a);
    /// ```
    pub fn inplace_transpose(&mut self) {
        if self.rows == self.cols {
            self.inplace_square_transpose();
        } else {
            self.c2r_transpose();
        }
    }

    fn inplace_square_transpose(&mut self) {
        unsafe {
            for i in 0..self.rows - 1 {
                for j in i + 1..self.cols {
                    ptr::swap(self.get_unchecked_mut([i, j]) as *mut T,
                              self.get_unchecked_mut([j, i]) as *mut T);
                }
            }
        }
    }

    fn c2r_transpose(&mut self) {
        let m = self.rows;
        let n = self.cols;
        let c = gcd(m, n);

        let a = m / c;
        let b = n / c;
        let a_inv = mmi(a, b).unwrap();

        let larger = cmp::max(m, n);
        let mut tmp = Vec::with_capacity(larger);
        unsafe {
            tmp.set_len(larger);

            // Create strength reduction structs for efficient div/mod
            let red_a = ReducedDivisor::new(a).unwrap();
            let red_b = ReducedDivisor::new(b).unwrap();
            let red_c = ReducedDivisor::new(c).unwrap();
            let red_m = ReducedDivisor::new(m).unwrap();
            if c > 1 {
                for j in 0..n {
                    let x = red_b.div(j);
                    for i in 0..m {
                        *tmp.get_unchecked_mut(i) =
                            *self.get_unchecked([gather_rot_col(i, x, &red_m), j]);
                    }

                    for i in 0..m {
                        *self.get_unchecked_mut([i, j]) = *tmp.get_unchecked(i);
                    }
                }
            }

            for i in 0..m {
                for j in 0..n {
                    *tmp.get_unchecked_mut(j) =
                        *self.get_unchecked([i, d_inverse(i, j, b, a_inv, m, n, c, &red_b, &red_c)]);
                }

                // This ensures the assignment is vectorized
                utils::in_place_vec_bin_op(self.get_row_unchecked_mut(i),
                                           &tmp[..n],
                                           |x, &y| *x = y);
            }

            for j in 0..n {
                for i in 0..m {
                    *tmp.get_unchecked_mut(i) =
                        *self.get_unchecked([gather_shuffle_col(i, j, n, &red_a, &red_m), j]);
                }

                for i in 0..m {
                    *self.get_unchecked_mut([i, j]) = *tmp.get_unchecked(i);
                }
            }
        }

        self.rows = n;
        self.cols = m;
    }

    /// Transposing efficiently out of place
    pub fn out_of_place_t(&self) -> Matrix<T> {
        let mut new_data: Vec<T> = Vec::with_capacity(self.rows * self.cols);
        unsafe {
            new_data.set_len(self.rows * self.cols);
        }

        let mut b = Matrix::new(self.cols, self.rows, new_data);

        Matrix::<T>::outer_t(self.as_slice(), b.as_mut_slice());

        b

    }

    fn outer_t<'a, 'b>(a: MatrixSlice<'a, T>, mut b: MatrixSliceMut<'b, T>) {
        let rows_larger: bool;
        let larger_dim: usize;
        if a.cols() >= b.rows() {
            rows_larger = false;
            larger_dim = a.cols();
        } else {
            rows_larger = true;
            larger_dim = a.rows();
        }

        if larger_dim <= 8 {
            Matrix::<T>::base_t(a, b);
        } else {
            if rows_larger {
                let split_point = a.rows() / 2;
                let (a_1, a_2) = a.split_at(split_point, Axes::Row);
                let (b_1, b_2) = b.split_at_mut(split_point, Axes::Col);

                Matrix::<T>::outer_t(a_1, b_1);
                Matrix::<T>::outer_t(a_2, b_2);
            } else {
                let split_point = a.cols / 2;
                let (a_1, a_2) = a.split_at(split_point, Axes::Col);
                let (b_1, b_2) = b.split_at_mut(split_point, Axes::Row);

                Matrix::<T>::outer_t(a_1, b_1);
                Matrix::<T>::outer_t(a_2, b_2);
            }
        }
    }

    fn base_t<'a, 'b>(a: MatrixSlice<'a, T>, mut b: MatrixSliceMut<'b, T>) {
        unsafe {
            for i in 0..a.cols() {
                for j in 0..a.rows() {
                    *b.get_unchecked_mut([i, j]) = *a.get_unchecked([j, i]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ReducedDivisor, gcd, mmi};
    use matrix::{Matrix, BaseMatrix};

    #[test]
    fn test_reduced_arith_div() {
        let a = ReducedDivisor::new(5).unwrap();

        assert_eq!(a.div(3), 0);
        assert_eq!(a.div(6), 1);
        assert_eq!(a.div(10), 2);
    }

    #[test]
    fn test_reduced_arith_mod() {
        let a = ReducedDivisor::new(5).unwrap();

        assert_eq!(a.modulus(5), 0);
        assert_eq!(a.modulus(4), 4);
        assert_eq!(a.modulus(6), 1);
    }

    #[test]
    fn test_reduced_arith_divmod() {
        let a = ReducedDivisor::new(5).unwrap();

        assert_eq!(a.divmod(5), (1, 0));
        assert_eq!(a.divmod(4), (0, 4));
        assert_eq!(a.divmod(6), (1, 1));
    }

    #[test]
    fn test_reduced_arith_2_pow() {
        let a = ReducedDivisor::new(4).unwrap();

        assert_eq!(a.divmod(0), (0, 0));
        assert_eq!(a.divmod(1), (0, 1));
        assert_eq!(a.divmod(4), (1, 0));
        assert_eq!(a.divmod(9), (2, 1));
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(0, 0), 0);
        assert_eq!(gcd(0, 10), 10);
        assert_eq!(gcd(10, 0), 10);
        assert_eq!(gcd(9, 6), 3);
        assert_eq!(gcd(8, 45), 1);
    }

    #[test]
    fn test_mmi() {
        assert_eq!(mmi(2, 5).unwrap(), 3);
        assert!(mmi(2, 4).is_err());
        assert_eq!(mmi(3, 10).unwrap(), 7);

    }

    #[test]
    fn test_square_c2r_transpose() {
        let mut a = Matrix::new(4, 4, (0..16).collect::<Vec<_>>());
        a.c2r_transpose();

        let transposed = vec![0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];

        assert_eq!(a.into_vec(), transposed);
    }

    #[test]
    fn test_inplace_square_transpose() {
        let mut a = Matrix::new(4, 4, (0..16).collect::<Vec<_>>());
        a.inplace_square_transpose();

        let transposed = vec![0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];

        assert_eq!(a.into_vec(), transposed);
    }

    #[test]
    fn test_row_large_c2r_transpose() {
        let mut a = Matrix::new(6, 3, (0..18).collect::<Vec<_>>());
        a.c2r_transpose();

        assert_eq!(a.rows, 3);
        assert_eq!(a.cols, 6);

        let transposed = vec![0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 16, 2, 5, 8, 11, 14, 17];


        assert_eq!(a.into_vec(), transposed);
    }

    #[test]
    fn test_col_large_c2r_transpose() {
        let mut a = Matrix::new(3, 6, (0..18).collect::<Vec<_>>());
        a.c2r_transpose();

        assert_eq!(a.rows, 6);
        assert_eq!(a.cols, 3);

        let transposed = vec![0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 4, 10, 16, 5, 11, 17];


        assert_eq!(a.into_vec(), transposed);
    }

    #[test]
    fn test_out_of_place_transpose_small() {
        let a = Matrix::new(3, 6, (0..18).collect::<Vec<_>>());
        let b = a.out_of_place_t();

        assert_eq!(b.rows, 6);
        assert_eq!(b.cols, 3);

        let transposed = vec![0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 4, 10, 16, 5, 11, 17];


        assert_eq!(b.into_vec(), transposed);
    }

    #[test]
    fn test_out_of_place_transpose_large() {
        let a = Matrix::new(64, 64, (0..64 * 64).collect::<Vec<_>>());
        let b = a.out_of_place_t();
        let c = a.transpose();

        assert_eq!(b.into_vec(), c.into_vec());
    }
}