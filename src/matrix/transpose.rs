//! Transposing
#![allow(dead_code)]

use super::Matrix;
use std::cmp;
use utils;

struct ReducedDivisor {
    _mul_coeff: u32,
    _shift_coeff: u32,
    y: u32,
}

impl ReducedDivisor {
    fn find_log_2(x: i32) -> u32 {
        let a = 31 - x.leading_zeros();
        if !a.is_power_of_two() {
            a + 1
        } else {
            a
        }
    }

    fn find_divisor(denom: u32) -> Result<(u32, u32), &'static str> {
        if denom == 0 {
            Err("Cannot find reduced divisor for 0")
        } else if denom == 1 {
            Ok((0, 0))
        } else {
            let p = 31 + ReducedDivisor::find_log_2(denom as i32);
            let m = ((1u64 << p) + denom as u64 - 1) / denom as u64;

            Ok((m as u32, p - 32))
        }
    }

    pub fn new(y: u32) -> Result<ReducedDivisor, &'static str> {
        let (m, s) = try!(ReducedDivisor::find_divisor(y));
        Ok(ReducedDivisor {
            _mul_coeff: m,
            _shift_coeff: s,
            y: y,
        })
    }

    #[inline(always)]
    pub fn div(&self, x: u32) -> u32 {
        if self._mul_coeff == 0 {
            x
        } else {
            ((x as u64 * self._mul_coeff as u64) >> 32) as u32 >> self._shift_coeff
        }
    }

    #[inline(always)]
    pub fn modulus(&self, x: u32) -> u32 {
        if self._mul_coeff == 0 {
            0
        } else {
            x - self.div(x) * self.y
        }
    }

    #[inline(always)]
    pub fn divmod(&self, x: u32) -> (u32, u32) {
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
fn gather_rot_col(i: usize, x: u32, red_m: &ReducedDivisor) -> usize {
    red_m.modulus(i as u32 + x) as usize
}

// #[inline(always)]
// fn gather_rot_col(i: usize, j: usize, b: usize, rows: usize) -> usize {
//     (i + j / b) % rows
// }

#[inline(always)]
fn scatter_row(i: usize, j: usize, b: usize, rows: usize, cols: usize) -> usize {
    ((i + j / b) % rows + j * rows) % cols
}

#[inline(always)]
fn f_helper(i: usize,
            j: usize,
            rows: usize,
            cols: usize,
            c: usize,
            red_c: &ReducedDivisor)
            -> usize {
    if (i + c) as u32 - red_c.modulus(j as u32) <= rows as u32 {
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
    let f_ij = f_helper(i, j, rows, cols, c, red_c) as u32;
    let (f_ij_div, f_ij_mod) = red_c.divmod(f_ij);
    (red_b.modulus((a_inv as u32 * red_b.modulus(f_ij_div))) + f_ij_mod * b as u32) as usize
}

#[inline(always)]
fn gather_shuffle_col(i: usize,
                      j: usize,
                      cols: usize,
                      red_a: &ReducedDivisor,
                      red_m: &ReducedDivisor)
                      -> usize {
    red_m.modulus((j + i * cols) as u32 - red_a.div(i as u32)) as usize
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
    /// use rulinalg::matrix::Matrix;
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
        self.c2r_transpose();
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

            let red_a = ReducedDivisor::new(a as u32).unwrap();
            let red_b = ReducedDivisor::new(b as u32).unwrap();
            let red_c = ReducedDivisor::new(c as u32).unwrap();
            let red_m = ReducedDivisor::new(m as u32).unwrap();
            if c > 1 {
                for j in 0..n {
                    let x = red_b.div(j as u32);
                    for i in 0..m {
                        // *tmp.get_unchecked_mut(i) =
                        //     *self.get_unchecked([gather_rot_col(i, j, b, m), j]);
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
                    // *tmp.get_unchecked_mut(scatter_row(i, j, b, m, n)) =
                    //     *self.get_unchecked([i, j]);
                    *tmp.get_unchecked_mut(j) =
                        *self.get_unchecked([i, d_inverse(i, j, b, a_inv, m, n, c, &red_b, &red_c)]);
                }

                utils::in_place_vec_bin_op(self.get_row_unchecked_mut(i), &tmp[..n], |x, &y| *x = y);
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
}

#[cfg(test)]
mod tests {
    use super::{ReducedDivisor, gcd, mmi};
    use matrix::Matrix;

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

        assert_eq!(a.divmod(0), (0,0));
        assert_eq!(a.divmod(1), (0,1));
        assert_eq!(a.divmod(4), (1,0));
        assert_eq!(a.divmod(9), (2,1));
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
}