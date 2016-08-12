//! Transposing

use super::Matrix;
use std::cmp;

fn gcd(mut u: usize, mut v: usize) -> usize {
    while v != 0 {
        let curr_v = v;
        v = u % v;;
        u = curr_v;
    }

    u
}

#[inline(always)]
fn gather_rot_col(i: usize, j: usize, b: usize, rows: usize) -> usize {
    (i + j / b) % rows
}

#[inline(always)]
fn scatter_row(i: usize, j: usize, b: usize, rows: usize, cols: usize) -> usize {
    ((i + j / b) % rows + j * rows) % cols
}

#[inline(always)]
fn gather_shuffle_col(i: usize, j: usize, a: usize, rows: usize, cols: usize) -> usize {
    (j + i * cols - i / a) % rows
}

impl<T: Copy> Matrix<T> {
    fn c2r_transpose(&mut self) {
        let m = self.rows;
        let n = self.cols;
        let c = gcd(m, n);

        let a = m / c;
        let b = n / c;

        let larger = cmp::max(m, n);
        let mut tmp = Vec::with_capacity(larger);
        unsafe { tmp.set_len(larger) };
        if c > 1 {
            for j in 0..n {
                for i in 0..m {
                    tmp[i] = self[[gather_rot_col(i, j, b, m), j]];
                }

                for i in 0..m {
                    self[[i, j]] = tmp[i];
                }
            }
        }

        for i in 0..m {
            for j in 0..n {
                tmp[scatter_row(i, j, b, m, n)] = self[[i, j]];
            }

            for j in 0..n {
                self[[i, j]] = tmp[j];
            }
        }

        for j in 0..n {
            for i in 0..m {
                tmp[i] = self[[gather_shuffle_col(i, j, a, m, n), j]];
            }

            for i in 0..m {
                self[[i, j]] = tmp[i];
            }
        }

        self.rows = n;
        self.cols = m;
    }
}

#[cfg(test)]
mod tests {
    use super::gcd;
    use matrix::Matrix;

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(0, 0), 0);
        assert_eq!(gcd(0, 10), 10);
        assert_eq!(gcd(10, 0), 10);
        assert_eq!(gcd(9, 6), 3);
        assert_eq!(gcd(8, 45), 1);
    }

    #[test]
    fn test_transpose() {
        let mut a = Matrix::new(4, 4, (0..16).collect::<Vec<_>>());
        a.c2r_transpose();

        let transposed = vec![0, 4, 8, 12,
                              1, 5, 9, 13,
                              2, 6, 10, 14,
                              3, 7, 11, 15];

        assert_eq!(a.into_vec(), transposed);
    }
}