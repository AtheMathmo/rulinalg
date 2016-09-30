use std::iter::{ExactSizeIterator, FromIterator};
use std::slice;

use super::{Matrix, MatrixSlice, MatrixSliceMut, Rows, RowsMut, Diagonal, DiagonalMut};
use super::slice::{BaseMatrix, BaseMatrixMut, SliceIter, SliceIterMut};


macro_rules! impl_iter_diag (
    ($diag:ident, $diag_type:ty, $get_unchecked:ident) => (

/// Iterates over the rows in the matrix.
impl<'a, T> Iterator for $diag<'a, T> {
    type Item = $diag_type;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|i| unsafe { self.matrix.$get_unchecked([i, i]) })
    }

    fn last(mut self) -> Option<Self::Item> {
        let n = self.inner.len() - 1;
        self.nth(n)
    }
    
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.inner.nth(n).map(|i| unsafe { self.matrix.$get_unchecked([i, i])})
    }

    fn count(self) -> usize {
        self.inner.count()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, T> ExactSizeIterator for $diag<'a, T> {}

    );

);

impl_iter_diag!(Diagonal, &'a T, get_unchecked);
impl_iter_diag!(DiagonalMut, &'a mut T, get_unchecked_mut);

macro_rules! impl_iter_rows (
    ($rows:ident, $row_type:ty, $slice_from_parts:ident) => (

/// Iterates over the rows in the matrix.
impl<'a, T> Iterator for $rows<'a, T> {
    type Item = $row_type;

    fn next(&mut self) -> Option<Self::Item> {
// Check if we have reached the end
        if self.row_pos < self.slice_rows {
            let row: $row_type;
            unsafe {
// Get pointer and create a slice from raw parts
                let ptr = self.slice_start.offset(self.row_pos as isize * self.row_stride);
                row = slice::$slice_from_parts(ptr, self.slice_cols);
            }

            self.row_pos += 1;
            Some(row)
        } else {
            None
        }
    }

    fn last(self) -> Option<Self::Item> {
// Check if already at the end
        if self.row_pos < self.slice_rows {
            unsafe {
// Get pointer to last row and create a slice from raw parts
                let ptr = self.slice_start.offset((self.slice_rows - 1) as isize * self.row_stride);
                Some(slice::$slice_from_parts(ptr, self.slice_cols))
            }
        } else {
            None
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if self.row_pos + n < self.slice_rows {
            let row: $row_type;
            unsafe {
                let ptr = self.slice_start.offset((self.row_pos + n) as isize * self.row_stride);
                row = slice::$slice_from_parts(ptr, self.slice_cols);
            }

            self.row_pos += n + 1;
            Some(row)
        } else {
            None
        }
    }

    fn count(self) -> usize {
        self.slice_rows - self.row_pos
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.slice_rows - self.row_pos, Some(self.slice_rows - self.row_pos))
    }
}

impl<'a, T> ExactSizeIterator for $rows<'a, T> {}
    );
);

impl_iter_rows!(Rows, &'a [T], from_raw_parts);
impl_iter_rows!(RowsMut, &'a mut [T], from_raw_parts_mut);

/// Creates a `Matrix` from an iterator over slices.
///
/// Each of the slices produced by the iterator will become a row in the matrix.
///
/// # Panics
///
/// Will panic if the iterators items do not have constant length.
///
/// # Examples
///
/// We can create a new matrix from some data.
///
/// ```
/// use rulinalg::matrix::{Matrix, BaseMatrix};
///
/// let a : Matrix<f64> = vec![4f64; 16].chunks(4).collect();
///
/// assert_eq!(a.rows(), 4);
/// assert_eq!(a.cols(), 4);
/// ```
///
/// We can also do more interesting things.
///
/// ```
/// use rulinalg::matrix::{Matrix, BaseMatrix};
///
/// let a = Matrix::new(4,2, (0..8).collect::<Vec<usize>>());
///
/// // Here we skip the first row and take only those
/// // where the first entry is less than 6.
/// let b = a.iter_rows()
///          .skip(1)
///          .filter(|x| x[0] < 6)
///          .collect::<Matrix<usize>>();
///
/// // We take the middle rows
/// assert_eq!(b.into_vec(), vec![2,3,4,5]);
/// ```
impl<'a, T: 'a + Copy> FromIterator<&'a [T]> for Matrix<T> {
    fn from_iter<I: IntoIterator<Item = &'a [T]>>(iterable: I) -> Self {
        let mut mat_data: Vec<T>;
        let cols: usize;
        let mut rows = 0;

        let mut iterator = iterable.into_iter();

        match iterator.next() {
            None => {
                return Matrix {
                    data: Vec::new(),
                    rows: 0,
                    cols: 0,
                }
            }
            Some(row) => {
                rows += 1;
                // Here we set the capacity - get iterator size and the cols
                let (lower_rows, _) = iterator.size_hint();
                cols = row.len();

                mat_data = Vec::with_capacity(lower_rows.saturating_add(1).saturating_mul(cols));
                mat_data.extend_from_slice(row);
            }
        }

        for row in iterator {
            assert!(row.len() == cols, "Iterator slice length must be constant.");
            mat_data.extend_from_slice(row);
            rows += 1;
        }

        mat_data.shrink_to_fit();

        Matrix {
            data: mat_data,
            rows: rows,
            cols: cols,
        }
    }
}

impl<'a, T> IntoIterator for MatrixSlice<'a, T> {
    type Item = &'a T;
    type IntoIter = SliceIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a MatrixSlice<'a, T> {
    type Item = &'a T;
    type IntoIter = SliceIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut MatrixSlice<'a, T> {
    type Item = &'a T;
    type IntoIter = SliceIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for MatrixSliceMut<'a, T> {
    type Item = &'a mut T;
    type IntoIter = SliceIterMut<'a, T>;

    fn into_iter(mut self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, T> IntoIterator for &'a MatrixSliceMut<'a, T> {
    type Item = &'a T;
    type IntoIter = SliceIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut MatrixSliceMut<'a, T> {
    type Item = &'a mut T;
    type IntoIter = SliceIterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{DiagOffset, MatrixSlice, MatrixSliceMut};
    use super::super::slice::{BaseMatrix, BaseMatrixMut};

    #[test]
    fn test_diag_offset_equivalence() {
        // This test will check that `Main`,
        // `Below(0)`, and `Above(0)` are all equivalent.
        let a = matrix![0.0, 1.0, 2.0;
                        3.0, 4.0, 5.0;
                        6.0, 7.0, 8.0];

        // Collect each diagonal and compare them
        let d1 = a.iter_diag(DiagOffset::Main).collect::<Vec<_>>();
        let d2 = a.iter_diag(DiagOffset::Above(0)).collect::<Vec<_>>();
        let d3 = a.iter_diag(DiagOffset::Below(0)).collect::<Vec<_>>();
        assert_eq!(d1, d2);
        assert_eq!(d2, d3);

        let b = MatrixSlice::from_matrix(&a, [0, 0], 2, 3);
        let d1 = b.iter_diag(DiagOffset::Main).collect::<Vec<_>>();
        let d2 = b.iter_diag(DiagOffset::Above(0)).collect::<Vec<_>>();
        let d3 = b.iter_diag(DiagOffset::Below(0)).collect::<Vec<_>>();
        assert_eq!(d1, d2);
        assert_eq!(d2, d3);
    }

    #[test]
    fn test_matrix_diag() {
        let mut a = matrix![0.0, 1.0, 2.0;
                            3.0, 4.0, 5.0;
                            6.0, 7.0, 8.0];

        let diags = vec![0.0, 4.0, 8.0];
        assert_eq!(a.iter_diag(DiagOffset::Main).cloned().collect::<Vec<_>>(), diags);
        let diags = vec![1.0, 5.0];
        assert_eq!(a.iter_diag(DiagOffset::Above(1)).cloned().collect::<Vec<_>>(), diags);
        let diags = vec![3.0, 7.0];
        assert_eq!(a.iter_diag(DiagOffset::Below(1)).cloned().collect::<Vec<_>>(), diags);
        let diags = vec![2.0];
        assert_eq!(a.iter_diag(DiagOffset::Above(2)).cloned().collect::<Vec<_>>(), diags);
        let diags = vec![6.0];
        assert_eq!(a.iter_diag(DiagOffset::Below(2)).cloned().collect::<Vec<_>>(), diags);

        {
            let diags_iter_mut = a.iter_diag_mut(DiagOffset::Main);
            for d in diags_iter_mut {
                *d = 1.0;
            }
        }

        for i in 0..3 {
            assert_eq!(a[[i,i]], 1.0);
        }
    }

    #[test]
    fn test_matrix_slice_diag() {
        let mut a = matrix![0.0, 1.0, 2.0, 3.0;
                            4.0, 5.0, 6.0, 7.0;
                            8.0, 9.0, 10.0, 11.0];
        {
            let b = MatrixSlice::from_matrix(&a, [0, 0], 2, 4);

            let diags = vec![0.0, 5.0];
            assert_eq!(b.iter_diag(DiagOffset::Main).cloned().collect::<Vec<_>>(), diags);
            let diags = vec![1.0, 6.0];
            assert_eq!(b.iter_diag(DiagOffset::Above(1)).cloned().collect::<Vec<_>>(), diags);
            let diags = vec![2.0, 7.0];
            assert_eq!(b.iter_diag(DiagOffset::Above(2)).cloned().collect::<Vec<_>>(), diags);
            let diags = vec![3.0];
            assert_eq!(b.iter_diag(DiagOffset::Above(3)).cloned().collect::<Vec<_>>(), diags);
            let diags = vec![4.0];
            assert_eq!(b.iter_diag(DiagOffset::Below(1)).cloned().collect::<Vec<_>>(), diags);
        }

        {
            let diags_iter_mut = a.iter_diag_mut(DiagOffset::Main);
            for d in diags_iter_mut {
                *d = 1.0;
            }
        }

        for i in 0..3 {
            assert_eq!(a[[i,i]], 1.0);
        }
    }

    #[test]
    fn test_matrix_diag_nth() {
        let a = matrix![0.0, 1.0, 2.0, 3.0;
                        4.0, 5.0, 6.0, 7.0;
                        8.0, 9.0, 10.0, 11.0];

        let mut diags_iter = a.iter_diag(DiagOffset::Main);
        assert_eq!(0.0, *diags_iter.nth(0).unwrap());
        assert_eq!(10.0, *diags_iter.nth(1).unwrap());
        assert_eq!(None, diags_iter.next());

        let mut diags_iter = a.iter_diag(DiagOffset::Above(1));
        assert_eq!(6.0, *diags_iter.nth(1).unwrap());
        assert_eq!(11.0, *diags_iter.next().unwrap());
        assert_eq!(None, diags_iter.next());

        let mut diags_iter = a.iter_diag(DiagOffset::Below(1));
        assert_eq!(9.0, *diags_iter.nth(1).unwrap());
        assert_eq!(None, diags_iter.next());
    }

    #[test]
    fn test_matrix_slice_diag_nth() {
        let a = matrix![0.0, 1.0, 2.0, 3.0;
                        4.0, 5.0, 6.0, 7.0;
                        8.0, 9.0, 10.0, 11.0];
        let b = MatrixSlice::from_matrix(&a, [0,0], 2, 4);

        let mut diags_iter = b.iter_diag(DiagOffset::Main);
        assert_eq!(5.0, *diags_iter.nth(1).unwrap());;
        assert_eq!(None, diags_iter.next());

        let mut diags_iter = b.iter_diag(DiagOffset::Above(1));
        assert_eq!(6.0, *diags_iter.nth(1).unwrap());
        assert_eq!(None, diags_iter.next());

        let mut diags_iter = b.iter_diag(DiagOffset::Below(1));
        assert_eq!(4.0, *diags_iter.nth(0).unwrap());
        assert_eq!(None, diags_iter.next());        
    }

    #[test]
    fn test_matrix_diag_last() {
        let a = matrix![0.0, 1.0, 2.0;
                        3.0, 4.0, 5.0;
                        6.0, 7.0, 8.0];

        let diags_iter = a.iter_diag(DiagOffset::Main);
        assert_eq!(8.0, *diags_iter.last().unwrap());

        let diags_iter = a.iter_diag(DiagOffset::Above(2));
        assert_eq!(2.0, *diags_iter.last().unwrap());

        let diags_iter = a.iter_diag(DiagOffset::Below(2));
        assert_eq!(6.0, *diags_iter.last().unwrap());    
    }

    #[test]
    fn test_matrix_slice_diag_last() {
        let a = matrix![0.0, 1.0, 2.0;
                        3.0, 4.0, 5.0;
                        6.0, 7.0, 8.0];
        let b = MatrixSlice::from_matrix(&a, [0,0], 3, 2);

        {
            let diags_iter = b.iter_diag(DiagOffset::Main);
            assert_eq!(4.0, *diags_iter.last().unwrap());
        }

        {
            let diags_iter = b.iter_diag(DiagOffset::Above(1));
            assert_eq!(1.0, *diags_iter.last().unwrap());
        }

        {
            let diags_iter = b.iter_diag(DiagOffset::Below(2));
            assert_eq!(6.0, *diags_iter.last().unwrap());
        }
    }

    #[test]
    fn test_matrix_diag_count() {
        let a = matrix![0.0, 1.0, 2.0;
                        3.0, 4.0, 5.0;
                        6.0, 7.0, 8.0];

        assert_eq!(3, a.iter_diag(DiagOffset::Main).count());
        assert_eq!(2, a.iter_diag(DiagOffset::Above(1)).count());
        assert_eq!(1, a.iter_diag(DiagOffset::Above(2)).count());
        assert_eq!(2, a.iter_diag(DiagOffset::Below(1)).count());
        assert_eq!(1, a.iter_diag(DiagOffset::Below(2)).count());

        let mut diags_iter = a.iter_diag(DiagOffset::Main);
        diags_iter.next();
        assert_eq!(2, diags_iter.count());
    }

    #[test]
    fn test_matrix_diag_size_hint() {
        let a = matrix![0.0, 1.0, 2.0;
                        3.0, 4.0, 5.0;
                        6.0, 7.0, 8.0];

        let mut diags_iter = a.iter_diag(DiagOffset::Main);
        assert_eq!((3, Some(3)), diags_iter.size_hint());
        diags_iter.next();

        assert_eq!((2, Some(2)), diags_iter.size_hint());
        diags_iter.next();
        diags_iter.next();

        assert_eq!((0, Some(0)), diags_iter.size_hint());
        assert_eq!(None, diags_iter.next());
        assert_eq!((0, Some(0)), diags_iter.size_hint());
    }


    #[test]
    fn test_matrix_rows() {
        let mut a = matrix![0, 1, 2;
                            3, 4, 5;
                            6, 7, 8];
        let data = [[0, 1, 2], [3, 4, 5], [6, 7, 8]];

        for (i, row) in a.iter_rows().enumerate() {
            assert_eq!(data[i], *row);
        }

        for (i, row) in a.iter_rows_mut().enumerate() {
            assert_eq!(data[i], *row);
        }

        for row in a.iter_rows_mut() {
            for r in row {
                *r = 0;
            }
        }

        assert_eq!(a.into_vec(), vec![0; 9]);
    }

    #[test]
    fn test_matrix_slice_rows() {
        let a = matrix![0, 1, 2;
                        3, 4, 5;
                        6, 7, 8];;

        let b = MatrixSlice::from_matrix(&a, [0, 0], 2, 2);

        let data = [[0, 1], [3, 4]];

        for (i, row) in b.iter_rows().enumerate() {
            assert_eq!(data[i], *row);
        }
    }

    #[test]
    fn test_matrix_slice_mut_rows() {
        let mut a = matrix![0, 1, 2;
                            3, 4, 5;
                            6, 7, 8];

        {
            let mut b = MatrixSliceMut::from_matrix(&mut a, [0, 0], 2, 2);

            let data = [[0, 1], [3, 4]];

            for (i, row) in b.iter_rows().enumerate() {
                assert_eq!(data[i], *row);
            }

            for (i, row) in b.iter_rows_mut().enumerate() {
                assert_eq!(data[i], *row);
            }

            for row in b.iter_rows_mut() {
                for r in row {
                    *r = 0;
                }
            }
        }

        assert_eq!(a.into_vec(), vec![0, 0, 2, 0, 0, 5, 6, 7, 8]);
    }

    #[test]
    fn test_matrix_rows_nth() {
        let a = matrix![0, 1, 2;
                        3, 4, 5;
                        6, 7, 8];

        let mut row_iter = a.iter_rows();

        assert_eq!([0, 1, 2], *row_iter.nth(0).unwrap());
        assert_eq!([6, 7, 8], *row_iter.nth(1).unwrap());

        assert_eq!(None, row_iter.next());
    }

    #[test]
    fn test_matrix_rows_last() {
        let a = matrix![0, 1, 2;
                        3, 4, 5;
                        6, 7, 8];

        let row_iter = a.iter_rows();

        assert_eq!([6, 7, 8], *row_iter.last().unwrap());

        let mut row_iter = a.iter_rows();

        row_iter.next();
        assert_eq!([6, 7, 8], *row_iter.last().unwrap());

        let mut row_iter = a.iter_rows();

        row_iter.next();
        row_iter.next();
        row_iter.next();
        row_iter.next();

        assert_eq!(None, row_iter.last());
    }

    #[test]
    fn test_matrix_rows_count() {
        let a = matrix![0, 1, 2;
                        3, 4, 5;
                        6, 7, 8];

        let row_iter = a.iter_rows();

        assert_eq!(3, row_iter.count());

        let mut row_iter_2 = a.iter_rows();
        row_iter_2.next();
        assert_eq!(2, row_iter_2.count());
    }

    #[test]
    fn test_matrix_rows_size_hint() {
        let a = matrix![0, 1, 2;
                        3, 4, 5;
                        6, 7, 8];

        let mut row_iter = a.iter_rows();

        assert_eq!((3, Some(3)), row_iter.size_hint());

        row_iter.next();

        assert_eq!((2, Some(2)), row_iter.size_hint());
        row_iter.next();
        row_iter.next();

        assert_eq!((0, Some(0)), row_iter.size_hint());

        assert_eq!(None, row_iter.next());
        assert_eq!((0, Some(0)), row_iter.size_hint());
    }

    #[test]
    fn into_iter_compile() {
        let a = matrix![2.0, 2.0, 2.0;
                        2.0, 2.0, 2.0;
                        2.0, 2.0, 2.0];
        let mut b = MatrixSlice::from_matrix(&a, [1, 1], 2, 2);

        for _ in b {
        }

        for _ in &b {
        }

        for _ in &mut b {
        }
    }

    #[test]
    fn into_iter_mut_compile() {
        let mut a = matrix![2.0, 2.0, 2.0;
                            2.0, 2.0, 2.0;
                            2.0, 2.0, 2.0];

        {
            let b = MatrixSliceMut::from_matrix(&mut a, [1, 1], 2, 2);

            for v in b {
                *v = 1.0;
            }
        }

        {
            let b = MatrixSliceMut::from_matrix(&mut a, [1, 1], 2, 2);

            for _ in &b {
            }
        }

        {
            let mut b = MatrixSliceMut::from_matrix(&mut a, [1, 1], 2, 2);

            for v in &mut b {
                *v = 1.0;
            }
        }
    }
}
