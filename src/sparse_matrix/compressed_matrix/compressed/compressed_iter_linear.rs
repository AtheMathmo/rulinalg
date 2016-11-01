use std::slice;

use super::{CompressedLinear, CompressedLinearMut};

macro_rules! impl_iter_linear (
    ($iter_struct:ident, $data_type:ty, $slice_from_parts:ident) => (
impl<'a, T> Iterator for $iter_struct<'a, T> {
    type Item = (&'a [usize], $data_type);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_pos >= self.positions {
            return None;
        }

        let data: $data_type;
		let ptr_range = (self.ptrs[self.current_pos], self.ptrs[self.current_pos + 1]);
        unsafe {
            let ptr = self.data.offset(ptr_range.0 as isize);
            data = slice::$slice_from_parts(ptr, ptr_range.1 - ptr_range.0);
        }
        self.current_pos += 1;
        Some((&self.indices[ptr_range.0..ptr_range.1], data))
    }

    fn last(self) -> Option<Self::Item> {
        if self.current_pos >= self.positions {
            return None;
        }

        let data: $data_type;
		let ptr_range = (self.ptrs[self.positions - 1], self.ptrs[self.positions]);
        unsafe {
            let ptr = self.data.offset(ptr_range.0 as isize);
            data = slice::$slice_from_parts(ptr, ptr_range.1 - ptr_range.0);
        }
        Some((&self.indices[ptr_range.0..ptr_range.1], data))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if self.current_pos + n >= self.positions {
            return None;
        }

        let data: $data_type;
		let ptr_range = (self.ptrs[self.current_pos + n], self.ptrs[self.current_pos + n + 1]);
        unsafe {
            let ptr = self.data.offset(ptr_range.0 as isize);
            data = slice::$slice_from_parts(ptr, ptr_range.1 - ptr_range.0);
        }
        self.current_pos += n + 1;
        Some((&self.indices[ptr_range.0..ptr_range.1], data))
    }

    fn count(self) -> usize {
        self.positions - self.current_pos
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.positions - self.current_pos, Some(self.positions - self.current_pos))
    }
}
    );
);

impl_iter_linear!(CompressedLinear, &'a [T], from_raw_parts);
impl_iter_linear!(CompressedLinearMut, &'a mut [T], from_raw_parts_mut);

impl<'a, T> ExactSizeIterator for CompressedLinear<'a, T> {}
impl<'a, T> ExactSizeIterator for CompressedLinearMut<'a, T> {}


#[cfg(test)]
mod tests {
    use sparse_matrix::{CompressedMatrix, SparseMatrix};
    use sparse_matrix::compressed_matrix::Compressed;

    #[test]
    fn test_matrix_positions_count() {
        let a = Compressed::new(7,
                                5,
                                vec![4, 2, 3, 5, 1, 6],
                                vec![1, 2, 0, 3, 0, 2],
                                vec![0, 0, 2, 2, 4, 4, 6, 6]);

        let row_iter = a.iter_linear();

        assert_eq!(7, row_iter.count());

        let mut row_iter_6 = a.iter_linear();
        row_iter_6.next();
        assert_eq!(6, row_iter_6.count());
    }

    #[test]
    fn test_matrix_positions_last() {
        let a = Compressed::new(6,
                                5,
                                vec![4, 2, 3, 5, 1, 6],
                                vec![1, 2, 0, 3, 0, 2],
                                vec![0, 0, 2, 2, 4, 4, 6]);

        let row_iter = a.iter_linear();

        let last0 = row_iter.last().unwrap();
        assert_eq!([0, 2], last0.0);
        assert_eq!([1, 6], last0.1);

        let mut row_iter = a.iter_linear();

        row_iter.next();

        let last1 = row_iter.last().unwrap();
        assert_eq!([0, 2], last1.0);
        assert_eq!([1, 6], last1.1);

        let mut row_iter = a.iter_linear();

        row_iter.next();
        row_iter.next();
        row_iter.next();
        row_iter.next();
        row_iter.next();
        row_iter.next();

        assert_eq!(None, row_iter.last());
    }

    /// Matrix:
    ///
    /// 0 4 2 0
    /// 0 0 0 0
    /// 3 0 0 5
    /// 0 0 0 0
    /// 1 0 6 0
    #[test]
    fn test_matrix_positions_next() {
        let mut a = Compressed::new(7,
                                    5,
                                    vec![4, 2, 3, 5, 1, 6],
                                    vec![1, 2, 0, 3, 0, 2],
                                    vec![0, 0, 2, 2, 4, 4, 6, 6]);

        let data = [[4, 2], [3, 5], [1, 6]];
        let mut counter = 0;
        let indices = [[1, 2], [0, 3], [0, 2]];

        for (cols, row) in a.iter_linear() {
            if row.len() != 0 {
                assert_eq!(data[counter], *row);
                assert_eq!(indices[counter], *cols);
                counter += 1;
            }
        }

        counter = 0;

        for (cols, row) in a.iter_linear_mut() {
            if row.len() != 0 {
                assert_eq!(data[counter], *row);
                assert_eq!(indices[counter], *cols);
                counter += 1;
            }
        }

        for (_, row) in a.iter_linear_mut() {
            for value in row {
                *value = 0;
            }
        }

        assert_eq!(a.into_vec(), &[0; 6]);
    }

    #[test]
    fn test_matrix_positions_nth() {
        let a = Compressed::new(7,
                                5,
                                vec![4, 2, 3, 5, 1, 6],
                                vec![1, 2, 0, 3, 0, 2],
                                vec![0, 0, 2, 2, 4, 4, 6, 6]);

        let mut row_iter = a.iter_linear();

        let nth0 = row_iter.nth(1).unwrap();
        assert_eq!([1usize, 2], nth0.0);
        assert_eq!([4, 2], nth0.1);

        let nth1 = row_iter.nth(1).unwrap();
        assert_eq!([0usize, 3], nth1.0);
        assert_eq!([3, 5], nth1.1);

        let nth2 = row_iter.nth(1).unwrap();
        assert_eq!([0usize, 2], nth2.0);
        assert_eq!([1, 6], nth2.1);

        row_iter.next();
        assert_eq!(None, row_iter.next());
    }

    #[test]
    fn test_matrix_positions_size_hint() {
        let a = Compressed::new(7,
                                5,
                                vec![4, 2, 3, 5, 1, 6],
                                vec![1, 2, 0, 3, 0, 2],
                                vec![0, 0, 2, 2, 4, 4, 6, 6]);

        let mut row_iter = a.iter_linear();

        assert_eq!((7, Some(7)), row_iter.size_hint());
        row_iter.next();
        assert_eq!((6, Some(6)), row_iter.size_hint());
        row_iter.next();
        assert_eq!((5, Some(5)), row_iter.size_hint());
        row_iter.next();
        assert_eq!((4, Some(4)), row_iter.size_hint());
        row_iter.next();
        assert_eq!((3, Some(3)), row_iter.size_hint());
        row_iter.next();
        assert_eq!((2, Some(2)), row_iter.size_hint());
        row_iter.next();
        assert_eq!((1, Some(1)), row_iter.size_hint());
        row_iter.next();
        assert_eq!((0, Some(0)), row_iter.size_hint());
        assert_eq!(None, row_iter.next());
        assert_eq!((0, Some(0)), row_iter.size_hint());

    }
}
