use super::{BaseMatrix, Row, RowMut, Column, ColumnMut};
use super::super::vector::Vector;

macro_rules! impl_row_to_vector (
    ($t:ident) => (

impl<'a, T: Clone> From<$t<'a, T>> for Vector<T> {
    fn from(row: $t<'a, T>) -> Vector<T> {
        Vector::new(row.raw_slice())
    }
}

    );
);
impl_row_to_vector!(Row);
impl_row_to_vector!(RowMut);

macro_rules! impl_column_to_vector (
    ($t:ident) => (

impl<'a, T: Clone> From<$t<'a, T>> for Vector<T> {
    fn from(column: $t<'a, T>) -> Vector<T> {
        column.iter().map(|x| x.clone()).collect::<Vector<T>>()
    }
}

    );
);
impl_column_to_vector!(Column);
impl_column_to_vector!(ColumnMut);

#[cfg(test)]
mod tests {
    use super::super::{Matrix, BaseMatrix, BaseMatrixMut};
    use super::super::super::vector::Vector;

    #[test]
    fn test_row_convert() {
        let a: Matrix<i64> = matrix![1, 2, 3, 4;
                                     5, 6, 7, 8;
                                     9, 10, 11, 12];
        let row = a.row(1);
        let v: Vector<i64> = row.into();
        assert_eq!(v, Vector::new(vec![5, 6, 7, 8]));

        let row = a.row(2);
        let v = Vector::from(row);
        assert_eq!(v, Vector::new(vec![9, 10, 11, 12]));
    }

    #[test]
    fn test_row_convert_mut() {
        let mut a: Matrix<i64> = matrix![1, 2, 3, 4;
                                         5, 6, 7, 8;
                                         9, 10, 11, 12];

        let row = a.row_mut(1);
        let v: Vector<i64> = row.into();
        assert_eq!(v, Vector::new(vec![5, 6, 7, 8]));

        let mut a: Matrix<i64> = matrix![1, 2, 3, 4;
                                         5, 6, 7, 8;
                                         9, 10, 11, 12];
        let row = a.row_mut(2);
        let v = Vector::from(row);
        assert_eq!(v, Vector::new(vec![9, 10, 11, 12]));
    }

    #[test]
    fn test_column_convert() {
        let a: Matrix<i64> = matrix![1, 2, 3, 4;
                                     5, 6, 7, 8;
                                     9, 10, 11, 12];
        let row = a.col(1);
        let v: Vector<i64> = row.into();
        assert_eq!(v, Vector::new(vec![2, 6, 10]));

        let row = a.col(2);
        let v = Vector::from(row);
        assert_eq!(v, Vector::new(vec![3, 7, 11]));
    }

    #[test]
    fn test_column_convert_mut() {
        let mut a: Matrix<i64> = matrix![1, 2, 3, 4;
                                         5, 6, 7, 8;
                                         9, 10, 11, 12];

        let row = a.col_mut(1);
        let v: Vector<i64> = row.into();
        assert_eq!(v, Vector::new(vec![2, 6, 10]));

        let mut a: Matrix<i64> = matrix![1, 2, 3, 4;
                                         5, 6, 7, 8;
                                         9, 10, 11, 12];
        let row = a.col_mut(2);
        let v = Vector::from(row);
        assert_eq!(v, Vector::new(vec![3, 7, 11]));
    }
}
