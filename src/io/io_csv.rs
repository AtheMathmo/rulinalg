//! csv read / write module
use csv;
use rustc_serialize::{Decodable, Encodable};
use std::io::{Read, Write};

use super::super::matrix::{Matrix, BaseMatrix};

impl<T> Matrix<T> where T: Decodable {

    /// Read csv file as Matrix
    pub fn read_csv<'a, R: Read>(mut reader: csv::Reader<R>)
        -> Result<Matrix<T>, csv::Error> {

        // headers read 1st row regardless of has_headers property
        let header: Vec<String> = try!(reader.headers());

        let mut nrows = 0;
        let ncols = header.len();

        let mut records: Vec<T> = vec![];
        for record in reader.decode() {
            let values: Vec<T> = try!(record);
            records.extend(values);
            nrows += 1;
        }
        Ok(Matrix::new(nrows, ncols, records))
    }
}

impl<T> Matrix<T> where T: Encodable {

    /// Write Matrix<f64> as csv file
    pub fn write_csv<W: Write>(&self, writer: &mut csv::Writer<W>)
        -> Result<(), csv::Error> {

        for row in self.row_iter() {
            try!(writer.encode(row.raw_slice()));
        }
        Ok(())
    }
}


#[cfg(test)]
mod tests {

    use csv;
    use super::super::super::matrix::Matrix;

    #[test]
    fn test_read_csv_with_header() {
        let data = "A,B,C
1,7,1.1
1,3,2.2
1,1,4.5";
        let rdr = csv::Reader::from_string(data).has_headers(true);
        let res = Matrix::<f64>::read_csv(rdr).unwrap();

        let exp = matrix![1., 7., 1.1;
                          1., 3., 2.2;
                          1., 1., 4.5];
        assert_matrix_eq!(res, exp);
    }

    #[test]
    fn test_read_csv_without_header() {
        let data = "1,7,1.1
1,3,2.2
1,1,4.5";
        let rdr = csv::Reader::from_string(data).has_headers(false);
        let res = Matrix::<f64>::read_csv(rdr).unwrap();

        let exp = matrix![1., 7., 1.1;
                          1., 3., 2.2;
                          1., 1., 4.5];
        assert_matrix_eq!(res, exp);
    }

    #[test]
    fn test_read_csv_integer_like() {
        let data = "1,7,1
1,3,2
1,1,4";
        let rdr = csv::Reader::from_string(data).has_headers(false);
        let res = Matrix::<f64>::read_csv(rdr).unwrap();

        let exp = matrix![1., 7., 1.;
                          1., 3., 2.;
                          1., 1., 4.];
        assert_matrix_eq!(res, exp);
    }

    #[test]
    fn test_read_csv_with_header_int() {
        let data = "A,B,C
1,2,3
4,5,6
7,8,9";
        let rdr = csv::Reader::from_string(data).has_headers(true);
        let res = Matrix::<usize>::read_csv(rdr).unwrap();

        let exp = matrix![1, 2, 3;
                          4, 5, 6;
                          7, 8, 9];
        assert_matrix_eq!(res, exp);
    }

    #[test]
    fn test_read_csv_empty() {
        let data = "";
        let rdr = csv::Reader::from_string(data).has_headers(true);
        let res = Matrix::<f64>::read_csv(rdr).unwrap();
        let exp: Matrix<f64> = Matrix::new(0, 0, vec![]);
        assert_matrix_eq!(res, exp);
    }

    #[test]
    fn test_read_csv_error_different_items() {
        let data = "A,B,C
1,7,1.1
1,3
1,1,4.5";
        let rdr = csv::Reader::from_string(data).has_headers(true);
        let res = Matrix::<f64>::read_csv(rdr);
        assert!(res.is_err())
    }

    #[test]
    fn test_write_csv() {
        let mat = matrix![1., 7., 1.1;
                          1., 3., 2.2;
                          1., 1., 4.5];
        let mut wtr = csv::Writer::from_memory();
        mat.write_csv(&mut wtr).unwrap();
        let res = wtr.as_string();
        assert_eq!(res, "1.0,7.0,1.1\n1.0,3.0,2.2\n1.0,1.0,4.5\n");

        // test round-trip
        let rdr = csv::Reader::from_string(res).has_headers(false);
        let res = Matrix::<f64>::read_csv(rdr).unwrap();
        assert_matrix_eq!(res, mat);
    }
}