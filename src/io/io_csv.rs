//! csv read / write module
use csv::{Reader, Writer};
use std::io::{Read, Write};

use super::super::matrix::{Matrix, BaseMatrix};

/// Read csv file as Matrix<f64>
pub fn read<'a, R: Read>(mut reader: Reader<R>) -> Matrix<f64> {
    // headers read 1st row regardless of has_headers property
    let header : Vec<String> = reader.headers().unwrap();

    let mut nrows = 0;
    let ncols = header.len();

    let mut records: Vec<f64> = Vec::with_capacity(header.len() * 100);
    for record in reader.decode() {
        let values: Vec<f64> = record.unwrap();
        records.extend(values);
        nrows += 1;
    }
    records.shrink_to_fit();
    Matrix::new(nrows, ncols, records)
}

/// Write Matrix<f64> as csv file
pub fn write<'a, W: Write>(writer: &'a mut Writer<W>, inputs: &Matrix<f64>) {
    for row in inputs.row_iter() {
        writer.encode(row.raw_slice()).unwrap();
    }
}

#[cfg(test)]
mod tests {

    use csv;
    use super::{read, write};

    #[test]
    fn test_read_csv_with_header() {
        let data = "A,B,C
1,7,1.1
1,3,2.2
1,1,4.5";
        let rdr = csv::Reader::from_string(data).has_headers(true);
        let res = read(rdr);

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
        let res = read(rdr);

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
        let res = read(rdr);

        let exp = matrix![1., 7., 1.;
                          1., 3., 2.;
                          1., 1., 4.];
        assert_matrix_eq!(res, exp);
    }

    #[test]
    #[should_panic]
    fn test_read_csv_different_items() {
        let data = "A,B,C
1,7,1.1
1,3
1,1,4.5";
        let rdr = csv::Reader::from_string(data).has_headers(false);
        let res = read(rdr);

        let exp = matrix![1., 7., 1.1;
                          1., 3., 2.2;
                          1., 1., 4.5];
        assert_matrix_eq!(res, exp);
    }

    #[test]
    fn test_write_csv() {
        let mat = matrix![1., 7., 1.1;
                          1., 3., 2.2;
                          1., 1., 4.5];
        let mut wtr = csv::Writer::from_memory();
        write(&mut wtr, &mat);
        let res = wtr.as_string();
        assert_eq!(res, "1.0,7.0,1.1\n1.0,3.0,2.2\n1.0,1.0,4.5\n");

        // test round-trip
        let rdr = csv::Reader::from_string(res).has_headers(false);
        let res = read(rdr);
        assert_matrix_eq!(res, mat);
    }
}