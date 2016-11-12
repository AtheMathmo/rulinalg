#[macro_use]
extern crate rulinalg;

use rulinalg::matrix::Matrix;

fn main() {
    // The purpose of this example is to demonstrate the type of errors generated
    // by `assert_matrix_eq!` when the comparison fails in the case
    // when the number of mismatched elements is large.
    let a = Matrix::<u64>::zeros(100, 100);
    let b = Matrix::<u64>::ones(100, 100);
    assert_matrix_eq!(a, b, comp = exact);
}
