#[macro_use]
extern crate rulinalg;

fn main() {
    // The purpose of this example is to demonstrate the type of errors generated
    // by `assert_matrix_eq!` when the comparison fails (which the below call will).
    let a = matrix![1.00000000, 2.0000000;
                    3.00000000, 4.0000000];
    let b = matrix![1.00000001, 2.0000000;
                    3.00000004, 4.0000000];
    assert_matrix_eq!(a, b, comp = ulp, tol = 4);
}
