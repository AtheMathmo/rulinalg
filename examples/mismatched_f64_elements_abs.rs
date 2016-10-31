#[macro_use]
extern crate rulinalg;

fn main() {
    // The purpose of this example is to demonstrate the type of errors generated
    // by `assert_matrix_eq!` when the comparison fails (which the below call will).
    let a = matrix![1.00, 2.00;
                    3.00, 4.00];
    let b = matrix![1.01, 2.00;
                    3.40, 4.00];
    assert_matrix_eq!(a, b, comp = abs, tol = 1e-8);
}
