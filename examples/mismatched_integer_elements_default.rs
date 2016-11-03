#[macro_use]
extern crate rulinalg;

fn main() {
    // The purpose of this example is to demonstrate the type of errors generated
    // by `assert_matrix_eq!` when the comparison fails (which the below call will).
    let a = matrix![1, 2;
                    3, 4];
    let b = matrix![1, 5;
                    3, 4];
    assert_matrix_eq!(a, b);
}
