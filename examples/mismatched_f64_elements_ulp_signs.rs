#[macro_use]
extern crate rulinalg;

fn main() {
    // Demonstrates the error generated when the signs differ when using ULP-based comparison.
    let a = matrix![1.00000000, -2.0000000;
                    3.00000000,  4.0000000];
    let b = matrix![1.00000000,  2.0000000;
                    3.00000000,  4.0000000];
    assert_matrix_eq!(a, b, comp = ulp, tol = 4);
}
