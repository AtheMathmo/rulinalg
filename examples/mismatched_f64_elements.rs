#[macro_use]
extern crate rulinalg;

fn main() {
    let a = matrix![1.00, 2.00;
                    3.00, 4.00];
    let b = matrix![1.01, 2.00;
                    3.40, 4.00];
    assert_matrix_eq!(a, b, comp = abs, tol = 1e-8);
}
