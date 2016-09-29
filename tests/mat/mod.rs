use rulinalg::vector::Vector;
use rulinalg::matrix::Matrix;
use rulinalg::matrix::slice::BaseMatrix;

#[test]
fn test_l_triangular_solve_errs() {
    let a = Matrix::<f64>::new(0, 0, vec![]);
    assert!(a.solve_l_triangular(Vector::new(vec![])).is_err());

    let a = Matrix::new(1, 1, vec![0.0]);
    assert!(a.solve_l_triangular(Vector::new(vec![1.0])).is_err());
}

#[test]
fn test_u_triangular_solve_errs() {
    let a = Matrix::<f64>::new(0, 0, vec![]);
    assert!(a.solve_u_triangular(Vector::new(vec![])).is_err());

    let a = Matrix::new(1, 1, vec![0.0]);
    assert!(a.solve_u_triangular(Vector::new(vec![1.0])).is_err());
}

#[test]
fn matrix_lup_decomp() {
    let a = Matrix::new(3, 3, vec![1., 3., 5., 2., 4., 7., 1., 1., 0.]);

    let (l, u, p) = a.lup_decomp().expect("Matrix SHOULD be able to be decomposed...");

    let l_true = vec![1., 0., 0., 0.5, 1., 0., 0.5, -1., 1.];
    let u_true = vec![2., 4., 7., 0., 1., 1.5, 0., 0., -2.];
    let p_true = vec![0., 1., 0., 1., 0., 0., 0., 0., 1.];

    assert_eq!(*p.data(), p_true);
    assert_eq!(*l.data(), l_true);
    assert_eq!(*u.data(), u_true);

    let e = Matrix::<f64>::new(5,
                               5,
                               vec![1., 2., 3., 4., 5., 3., 0., 4., 5., 6., 2., 1., 2., 3., 4.,
                                    0., 0., 0., 6., 5., 0., 0., 0., 5., 6.]);

    let (l, u, p) = e.lup_decomp().expect("Matrix SHOULD be able to be decomposed...");
    let k = p.transpose() * l * u;

    for i in 0..25 {
        assert_eq!(e.data()[i], k.data()[i]);
    }
}

#[test]
fn cholesky() {
    let a = Matrix::new(3, 3, vec![25., 15., -5., 15., 18., 0., -5., 0., 11.]);

    let l = a.cholesky();

    assert!(l.is_ok());

    assert_eq!(*l.unwrap().data(), vec![5., 0., 0., 3., 3., 0., -1., 1., 3.]);
}

#[test]
fn qr() {
    let a = Matrix::new(3, 3, vec![12., -51., 4., 6., 167., -68., -4., 24., -41.]);

    let res = a.qr_decomp();

    assert!(res.is_ok());

    let (q, r) = res.unwrap();

    let tol = 1e-6;

    let true_q = Matrix::new(3,
                             3,
                             vec![-0.857143, 0.394286, 0.331429, -0.428571, -0.902857, -0.034286,
                                  0.285715, -0.171429, 0.942857]);
    let true_r = Matrix::new(3, 3, vec![-14., -21., 14., 0., -175., 70., 0., 0., -35.]);

    let q_diff = (q - &true_q).into_vec();
    let r_diff = (r - &true_r).into_vec();

    for (val, expected) in q_diff.into_iter().zip(true_q.data().iter()) {
        assert!(val < tol, format!("diff is {0}, expecting {1}", val, expected));
    }

    for (val, expected) in r_diff.into_iter().zip(true_r.data().iter()) {
        assert!(val < tol, format!("diff is {0}, expecting {1}", val, expected));
    }

}
