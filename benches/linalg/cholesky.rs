use rulinalg::matrix::Matrix;
use rulinalg::matrix::decomposition::{Cholesky, Decomposition};
use test::Bencher;

#[bench]
fn cholesky_decompose_unpack_100x100(b: &mut Bencher) {
    let n = 100;
    let x = Matrix::<f64>::identity(n);
    b.iter(|| {
        // Assume that the cost of cloning x is roughly
        // negligible in comparison with the cost of LU
        Cholesky::decompose(x.clone()).expect("Matrix is invertible")
                                      .unpack()
    })
}

#[bench]
fn cholesky_decompose_unpack_500x500(b: &mut Bencher) {
    let n = 500;
    let x = Matrix::<f64>::identity(n);
    b.iter(|| {
        // Assume that the cost of cloning x is roughly
        // negligible in comparison with the cost of LU
        Cholesky::decompose(x.clone()).expect("Matrix is invertible")
                                      .unpack()
    })
}

#[bench]
fn cholesky_100x100(b: &mut Bencher) {
    // Benchmark for legacy cholesky(). Remove when
    // cholesky() has been removed.
    let n = 100;
    let x = Matrix::<f64>::identity(n);
    b.iter(|| {
        x.cholesky().expect("Matrix is invertible")
    })
}

#[bench]
fn cholesky_500x500(b: &mut Bencher) {
    // Benchmark for legacy cholesky(). Remove when
    // cholesky() has been removed.
    let n = 500;
    let x = Matrix::<f64>::identity(n);
    b.iter(|| {
        x.cholesky().expect("Matrix is invertible")
    })
}
