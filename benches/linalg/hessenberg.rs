use rulinalg::matrix::decomposition::{Decomposition, HessenbergDecomposition};

use linalg::util::reproducible_random_matrix;

use test::Bencher;

#[bench]
fn hessenberg_decomposition_decompose_100x100(b: &mut Bencher) {
    let x = reproducible_random_matrix(100, 100);
    b.iter(|| {
        HessenbergDecomposition::decompose(x.clone())
    })
}

#[bench]
fn hessenberg_decomposition_decompose_unpack_100x100(b: &mut Bencher) {
    let x = reproducible_random_matrix(100, 100);
    b.iter(|| {
        HessenbergDecomposition::decompose(x.clone()).unpack()
    })
}

#[bench]
fn hessenberg_decomposition_decompose_200x200(b: &mut Bencher) {
    let x = reproducible_random_matrix(200, 200);
    b.iter(|| {
        HessenbergDecomposition::decompose(x.clone())
    })
}

#[bench]
fn hessenberg_decomposition_decompose_unpack_200x200(b: &mut Bencher) {
    let x = reproducible_random_matrix(200, 200);
    b.iter(|| {
        HessenbergDecomposition::decompose(x.clone()).unpack()
    })
}

#[bench]
#[allow(deprecated)]
fn upper_hessenberg_100x100(b: &mut Bencher) {
    let x = reproducible_random_matrix(100, 100);
    b.iter(|| {
        x.clone().upper_hessenberg()
    })
}

#[bench]
#[allow(deprecated)]
fn upper_hess_decomp_100x100(b: &mut Bencher) {
    let x = reproducible_random_matrix(100, 100);
    b.iter(|| {
        x.clone().upper_hess_decomp()
    })
}

#[bench]
#[allow(deprecated)]
fn upper_hessenberg_200x200(b: &mut Bencher) {
    let x = reproducible_random_matrix(200, 200);
    b.iter(|| {
        x.clone().upper_hessenberg()
    })
}

#[bench]
#[allow(deprecated)]
fn upper_hess_decomp_200x200(b: &mut Bencher) {
    let x = reproducible_random_matrix(200, 200);
    b.iter(|| {
        x.clone().upper_hess_decomp()
    })
}
