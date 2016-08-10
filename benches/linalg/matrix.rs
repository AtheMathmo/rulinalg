use rulinalg::matrix::Matrix;
use rulinalg::matrix::slice::BaseMatrix;
use test::Bencher;
use test::black_box;

#[bench]
fn empty(b: &mut Bencher) {
    b.iter(|| 1)
}

#[bench]
fn mat_ref_add_100_100(b: &mut Bencher) {

    let a = Matrix::new(100, 100, vec![2.0;10000]);
    let c = Matrix::new(100, 100, vec![3.0;10000]);

    b.iter(|| {
    	&a + &c
    })
}

#[bench]
fn mat_create_add_100_100(b: &mut Bencher) {
    let c = Matrix::new(100, 100, vec![3.0;10000]);

    b.iter(|| {
    	let a = Matrix::new(100, 100, vec![2.0;10000]);
    	a + &c
    })
}

#[bench]
fn mat_create_100_100(b: &mut Bencher) {
    b.iter(|| {
    	let a = Matrix::new(100, 100, vec![2.0;10000]);
    	a
    })
}

#[bench]
fn mat_mul_10_10(b: &mut Bencher) {

    let a = Matrix::new(10, 10, vec![2f32; 100]);
    let c = Matrix::new(10, 10, vec![3f32; 100]);

    b.iter(|| &a * &c)
}

#[bench]
fn mat_mul_128_100(b: &mut Bencher) {

    let a = Matrix::new(128, 100, vec![2f32; 12800]);
    let c = Matrix::new(100, 128, vec![3f32; 12800]);

    b.iter(|| &a * &c)
}

#[bench]
fn mat_mul_128_1000(b: &mut Bencher) {

    let a = Matrix::new(128, 1000, vec![2f32; 128000]);
    let c = Matrix::new(1000, 128, vec![3f32; 128000]);

    b.iter(|| &a * &c)
}

#[bench]
fn mat_elemul_63_1000(b: &mut Bencher) {

    let a = Matrix::new(63, 1000, vec![2f32; 63000]);
    let c = Matrix::new(63, 1000, vec![3f32; 63000]);

    b.iter(|| a.elemul(&c))
}


#[bench]
fn mat_mat_elemul_63_1000(b: &mut Bencher) {

    use rulinalg::utils;
    use std::ops::Mul;

    let a = Matrix::new(63, 1000, vec![2f32; 63000]);
    let c = Matrix::new(63, 1000, vec![3f32; 63000]);

    b.iter(|| {
        utils::vec_bin_op(a.data(), c.data(), f32::mul)
    })
}

#[bench]
fn mat_elediv_63_1000(b: &mut Bencher) {

    let a = Matrix::new(63, 1000, vec![2f32; 63000]);
    let c = Matrix::new(63, 1000, vec![3f32; 63000]);

    b.iter(|| a.elediv(&c))
}

#[bench]
fn mat_sum_rows_and_cols_128_100(b: &mut Bencher) {

    let v = (0..100).collect::<Vec<_>>();
    let mut data = Vec::with_capacity(128000);
    for _ in 0..128 {
        data.extend_from_slice(&v);
    }
    let m = Matrix::new(128, 100, data);

    b.iter(|| {
        let sum_rows = black_box(m.sum_rows());
        let sum_cols = m.sum_cols();
        let sum = m.sum();
        assert_eq!(sum_cols.data(), &vec![100 * 99 / 2; 128]);
        assert_eq!(sum_rows.data(), &(0..100).map(|i| i * 128).collect::<Vec<_>>());
        assert_eq!(sum, 100 * 99 / 2 * 128);
    })
}
