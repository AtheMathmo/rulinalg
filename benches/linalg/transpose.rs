use rulinalg::matrix::{Matrix, BaseMatrix};
use test::Bencher;

#[bench]
fn allocating_transpose_10_10(b: &mut Bencher) {
    let a = Matrix::<f64>::zeros(10, 10);

    b.iter(|| {
    	a.transpose()
    })
}

#[bench]
fn allocating_transpose_10_100(b: &mut Bencher) {
    let a = Matrix::<f64>::zeros(10, 100);

    b.iter(|| {
    	a.transpose()
    })
}

#[bench]
fn allocating_transpose_100_10(b: &mut Bencher) {
    let a = Matrix::<f64>::zeros(100, 10);

    b.iter(|| {
    	a.transpose()
    })
}

#[bench]
fn allocating_transpose_100_100(b: &mut Bencher) {
    let a = Matrix::<f64>::zeros(100, 100);

    b.iter(|| {
    	a.transpose()
    })
}

#[bench]
fn allocating_transpose_1000_1000(b: &mut Bencher) {
    let a = Matrix::<f64>::zeros(1000, 1000);

    b.iter(|| {
    	a.transpose()
    })
}

#[bench]
fn allocating_transpose_10000_100(b: &mut Bencher) {
    let a = Matrix::<f64>::zeros(10000, 100);

    b.iter(|| {
    	a.transpose()
    })
}

// #[bench]
// fn allocating_transpose_10000_10000(b: &mut Bencher) {
//     let a = Matrix::<f64>::zeros(10000, 10000);

//     b.iter(|| {
//     	a.transpose()
//     })
// }

#[bench]
fn inplace_transpose_10_10(b: &mut Bencher) {
    let mut a = Matrix::<f64>::zeros(10, 10);

    b.iter(|| {
    	a.inplace_transpose();
    })
}

#[bench]
fn inplace_transpose_100_100(b: &mut Bencher) {
    let mut a = Matrix::<f64>::zeros(100, 100);

    b.iter(|| {
    	a.inplace_transpose();
    })
}

#[bench]
fn inplace_transpose_1000_1000(b: &mut Bencher) {
    let mut a = Matrix::<f64>::zeros(1000, 1000);

    b.iter(|| {
    	a.inplace_transpose();
    })
}

#[bench]
fn inplace_transpose_10000_100(b: &mut Bencher) {
    let mut a = Matrix::<f64>::zeros(10000, 100);

    b.iter(|| {
    	a.inplace_transpose();
    })
}

// #[bench]
// fn inplace_transpose_10000_10000(b: &mut Bencher) {
//     let mut a = Matrix::<f64>::zeros(10000, 10000);

//     b.iter(|| {
//     	a.inplace_transpose();
//     })
// }
