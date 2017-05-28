use rulinalg::matrix::Matrix;
use rulinalg::matrix::BaseMatrix;
use test::Bencher;
use test::black_box;

macro_rules! bench_transpose {
    ($name:ident, $rows:expr, $cols:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let a = Matrix::new($rows, $cols, vec![2.0; $rows * $cols]);

            b.iter(|| {
                let _ = black_box(a.transpose());
            });
        }
    }
}

bench_transpose!(bench_transpose_10_50, 10, 50);
bench_transpose!(bench_transpose_50_10, 50, 10);
bench_transpose!(bench_transpose_10_500, 10, 500);
bench_transpose!(bench_transpose_500_10, 500, 10);
bench_transpose!(bench_transpose_100_5000, 100, 5000);
bench_transpose!(bench_transpose_5000_100, 5000, 100);
