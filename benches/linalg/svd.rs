use test::Bencher;
use rand;
use rand::{Rng, SeedableRng};
use rulinalg::matrix::Matrix;

fn reproducible_random_matrix(rows: usize, cols: usize) -> Matrix<f64> {
    const STANDARD_SEED: [usize; 4] = [12, 2049, 4000, 33];
    let mut rng = rand::StdRng::from_seed(&STANDARD_SEED);
    let elements: Vec<_> = rng.gen_iter::<f64>().take(rows * cols).collect();
    Matrix::new(rows, cols, elements)
}

#[bench]
fn svd_10_10(b: &mut Bencher) {
    let mat = reproducible_random_matrix(10, 10);

    b.iter(|| mat.clone().svd())
}

#[bench]
fn svd_100_100(b: &mut Bencher) {
    let mat = reproducible_random_matrix(100, 100);

    b.iter(|| mat.clone().svd())
}
