#![feature(test)]

#[macro_use]
extern crate rulinalg;
extern crate num as libnum;
extern crate test;
extern crate rand;

pub mod linalg {
	mod cholesky;
	mod iter;
	mod lu;
	mod matrix;
	mod norm;
	mod permutation;
	mod qr;
	mod svd;
	mod transpose;
	mod triangular;
	pub mod util;
}
