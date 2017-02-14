#![feature(test)]

#[macro_use]
extern crate rulinalg;
extern crate num as libnum;
extern crate test;
extern crate rand;

mod linalg {
	mod iter;
	mod matrix;
	mod svd;
	mod norm;
	mod triangular;
}
