//! # The rulinalg crate.
//! 
//! A crate that provides high-dimensional linear algebra
//! implemented entirely in Rust.
//!
//! ---
//!
//! This crate provides two core data structures: `Matrix` and
//! `Vector`. These structs are designed to behave as you would expect
//! with relevant operator overloading.
//!
//! The library currently contains (at least) the following linear algebra
//! methods:
//!
//! - Matrix inversion
//! - LUP decomposition
//! - QR decomposition
//! - SVD decomposition
//! - Cholesky decomposition
//! - Eigenvalue decomposition
//! - Upper Hessenberg decomposition
//! - Linear system solver
//! - Other standard transformations, e.g. Transposing, concatenation, etc.
//!
//! ---
//!
//! ## Usage
//!
//! Specific usage of modules is described within the modules themselves. This section
//! will highlight the basic usage.
//!
//! We can create new matrices.
//!
//! ```
//! use rulinalg::matrix::Matrix;
//!
//! // A new matrix with 3 rows and 2 columns.
//! let a = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//! ```
//!
//! The matrices are stored in row-major order. This means in the example above the top
//! row will be [1,2,3].
//! 
//! We can perform operations on matrices.
//!
//! ```
//! use rulinalg::matrix::Matrix;
//!
//! // A new matrix with 3 rows and 2 columns.
//! let a = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//! let b = Matrix::new(3, 2, vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
//!
//! // Produces a 3x2 matrix filled with sevens.
//! let c = a + b;
//! ```
//!
//! Of course the library can support more complex operations but you should check the individual
//! modules for more information.

#![deny(missing_docs)]
#![warn(missing_debug_implementations)]

extern crate num as libnum;
extern crate matrixmultiply;
#[cfg(target_pointer_width = "64")]
extern crate extprim;

pub mod matrix;
pub mod convert;
pub mod macros;
pub mod error;
pub mod utils;
pub mod vector;

/// Trait for linear algebra metrics.
///
/// Currently only implements basic euclidean norm.
pub trait Metric<T> {
    /// Computes the euclidean norm.
    fn norm(&self) -> T;
}
