//! Macros for the linear algebra modules.

#[macro_use]
mod matrix;

#[macro_use]
mod matrix_eq;

pub use self::matrix_eq::{elementwise_matrix_comparison, AbsoluteElementwiseComparator, ExactElementwiseComparator};
