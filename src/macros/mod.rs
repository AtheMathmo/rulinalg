//! Macros for the linear algebra modules.

#[macro_use]
mod vector;

#[macro_use]
mod matrix;

#[macro_use]
mod matrix_eq;

mod comparison;

pub use self::comparison::{
    AbsoluteElementwiseComparator,
    ExactElementwiseComparator,
    UlpElementwiseComparator,
    FloatElementwiseComparator,

    // The following are just imported because we want to
    // expose trait bounds in the documentation
    ElementwiseComparator
};

pub use self::matrix_eq::{
    elementwise_matrix_comparison,
    elementwise_vector_comparison,
};
