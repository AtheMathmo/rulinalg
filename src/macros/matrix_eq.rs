use matrix::BaseMatrix;

use libnum::{Num};

use std::fmt;

#[doc(hidden)]
#[derive(Debug, Copy, Clone)]
pub struct ElementComparisonError<T> {
    pub x: T,
    pub y: T,
    pub error: T,
    pub row: usize,
    pub col: usize
}

impl<T> fmt::Display for ElementComparisonError<T> where T: fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f,
            "({i}, {j}): x = {x}, y = {y}. Error: {error}",
            i = self.row,
            j = self.col,
            x = self.x,
            y = self.y,
            error = self.error)
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub enum MatrixComparisonResult<T, C> where T: Copy, C: ElementwiseComparator<T> {
    Match,
    MismatchedDimensions { dim_x: (usize, usize), dim_y: (usize, usize) },
    MismatchedElements { comparator: C, mismatches: Vec<ElementComparisonError<T>> }
}

#[doc(hidden)]
pub trait ElementwiseComparator<T> where T: Copy {
    fn compare(&self, x: T, y: T) -> Option<T>;
    fn description(&self) -> String;
    fn definition(&self) -> String;
}

#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct AbsoluteElementwiseComparator<T> {
    pub tol: T
}

impl<T, C> MatrixComparisonResult<T, C> where T: Copy + fmt::Display, C: ElementwiseComparator<T> {
    pub fn panic_message(&self) -> Option<String> {
        match self {
            &MatrixComparisonResult::MismatchedElements { ref comparator, ref mismatches } => {
                // TODO: Aligned output
                let mut formatted_mismatches = String::new();
                for mismatch in mismatches {
                    formatted_mismatches.push_str(" ");
                    formatted_mismatches.push_str(&mismatch.to_string());
                    formatted_mismatches.push_str("\n");
                }
                Some(format!("\n
Matrices X and Y have {num} mismatched element pairs. The mismatched elements are listed below, in the format
(row, col): x = x[[row, col]], y = y[[row, col]].

{mismatches}
Comparison criterion: {description}, defined by
    {definition}.
\n",
                    num = mismatches.len(),
                    description = comparator.description(),
                    definition = comparator.definition(),
                    mismatches = formatted_mismatches))
            },
            &MatrixComparisonResult::MismatchedDimensions { dim_x, dim_y } => {
                Some(format!("\n
Dimensions of matrices X and Y do not match.
 dim(X) = {x_rows} x {x_cols}
 dim(Y) = {y_rows} x {y_cols}
\n",
                    x_rows = dim_x.0, x_cols = dim_x.1,
                    y_rows = dim_y.0, y_cols = dim_y.1))
            },
            _ => None
        }
    }
}

#[doc(hidden)]
pub fn elementwise_matrix_comparison<T, M, C>(x: &M, y: &M, comparator: C) -> MatrixComparisonResult<T, C>
    where M: BaseMatrix<T>, T: Copy, C: ElementwiseComparator<T> {
    if x.rows() == y.rows() && x.cols() == y.cols() {
        let mismatches = {
            let mut mismatches = Vec::new();
            let x = x.as_slice();
            let y = y.as_slice();
            for i in 0 .. x.rows() {
                for j in 0 .. x.cols() {
                    let a = x[[i, j]].to_owned();
                    let b = y[[i, j]].to_owned();
                    if let Some(error) = comparator.compare(a, b) {
                        mismatches.push(ElementComparisonError {
                            x: a,
                            y: b,
                            error: error,
                            row: i,
                            col: j
                        });
                    }
                }
            }
            mismatches
        };
        
        if mismatches.is_empty() {
            MatrixComparisonResult::Match
        } else {
            MatrixComparisonResult::MismatchedElements { comparator: comparator, mismatches: mismatches }
        }
    } else {
        MatrixComparisonResult::MismatchedDimensions { dim_x: (x.rows(), x.cols()), dim_y: (y.rows(), y.cols()) }
    }
}

impl<T> ElementwiseComparator<T> for AbsoluteElementwiseComparator<T>
    where T: Copy + fmt::Display + Num + PartialOrd<T> {

    fn compare(&self, a: T, b: T) -> Option<T> {
        // Note: Cannot use num::abs because we do not want to restrict
        // ourselves to Signed types (i.e. we still want to be able to
        // handle unsigned types).
        if a == b {
            None
        } else {
            let distance = if a > b { a - b } else { b - a };
            if distance <= self.tol {
                None
            } else {
                Some(distance)
            }
        }
    }

    fn description(&self) -> String {
        format!("absolute difference")
    }

    fn definition(&self) -> String {
        format!("|x - y| <= {tol}", tol = self.tol)
    }
}

/// Compare matrices for approximate equality.
/// # Examples
///
/// ```
/// #[macro_use]
/// extern crate rulinalg;
///
/// # fn main() {
/// let a = matrix![1.000, 2.000,
///                 3.000, 4.000];
/// let b = matrix![0.999, 2.001,
///                 2.998, 4.000 ];
/// assert_matrix_eq!(a, b, comp = abs, tol = 0.01);
/// # }
/// ```
#[macro_export]
macro_rules! assert_matrix_eq {
    ($x:expr, $y:expr, comp = abs, tol = $tol:expr) => {
        {
            use $crate::macros::{elementwise_matrix_comparison, AbsoluteElementwiseComparator};
            let msg = elementwise_matrix_comparison(&$x, &$y, AbsoluteElementwiseComparator { tol: $tol }).panic_message();
            if let Some(msg) = msg {
                // Note: We need the panic to incur here inside of the macro in order
                // for the line number to be correct when using it for tests,
                // hence we build the panic message in code, but panic here.
                panic!(msg);
            }
        }
    };
}

#[cfg(test)]
mod tests {
    #[test]
    pub fn matrix_eq_absolute_compare_self_for_integer() {
        let x = matrix![1, 2, 3;
                        4, 5, 6];
        assert_matrix_eq!(x, x, comp = abs, tol = 0);
    }

    #[test]
    pub fn matrix_eq_absolute_compare_self_for_floating_point() {
        let x = matrix![1.0, 2.0, 3.0;
                        4.0, 5.0, 6.0];
        assert_matrix_eq!(x, x, comp = abs, tol = 1e-10);
    }

    #[test]
    #[should_panic]
    pub fn matrix_eq_absolute_mismatched_dimensions() {
        let x = matrix![1, 2, 3;
                        4, 5, 6];
        let y = matrix![1, 2;
                        3, 4];
        assert_matrix_eq!(x, y, comp = abs, tol = 0);
    }

    #[test]
    #[should_panic]
    pub fn matrix_eq_absolute_mismatched_floating_point_elements() {
        let x = matrix![1.00,  2.00,  3.00;
                        4.00,  5.00,  6.00];
        let y = matrix![1.00,  2.01,  3.00;
                        3.99,  5.00,  6.00];
        assert_matrix_eq!(x, y, comp = abs, tol = 1e-10);
    }


    // TODO: test for differently sized matrices
}
