use ulp;
use ulp::Ulp;

use libnum::{Num, Float};

use std::fmt;

/// Trait that describes elementwise comparators for [assert_matrix_eq!](../macro.assert_matrix_eq!.html).
///
/// Usually you should not need to interface with this trait directly. It is a part of the documentation
/// only so that the trait bounds for the comparators are made public.
pub trait ElementwiseComparator<T, E> where T: Copy, E: ComparisonFailure {
    /// Compares two elements.
    ///
    /// Returns the error associated with the comparison if it failed.
    fn compare(&self, x: T, y: T) -> Result<(), E>;

    /// A description of the comparator.
    fn description(&self) -> String;
}

#[doc(hidden)]
pub trait ComparisonFailure {
    fn failure_reason(&self) -> Option<String>;
}

#[doc(hidden)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AbsoluteError<T>(pub T);

/// The `abs` comparator used with [assert_matrix_eq!](../macro.assert_matrix_eq!.html).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AbsoluteElementwiseComparator<T> {
    /// The maximum absolute difference tolerated (inclusive).
    pub tol: T
}

impl<T> ComparisonFailure for AbsoluteError<T> where T: fmt::Display {
    fn failure_reason(&self) -> Option<String> {
        Some(
            format!("Absolute error: {error}.", error = self.0)
        )
    }
}

impl<T> ElementwiseComparator<T, AbsoluteError<T>> for AbsoluteElementwiseComparator<T>
    where T: Copy + fmt::Display + Num + PartialOrd<T> {

    fn compare(&self, a: T, b: T) -> Result<(), AbsoluteError<T>> {
        assert!(self.tol >= T::zero());

        // Note: Cannot use num::abs because we do not want to restrict
        // ourselves to Signed types (i.e. we still want to be able to
        // handle unsigned types).

        if a == b {
            Ok(())
        } else {
            let distance = if a > b { a - b } else { b - a };
            if distance <= self.tol {
                Ok(())
            } else {
                Err(AbsoluteError(distance))
            }
        }
    }

    fn description(&self) -> String {
        format!("absolute difference, |x - y| <= {tol}.", tol = self.tol)
    }
}

/// The `exact` comparator used with [assert_matrix_eq!](../macro.assert_matrix_eq!.html).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ExactElementwiseComparator;

#[doc(hidden)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ExactError;

impl ComparisonFailure for ExactError {
    fn failure_reason(&self) -> Option<String> { None }
}

impl<T> ElementwiseComparator<T, ExactError> for ExactElementwiseComparator
    where T: Copy + fmt::Display + PartialEq<T> {

    fn compare(&self, a: T, b: T) -> Result<(), ExactError> {
        if a == b {
            Ok(())
        } else {
            Err(ExactError)
        }
    }

    fn description(&self) -> String {
        format!("exact equality x == y.")
    }
}

/// The `ulp` comparator used with [assert_matrix_eq!](../macro.assert_matrix_eq!.html).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct UlpElementwiseComparator {
    /// The maximum difference in ULP units tolerated (inclusive).
    pub tol: u64
}

#[doc(hidden)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct UlpError(pub ulp::UlpComparisonResult);

impl ComparisonFailure for UlpError {
    fn failure_reason(&self) -> Option<String> {
        use ulp::UlpComparisonResult;
        match self.0 {
            UlpComparisonResult::Difference(diff) =>
                Some(format!("Difference: {diff} ULP.", diff=diff)),
            UlpComparisonResult::IncompatibleSigns =>
                Some(format!("Numbers have incompatible signs.")),
            _ => None
        }
    }
}

impl<T> ElementwiseComparator<T, UlpError> for UlpElementwiseComparator
    where T: Copy + Ulp {

    fn compare(&self, a: T, b: T) -> Result<(), UlpError> {
        let diff = Ulp::ulp_diff(&a, &b);
        match diff {
            ulp::UlpComparisonResult::ExactMatch => Ok(()),
            ulp::UlpComparisonResult::Difference(diff) if diff <= self.tol => Ok(()),
            _ => Err(UlpError(diff))
        }
    }

    fn description(&self) -> String {
        format!("ULP difference less than or equal to {tol}. See documentation for details.",
                tol = self.tol)
    }
}

/// The `float` comparator used with [assert_matrix_eq!](../macro.assert_matrix_eq!.html).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FloatElementwiseComparator<T> {
    abs: AbsoluteElementwiseComparator<T>,
    ulp: UlpElementwiseComparator
}

#[doc(hidden)]
#[allow(dead_code)]
impl<T> FloatElementwiseComparator<T> where T: Float + Ulp {
    pub fn default() -> Self {
        FloatElementwiseComparator {
            abs: AbsoluteElementwiseComparator { tol: T::epsilon() },
            ulp: UlpElementwiseComparator { tol: 4 }
        }
    }

    pub fn eps(self, eps: T) -> Self {
        FloatElementwiseComparator {
            abs: AbsoluteElementwiseComparator { tol: eps },
            ulp: self.ulp
        }
    }

    pub fn ulp(self, max_ulp: u64) -> Self {
        FloatElementwiseComparator {
            abs: self.abs,
            ulp: UlpElementwiseComparator { tol: max_ulp }
        }
    }
}

impl<T> ElementwiseComparator<T, UlpError> for FloatElementwiseComparator<T>
    where T: Copy + Ulp + Float + fmt::Display {
    fn compare(&self, a: T, b: T) -> Result<(), UlpError> {
        // First perform an absolute comparison with a presumably very small epsilon tolerance
        if let Err(_) = self.abs.compare(a, b) {
            // Then fall back to an ULP-based comparison
            self.ulp.compare(a, b)
        } else {
            // If the epsilon comparison succeeds, we have a match
             Ok(())
        }
    }

    fn description(&self) -> String {
        format!("
Epsilon-sized absolute comparison, followed by an ULP-based comparison.
Please see the documentation for details.
Epsilon:       {eps}
ULP tolerance: {ulp}",
            eps = self.abs.tol,
            ulp = self.ulp.tol)
    }
}