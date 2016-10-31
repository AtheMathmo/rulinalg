//! The ULP module.
use std::mem;

/// TODO: Docs
#[derive(Debug, Copy, Clone)]
pub enum UlpComparisonResult
{
    /// Signifies an exact match between two floating point numbers.
    ExactMatch,
    /// The difference in ULP between two floating point numbers.
    Difference(u64),
    /// The two floating point numbers have different signs,
    /// and cannot be compared in a meaningful way.
    IncompatibleSigns,
    /// One or both of the two floating point numbers is a NaN,
    /// in which case the ULP comparison is not meaningful.
    NanPresent
}

/// Floating point types for which two instances can be compared for Unit in the Last Place (ULP) difference.
/// Implementing this trait enables the usage of the `ulp` comparator in `assert_matrix_eq!` for the given type.
///
/// The definition here leverages the fact that for two adjacent floating point numbers,
/// their integer representations are also adjacent.
///
/// A somewhat accessible (but not exhaustive) guide on the topic is available in the popular article
/// [Comparing Floating Point Numbers, 2012 Edition]
/// (https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/).
pub trait Ulp {
    /// Returns the difference between two floating point numbers, measured in ULP.
    fn ulp_diff(a: &Self, b: &Self) -> UlpComparisonResult;
}

macro_rules! impl_float_ulp {
    ($ftype:ty, $itype:ty) => {
        impl Ulp for $ftype {
            fn ulp_diff(a: &Self, b: &Self) -> UlpComparisonResult {
                if a == b {
                    UlpComparisonResult::ExactMatch
                } else if a.is_sign_positive() != b.is_sign_positive() {
                    UlpComparisonResult::IncompatibleSigns
                } else if a.is_nan() || b.is_nan() {
                    // Nor does it make much sense for NAN
                    UlpComparisonResult::NanPresent
                } else {
                    // Otherwise, we compute the ULP diff as the difference of the signed integer representations
                    let a_int = unsafe { mem::transmute::<$ftype, $itype>(a.to_owned()) };
                    let b_int = unsafe { mem::transmute::<$ftype, $itype>(b.to_owned()) };
                    UlpComparisonResult::Difference((b_int - a_int).abs() as u64)
                }
            }
        }
    }
}

impl_float_ulp!(f32, i32);
impl_float_ulp!(f64, i64);

// TODO: Tests for ULP
