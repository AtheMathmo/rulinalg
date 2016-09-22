use libnum::Float;
use std::f32;
use std::f64;

/// Expose the machine epsilon of floating point numbers.
/// This trait should only need to exist for a short time,
/// until the Float trait from the Num crate has the same
/// capabilities.
pub trait MachineEpsilon: Float {
    /// Returns the machine epsilon for the given Float type.
    fn epsilon() -> Self;
}

impl MachineEpsilon for f32 {
    fn epsilon() -> f32 {
        f32::EPSILON
    }
}

impl MachineEpsilon for f64 {
    fn epsilon() -> f64 {
        f64::EPSILON
    }
}
