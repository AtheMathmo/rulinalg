//! Everything related to coordinate

/// Represents a coordinate (row and column) with repective value
pub trait Coordinate<T> {
    /// Returns row
    fn row(&self) -> usize;
    /// Returns col
    fn col(&self) -> usize;
    /// Returns value
    fn value(&self) -> T;
}
