//! Everything related to triplet

/// A triplet, i.e., a struct that represents a row, a column and a value
pub trait Triplet<T> {
    /// Returns row
    fn row(&self) -> usize;
    /// Returns col
    fn col(&self) -> usize;
    /// Returns value
    fn value(&self) -> T;
}
