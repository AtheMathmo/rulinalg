//! Everything related to triplet

/// A triplet, i.e., a struct that represents a row, a column and a value
pub trait Triplet<T> {
    /// Returns row
    fn row(&self) -> usize;
    /// Returns column
    fn col(&self) -> usize;
    /// Returns value
    fn value(&self) -> T;
}

impl<T: Copy> Triplet<T> for (usize, usize, T) {
    fn row(&self) -> usize {
        self.0
    }
    fn col(&self) -> usize {
        self.1
    }
    fn value(&self) -> T {
        self.2
    }
}
