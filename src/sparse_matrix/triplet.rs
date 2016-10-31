//! Everything related to triplet

use super::MatrixCoordinate;

/// A triplet, i.e., a struct that represents a row, a column and a value
pub trait Triplet<T> {
    /// Returns row coordinate
    fn row(&self) -> usize;
    /// Returns column coordinate
    fn col(&self) -> usize;
    /// Returns value
    fn value(&self) -> T;
    /// Returns a given coordinate of this triplet
    fn from_coordinate(&self, coo: MatrixCoordinate) -> usize;
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
    fn from_coordinate(&self, coo: MatrixCoordinate) -> usize {
        match coo {
            MatrixCoordinate::ROW => self.0,
            MatrixCoordinate::COL => self.1,
        }
    }
}
