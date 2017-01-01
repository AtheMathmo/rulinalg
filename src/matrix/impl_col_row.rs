use std;

use super::{BaseMatrix, BaseMatrixMut, Column, ColumnMut, Row, RowMut};

impl<'a, T: 'a> Column<'a, T> {
    /// Returns the row as a slice.
    pub fn raw_slice(&self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.col.as_ptr(), self.col.rows()) }
    }
}

impl<'a, T: 'a> ColumnMut<'a, T> {
    /// Returns the row as a slice.
    pub fn raw_slice(&self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.col.as_ptr(), self.col.rows()) }
    }

    /// Returns the row as a slice.
    pub fn raw_slice_mut(&mut self) -> &'a mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.col.as_mut_ptr(), self.col.rows()) }
    }
}

impl<'a, T: 'a> Row<'a, T> {
    /// Returns the row as a slice.
    pub fn raw_slice(&self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.row.as_ptr(), self.row.cols()) }
    }
}

impl<'a, T: 'a> RowMut<'a, T> {
    /// Returns the row as a slice.
    pub fn raw_slice(&self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.row.as_ptr(), self.row.cols()) }
    }

    /// Returns the row as a slice.
    pub fn raw_slice_mut(&mut self) -> &'a mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.row.as_mut_ptr(), self.row.cols()) }
    }
}