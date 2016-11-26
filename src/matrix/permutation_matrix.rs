// use matrix::BaseMatrixMut;
// use std::ops::Mul;
// use std::any::Any;
use std;

/// TODO
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct PermutationMatrix<T> {
    // An N x N permutation matrix P can be seen as a map
    // { 1, ..., N } -> { 1, ..., N }
    // Hence, we merely store N indices, such that
    // perm[[i]] = j
    // means that index i is mapped to index j
    perm: Vec<usize>,

    // Currently, we need to let PermutationMatrix be generic over T,
    // because BaseMatrixMut is.
    marker: std::marker::PhantomData<T>
}

impl<T> PermutationMatrix<T> {
    /// The identity permutation.
    pub fn identity(n: usize) -> Self {
        PermutationMatrix {
            perm: (0 .. n).collect(),
            marker: std::marker::PhantomData
        }
    }

    /// Swaps indices i and j
    pub fn swap(&mut self, i: usize, j: usize) {
        self.perm.swap(i, j);
    }

    // pub fn inverse(...)
}
