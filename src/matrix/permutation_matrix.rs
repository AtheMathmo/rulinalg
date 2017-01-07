use std;

use matrix::{Matrix};
use error::Error;
use utils::Permutation;

use libnum::Num;

/// An efficient implementation of a permutation matrix.
///
/// A [permutation matrix](https://en.wikipedia.org/wiki/Permutation_matrix)
/// is a very special kind of matrix. It is essentially a matrix representation
/// of the more general concept of a permutation. That is, an `n` x `n` permutation
/// matrix corresponds to a permutation of ordered sets whose cardinality is `n`.
/// In particular, given an `m` x `n` matrix `A` and an `n` x `n` permutation
/// matrix `P`, the action of left-multiplying `A` by `P`, `PA`, corresponds
/// to permuting the rows of `A` by the given permutation represented by `P`.
/// Conversely, right-multiplication corresponds to column permutation.
/// More precisely, given another permutation matrix `K` of size `m` x `m`,
/// then `AK` is the corresponding permutation of the columns of `A`.
///
/// Due to their unique structure, permutation matrices can be much more
/// efficiently represented and applied than general matrices. Recall that
/// for general matrices `X` and `Y` of size `m` x `m` and `n` x `n` respectively,
/// the storage of `X` requires O(`m`<sup>2</sup>) memory and the storage of
/// `Y` requires O(`n`<sup>2</sup>) memory. Ignoring for the moment the existence
/// of Strassen's matrix multiplication algorithm and more theoretical alternatives,
/// the multiplication `XA` requires O(`m`<sup>2</sup>`n`) operations, and
/// the multiplication `AY` requires O(`m``n`<sup>2</sup>) operations.
///
/// By constrast, the storage of `P` requires only O(`m`) memory, and
/// the storage of `K` requires O(`n`) memory. Moreover, the products
/// `PA` and `AK` both require merely O(`mn`) operations.
///
/// # Representation
/// A permutation of an ordered set of cardinality *n* is a map of the form
///
/// ```text
/// p: { 1, ..., n } -> { 1, ..., n }.
/// ```
///
/// That is, for any index `i`, the permutation `p` sends `i` to some
/// index `j = p(i)`, and hence the map may be represented as an array of integers
/// of length *n*.
///
/// # Examples
/// ```
/// # #[macro_use] extern crate rulinalg; fn main() {
/// use rulinalg::matrix::PermutationMatrix;
///
/// let ref x = matrix![1, 2, 3;
///                     4, 5, 6;
///                     7, 8, 9];
///
/// // Swap the two first rows of x by left-multiplying a permutation matrix
/// let expected = matrix![4, 5, 6;
///                        1, 2, 3;
///                        7, 8, 9];
/// let mut p = PermutationMatrix::identity(3);
/// p.swap(0, 1);
/// assert_eq!(expected, p * x);
///
/// // Swap the two last columns of x by right-multiplying a permutation matrix
/// let expected = matrix![1, 3, 2;
///                        4, 6, 5;
///                        7, 9, 8];
/// let mut p = PermutationMatrix::identity(3);
/// p.swap(1, 2);
/// assert_eq!(expected, x * p);
///
/// // One can also construct the same permutation matrix directly
/// // from an array representation.
/// let ref p = PermutationMatrix::from_array(vec![0, 2, 1]).unwrap();
/// assert_eq!(expected, x * p);
///
/// // One may also obtain a full matrix representation of the permutation
/// assert_eq!(p.as_matrix(), matrix![1, 0, 0;
///                                   0, 0, 1;
///                                   0, 1, 0]);
///
/// // The inverse of a permutation matrix can efficiently be obtained
/// let p_inv = p.inverse();
/// // TODO: Implement product of permutation matrices to show that
/// // the p * p_inv is the identity
/// # }
/// ```
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct PermutationMatrix<T> {
    // A permutation matrix of dimensions NxN is represented as a permutation of the rows
    // of an NxM matrix for any M.
    perm: Permutation,

    // Currently, we need to let PermutationMatrix be generic over T,
    // because BaseMatrixMut is.
    marker: std::marker::PhantomData<T>
}

impl<T> PermutationMatrix<T> {
    /// The identity permutation.
    pub fn identity(n: usize) -> Self {
        PermutationMatrix {
            perm: Permutation::identity(n),
            marker: std::marker::PhantomData
        }
    }

    /// Swaps indices i and j
    pub fn swap(&mut self, i: usize, j: usize) {
        self.perm.swap(i, j);
    }

    /// The inverse of the permutation matrix.
    pub fn inverse(&self) -> PermutationMatrix<T> {
        PermutationMatrix {
            perm: self.perm.inverse(),
            marker: std::marker::PhantomData
        }
    }

    /// The dimensions of the permutation matrix.
    ///
    /// A permutation matrix is a square matrix, so `dim()` is equal
    /// to both the number of rows, as well as the number of columns.
    pub fn dim(&self) -> usize {
        self.perm.cardinality()
    }

    /// Constructs a `PermutationMatrix` from an array.
    ///
    /// # Errors
    /// The supplied N-length array must satisfy the following:
    ///
    /// - Each element must be in the half-open range [0, N).
    /// - Each element must be unique.
    pub fn from_array<A: Into<Vec<usize>>>(array: A) -> Result<PermutationMatrix<T>, Error> {
        let perm = Permutation::from_array(array)?;
        Ok(PermutationMatrix {
            perm: perm,
            marker: std::marker::PhantomData
        })
    }

    /// Constructs a `PermutationMatrix` from an array, without checking the validity of
    /// the supplied permutation.
    ///
    /// However, to ease development, a `debug_assert` with regards to the validity
    /// is still performed.
    pub fn from_array_unchecked<A: Into<Vec<usize>>>(array: A) -> PermutationMatrix<T> {
        let perm = Permutation::from_array_unchecked(array);
        PermutationMatrix {
            perm: perm,
            marker: std::marker::PhantomData
        }
    }

}

impl<T: Num> PermutationMatrix<T> {
    /// The permutation matrix in an equivalent full matrix representation.
    pub fn as_matrix(&self) -> Matrix<T> {
        Matrix::from_fn(self.dim(), self.dim(), |i, j|
            if self.perm.map_index(i) == j {
                T::one()
            } else {
                T::zero()
            }
        )
    }
}

impl<T> Into<Permutation> for PermutationMatrix<T> {
    fn into(self) -> Permutation {
        self.perm
    }
}

impl<'a, T> Into<&'a Permutation> for &'a PermutationMatrix<T> {
    fn into(self) -> &'a Permutation {
        &self.perm
    }
}

impl<T> From<Permutation> for PermutationMatrix<T> {
    fn from(perm: Permutation) -> Self {
        PermutationMatrix {
            perm: perm,
            marker: std::marker::PhantomData
        }
    }
}

#[cfg(test)]
mod tests {
    use matrix::Matrix;
    use quickcheck::TestResult;
    use super::PermutationMatrix;
    use utils::Permutation;

    quickcheck! {
        fn property_identity_is_permutation_identity(size: usize) -> TestResult {
            let p = PermutationMatrix::<u64>::identity(size);
            let expected = Permutation::identity(size);
            let p_as_permutation: Permutation = p.into();
            TestResult::from_bool(p_as_permutation == expected)
        }
    }

    #[test]
    fn swap() {
        let mut p = PermutationMatrix::<u64>::identity(4);
        p.swap(0, 3);
        p.swap(1, 3);

        let expected_permutation = Permutation::from_array(vec![3, 0, 2, 1]).unwrap();
        let p_as_permutation: Permutation = p.into();
        assert_eq!(p_as_permutation, expected_permutation);
    }

    #[test]
    fn as_matrix() {
        let permutation = Permutation::from_array(vec![2, 1, 0, 3]).unwrap();
        let p = PermutationMatrix::from(permutation);

        let expected_matrix: Matrix<u32> = matrix![0, 0, 1, 0;
                                                   0, 1, 0, 0;
                                                   1, 0, 0, 0;
                                                   0, 0, 0, 1];

        assert_eq!(expected_matrix, p.as_matrix());
    }

    #[test]
    fn from_array() {
        let array = vec![1, 0, 3, 2];
        let p = PermutationMatrix::<u32>::from_array(array.clone()).unwrap();
        let p_as_permutation: Permutation = p.into();
        let permutation = Permutation::from_array(array.clone()).unwrap();
        assert_eq!(p_as_permutation, permutation);
    }

    #[test]
    fn from_array_unchecked() {
        let array = vec![1, 0, 3, 2];
        let p = PermutationMatrix::<u32>::from_array_unchecked(array.clone());
        let p_as_permutation: Permutation = p.into();
        let permutation = Permutation::from_array_unchecked(array.clone());
        assert_eq!(p_as_permutation, permutation);
    }

    #[test]
    fn from_array_invalid() {
        assert!(PermutationMatrix::<u32>::from_array(vec![0, 1, 3]).is_err());
        assert!(PermutationMatrix::<u32>::from_array(vec![0, 0]).is_err());
        assert!(PermutationMatrix::<u32>::from_array(vec![3, 0, 1]).is_err());
    }
}
