use std;

use matrix::{Matrix};
use utils::Permutation;

use libnum::Num;

/// TODO
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
}
