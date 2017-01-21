use std;

use matrix::{Matrix, BaseMatrix, BaseMatrixMut};
use vector::Vector;
use error::{Error, ErrorKind};

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

    /// The inverse of the permutation matrix.
    pub fn inverse(&self) -> PermutationMatrix<T> {
        let mut inv: Vec<usize> = vec![0; self.dim()];

        for (source, target) in self.perm.iter().cloned().enumerate() {
            inv[target] = source;
        }

        PermutationMatrix {
            perm: inv,
            marker: std::marker::PhantomData
        }
    }

    /// The dimensions of the permutation matrix.
    ///
    /// A permutation matrix is a square matrix, so `dim()` is equal
    /// to both the number of rows, as well as the number of columns.
    pub fn dim(&self) -> usize {
        self.perm.len()
    }

    /// Constructs a `PermutationMatrix` from an array.
    ///
    /// # Errors
    /// The supplied N-length array must satisfy the following:
    ///
    /// - Each element must be in the half-open range [0, N).
    /// - Each element must be unique.
    pub fn from_array<A: Into<Vec<usize>>>(array: A) -> Result<PermutationMatrix<T>, Error> {
        let p = PermutationMatrix {
            perm: array.into(),
            marker: std::marker::PhantomData
        };
        validate_permutation(&p.perm).map(|_| p)
    }

    /// Constructs a `PermutationMatrix` from an array, without checking the validity of
    /// the supplied permutation.
    ///
    /// However, to ease development, a `debug_assert` with regards to the validity
    /// is still performed.
    pub fn from_array_unchecked<A: Into<Vec<usize>>>(array: A) -> PermutationMatrix<T> {
        let p = PermutationMatrix {
            perm: array.into(),
            marker: std::marker::PhantomData
        };
        debug_assert!(validate_permutation(&p.perm).is_ok(), "Permutation is not valid");
        p
    }

    /// Maps the given row index into the resulting row index in the permuted matrix.
    pub fn map_row(&self, row_index: usize) -> usize {
        self.perm[row_index]
    }
}

impl<T: Num> PermutationMatrix<T> {
    /// The permutation matrix in an equivalent full matrix representation.
    pub fn as_matrix(&self) -> Matrix<T> {
        Matrix::from_fn(self.dim(), self.dim(), |i, j|
            if self.perm[i] == j {
                T::one()
            } else {
                T::zero()
            }
        )
    }
}

impl<T> PermutationMatrix<T> {
    /// TODO
    pub fn permute_rows_in_place<M>(mut self, matrix: &mut M) where M: BaseMatrixMut<T> {
        validate_permutation_left_mul_dimensions(&self, matrix);
        permute_by_swap(&mut self.perm, |i, j| matrix.swap_rows(i, j));
    }

    /// TODO
    pub fn permute_cols_in_place<M>(mut self, matrix: &mut M) where M: BaseMatrixMut<T> {
        validate_permutation_right_mul_dimensions(matrix, &self);
        permute_by_swap(&mut self.perm, |i, j| matrix.swap_cols(i, j));
    }

    /// TODO
    pub fn permute_vector_in_place(mut self, vector: &mut Vector<T>) {
        validate_permutation_vector_dimensions(&self, vector);
        permute_by_swap(&mut self.perm, |i, j| vector.mut_data().swap(i, j));
    }
}

impl<T: Clone> PermutationMatrix<T> {
    /// TODO
    pub fn permute_rows<X, Y>(&self, source_matrix: &X, target_matrix: &mut Y)
        where X: BaseMatrix<T>, Y: BaseMatrixMut<T> {
        assert!(source_matrix.rows() == target_matrix.rows()
                && source_matrix.cols() == target_matrix.cols(),
                "Source and target matrix must have equal dimensions.");
        validate_permutation_left_mul_dimensions(self, source_matrix);
        for (source_row, target_row_index) in source_matrix.row_iter()
                                                           .zip(self.perm.iter()
                                                                         .cloned()) {
            target_matrix.row_mut(target_row_index)
                         .raw_slice_mut()
                         .clone_from_slice(source_row.raw_slice());
        }
    }

    /// TODO
    pub fn permute_cols<X, Y>(&self, source_matrix: &X, target_matrix: &mut Y)
        where X: BaseMatrix<T>, Y: BaseMatrixMut<T> {
        assert!(source_matrix.rows() == target_matrix.rows()
                && source_matrix.cols() == target_matrix.cols(),
                "Source and target matrix must have equal dimensions.");
        validate_permutation_right_mul_dimensions(source_matrix, self);

        // Permute columns in one row at a time for (presumably) better cache performance
        for (row_index, source_row) in source_matrix.row_iter()
                                                           .map(|r| r.raw_slice())
                                                           .enumerate() {
            let target_row = target_matrix.row_mut(row_index).raw_slice_mut();
            for (source_element, target_col) in source_row.iter().zip(self.perm.iter().cloned()) {
                target_row[target_col] = source_element.clone();
            }
        }
    }

    /// TODO
    pub fn permute_vector(
        &self,
        source_vector: &Vector<T>,
        target_vector: &mut Vector<T>
    ) {
        assert!(source_vector.size() == target_vector.size(),
               "Source and target vector must have equal dimensions.");
        validate_permutation_vector_dimensions(self, source_vector);
        for (source_element, target_index) in source_vector.data()
                                                           .iter()
                                                           .zip(self.perm.iter().cloned()) {
            target_vector[target_index] = source_element.clone();
        }
    }

    /// TODO
    pub fn compose(
        &self,
        source_perm: &PermutationMatrix<T>,
        target_perm: &mut PermutationMatrix<T>
    ) {
        assert!(source_perm.dim() == target_perm.dim(),
            "Source and target permutation matrix must have equal dimensions.");
        validate_permutation_matrix_product_dimensions(source_perm, target_perm);
        let left = self;
        let right = source_perm;
        for i in 0 .. self.perm.len() {
            target_perm.perm[i] = left.perm[right.perm[i]];
        }
    }
}

impl<T> From<PermutationMatrix<T>> for Vec<usize> {
    fn from(p: PermutationMatrix<T>) -> Vec<usize> {
        p.perm
    }
}

impl<'a, T> Into<&'a [usize]> for &'a PermutationMatrix<T> {
    fn into(self) -> &'a [usize] {
        &self.perm
    }
}

fn validate_permutation_vector_dimensions<T>(p: &PermutationMatrix<T>, v: &Vector<T>) {
    assert!(p.dim() == v.size(),
            "Permutation matrix and Vector dimensions are not compatible.");
}


fn validate_permutation_left_mul_dimensions<T, M>(p: &PermutationMatrix<T>, rhs: &M)
    where M: BaseMatrix<T> {
     assert!(p.dim() == rhs.rows(),
            "Permutation matrix and right-hand side matrix dimensions
             are not compatible.");
}

fn validate_permutation_right_mul_dimensions<T, M>(lhs: &M, p: &PermutationMatrix<T>)
    where M: BaseMatrix<T> {
     assert!(lhs.cols() == p.dim(),
            "Left-hand side matrix and permutation matrix dimensions
             are not compatible.");
}

fn validate_permutation_matrix_product_dimensions<T>(
                    lhs: &PermutationMatrix<T>,
                    rhs: &PermutationMatrix<T>) {
    assert!(lhs.dim() == rhs.dim(),
        "Permutation matrices do not have compatible dimensions for multiplication.");
}

fn validate_permutation(perm: &[usize]) -> Result<(), Error> {
    // Recall that a permutation array of size n is valid if:
    // 1. All elements are in the range [0, n)
    // 2. All elements are unique

    let n = perm.len();

    // Here we use a vector of boolean. If memory usage or performance
    // is ever an issue, we could replace the vector with a bit vector
    // (from e.g. the bit-vec crate), which would cut memory usage
    // to 1/8, and likely improve performance due to more data
    // fitting in cache.
    let mut visited = vec![false; n];
    for p in perm.iter().cloned() {
        if p < n {
            visited[p] = true;
        }
        else {
            return Err(Error::new(ErrorKind::InvalidPermutation,
                "Supplied permutation array contains elements out of bounds."));
        }
    }
    let all_unique = visited.iter().all(|x| x.clone());
    if all_unique {
        Ok(())
    } else {
        Err(Error::new(ErrorKind::InvalidPermutation,
            "Supplied permutation array contains duplicate elements."))
    }
}

/// Applies the permutation by swapping elements in an abstract
/// container.
///
/// The permutation is applied by calls to `swap(i, j)` for indices
/// `i` and `j`.
///
/// # Complexity
///
/// - O(1) memory usage.
/// - O(n) worst case number of calls to `swap`.
fn permute_by_swap<S>(perm: &mut [usize], mut swap: S) where S: FnMut(usize, usize) -> () {
    // Please see https://en.wikipedia.org/wiki/Cyclic_permutation
    // for some explanation to the terminology used here.
    // Some useful resources I found on the internet:
    //
    // https://blog.merovius.de/2014/08/12/applying-permutation-in-constant.html
    // http://stackoverflow.com/questions/16501424/algorithm-to-apply-permutation-in-constant-memory-space
    //
    // A fundamental property of permutations on finite sets is that
    // any such permutation can be decomposed into a collection of
    // cycles on disjoint orbits.
    //
    // An observation is thus that given a permutation P,
    // we can trace out the cycle that includes index i
    // by starting at i and moving to P[i] recursively.
    for i in 0 .. perm.len() {
        let mut target = perm[i];
        while i != target {
            // When resolving a cycle, we resolve each index in the cycle
            // by repeatedly moving the current item into the target position,
            // and item in the target position into the current position.
            // By repeating this until we hit the start index,
            // we effectively resolve the entire cycle.
            let new_target = perm[target];
            swap(i, target);
            perm[target] = target;
            target = new_target;
        }
        perm[i] = i;
    }
}

#[cfg(test)]
mod tests {
    use matrix::Matrix;
    use vector::Vector;
    use super::PermutationMatrix;
    use super::{permute_by_swap, validate_permutation};

    #[test]
    fn swap() {
        let mut p = PermutationMatrix::<u64>::identity(4);
        p.swap(0, 3);
        p.swap(1, 3);

        let expected_permutation = PermutationMatrix::from_array(vec![3, 0, 2, 1]).unwrap();
        assert_eq!(p, expected_permutation);
    }

    #[test]
    fn as_matrix() {
        let p = PermutationMatrix::from_array(vec![2, 1, 0, 3]).unwrap();
        let expected_matrix: Matrix<u32> = matrix![0, 0, 1, 0;
                                                   0, 1, 0, 0;
                                                   1, 0, 0, 0;
                                                   0, 0, 0, 1];

        assert_matrix_eq!(expected_matrix, p.as_matrix());
    }

    #[test]
    fn from_array() {
        let array = vec![1, 0, 3, 2];
        let p = PermutationMatrix::<u32>::from_array(array.clone()).unwrap();
        let p_as_array: Vec<usize> = p.into();
        assert_eq!(p_as_array, array);
    }

    #[test]
    fn from_array_unchecked() {
        let array = vec![1, 0, 3, 2];
        let p = PermutationMatrix::<u32>::from_array_unchecked(array.clone());
        let p_as_array: Vec<usize> = p.into();
        assert_eq!(p_as_array, array);
    }

    #[test]
    fn from_array_invalid() {
        assert!(PermutationMatrix::<u32>::from_array(vec![0, 1, 3]).is_err());
        assert!(PermutationMatrix::<u32>::from_array(vec![0, 0]).is_err());
        assert!(PermutationMatrix::<u32>::from_array(vec![3, 0, 1]).is_err());
    }

    #[test]
    fn vec_from_permutation() {
        let source_vec = vec![0, 2, 1];
        let p = PermutationMatrix::<u32>::from_array(source_vec.clone()).unwrap();
        let vec = Vec::from(p);
        assert_eq!(&source_vec, &vec);
    }

    #[test]
    fn into_slice_ref() {
        let source_vec = vec![0, 2, 1];
        let ref p = PermutationMatrix::<u32>::from_array(source_vec.clone()).unwrap();
        let p_as_slice_ref: &[usize] = p.into();
        assert_eq!(source_vec.as_slice(), p_as_slice_ref);
    }

    #[test]
    fn map_row() {
        let p = PermutationMatrix::<u32>::from_array(vec![0, 2, 1]).unwrap();
        assert_eq!(p.map_row(0), 0);
        assert_eq!(p.map_row(1), 2);
        assert_eq!(p.map_row(2), 1);
    }

    #[test]
    fn inverse() {
        let p = PermutationMatrix::<u32>::from_array(vec![1, 2, 0]).unwrap();
        let expected_inverse = PermutationMatrix::<u32>::from_array(vec![2, 0, 1]).unwrap();
        assert_eq!(p.inverse(), expected_inverse);
    }

    #[test]
    fn permute_by_swap_on_empty_array() {
        let mut x = Vec::<char>::new();
        let mut permutation_array = Vec::new();
        permute_by_swap(&mut permutation_array, |i, j| x.swap(i, j));
    }

    #[test]
    fn permute_by_swap_on_arbitrary_array() {
        let mut x = vec!['a', 'b', 'c', 'd'];
        let mut permutation_array = vec![0, 2, 3, 1];

        permute_by_swap(&mut permutation_array, |i, j| x.swap(i, j));
        assert_eq!(x, vec!['a', 'd', 'b', 'c']);
    }

    #[test]
    fn permute_by_swap_identity_on_arbitrary_array() {
        let mut x = vec!['a', 'b', 'c', 'd'];
        let mut permutation_array = vec![0, 1, 2, 3];
        permute_by_swap(&mut permutation_array, |i, j| x.swap(i, j));
        assert_eq!(x, vec!['a', 'b', 'c', 'd']);
    }

    #[test]
    fn compose() {
        let p = PermutationMatrix::<u32>::from_array(vec![2, 1, 0]).unwrap();
        let q = PermutationMatrix::<u32>::from_array(vec![1, 0, 2]).unwrap();
        let pq_expected = PermutationMatrix::<u32>::from_array(vec![1, 2, 0]).unwrap();
        let qp_expected = PermutationMatrix::<u32>::from_array(vec![2, 0, 1]).unwrap();

        {
            let mut pq = PermutationMatrix::identity(3);
            p.compose(&q, &mut pq);
            assert_eq!(pq, pq_expected);
        }

        {
            let mut qp = PermutationMatrix::identity(3);
            q.compose(&p, &mut qp);
            assert_eq!(qp, qp_expected);
        }
    }

    #[test]
    fn compose_regression() {
        // At some point during development, this example failed to
        // give the expected results
        let p = PermutationMatrix::<u32>::from_array(vec![1, 2, 0]).unwrap();
        let q = PermutationMatrix::<u32>::from_array(vec![2, 0, 1]).unwrap();
        let pq_expected = PermutationMatrix::<u32>::from_array(vec![0, 1, 2]).unwrap();

        let mut pq = PermutationMatrix::identity(3);
        p.compose(&q, &mut pq);
        assert_eq!(pq, pq_expected);
    }

    #[test]
    fn permute_rows() {
        let x = matrix![ 0;
                         1;
                         2;
                         3];
        let p = PermutationMatrix::from_array(vec![2, 1, 3, 0]).unwrap();
        let mut output = Matrix::zeros(4, 1);
        p.permute_rows(&x, &mut output);
        assert_matrix_eq!(output, matrix![ 3; 1; 0; 2]);
    }

    #[test]
    fn permute_rows_in_place() {
        let mut x = matrix![ 0;
                         1;
                         2;
                         3];
        let p = PermutationMatrix::from_array(vec![2, 1, 3, 0]).unwrap();
        p.permute_rows_in_place(&mut x);
        assert_matrix_eq!(x, matrix![ 3; 1; 0; 2]);
    }

    #[test]
    fn permute_cols() {
        let x = matrix![ 0, 1, 2, 3];
        let p = PermutationMatrix::from_array(vec![2, 1, 3, 0]).unwrap();
        let mut output = Matrix::zeros(1, 4);
        p.permute_cols(&x, &mut output);
        assert_matrix_eq!(output, matrix![ 3, 1, 0, 2]);
    }

    #[test]
    fn permute_cols_in_place() {
        let mut x = matrix![ 0, 1, 2, 3];
        let p = PermutationMatrix::from_array(vec![2, 1, 3, 0]).unwrap();
        p.permute_cols_in_place(&mut x);
        assert_matrix_eq!(x, matrix![ 3, 1, 0, 2]);
    }

    #[test]
    fn permute_vector() {
        let x = vector![ 0, 1, 2, 3];
        let p = PermutationMatrix::from_array(vec![2, 1, 3, 0]).unwrap();
        let mut output = Vector::zeros(4);
        p.permute_vector(&x, &mut output);
        assert_vector_eq!(output, vector![ 3, 1, 0, 2]);
    }

    #[test]
    fn permute_vector_in_place() {
        let mut x = vector![ 0, 1, 2, 3];
        let p = PermutationMatrix::from_array(vec![2, 1, 3, 0]).unwrap();
        p.permute_vector_in_place(&mut x);
        assert_vector_eq!(x, vector![ 3, 1, 0, 2]);
    }

    use quickcheck::{Arbitrary, Gen};

    // In order to write property tests for the validation of a permutation,
    // we need to be able to generate arbitrary (valid) permutations.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ValidPermutationArray(pub Vec<usize>);

    impl Arbitrary for ValidPermutationArray {
        fn arbitrary<G: Gen>(g: &mut G) -> Self {
            let upper_bound = g.size();
            let mut array = (0 .. upper_bound).collect::<Vec<usize>>();
            g.shuffle(&mut array);
            ValidPermutationArray(array)
        }
    }

    // We also want to be able to generate invalid permutations for
    // the same reasons
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct InvalidPermutationArray(pub Vec<usize>);

    impl Arbitrary for InvalidPermutationArray {
        fn arbitrary<G: Gen>(g: &mut G) -> Self {
            // Take an arbitrary valid permutation and mutate it so that
            // it is invalid
            let mut permutation_array = ValidPermutationArray::arbitrary(g).0;
            let n = permutation_array.len();

            // There are two essential sources of invalidity:
            // 1. Duplicate elements
            // 2. Indices out of bounds
            // We want to have either or both

            let should_have_duplicates = g.gen::<bool>();
            let should_have_out_of_bounds = !should_have_duplicates || g.gen::<bool>();
            assert!(should_have_duplicates || should_have_out_of_bounds);

            if should_have_out_of_bounds {
                let num_out_of_bounds_rounds = g.gen_range::<usize>(1, n);
                for _ in 0 .. num_out_of_bounds_rounds {
                    let interior_index = g.gen_range::<usize>(0, n);
                    let exterior_index = n + g.gen::<usize>();
                    permutation_array[interior_index] = exterior_index;
                }
            }

            if should_have_duplicates {
                let num_duplicates = g.gen_range::<usize>(1, n);
                for _ in 0 .. num_duplicates {
                    let interior_index = g.gen_range::<usize>(0, n);
                    let duplicate_value = permutation_array[interior_index];
                    permutation_array.push(duplicate_value);
                }
            }

            // The duplicates are placed at the end, so we perform
            // an additional shuffle to end up with a more or less
            // arbitrary invalid permutation
            g.shuffle(&mut permutation_array);
            InvalidPermutationArray(permutation_array)
        }
    }

    impl<T: Send + Clone + 'static> Arbitrary for PermutationMatrix<T> {
        fn arbitrary<G: Gen>(g: &mut G) -> Self {
            let ValidPermutationArray(array) = ValidPermutationArray::arbitrary(g);
            PermutationMatrix::from_array(array)
                .expect("The generated permutation array should always be valid.")
        }
    }

    quickcheck! {
        fn property_validate_permutation_is_ok_for_valid_input(array: ValidPermutationArray) -> bool {
            validate_permutation(&array.0).is_ok()
        }
    }

    quickcheck! {
        fn property_validate_permutation_is_err_for_invalid_input(array: InvalidPermutationArray) -> bool {
            validate_permutation(&array.0).is_err()
        }
    }

    quickcheck! {
        fn property_identity_has_identity_array(size: usize) -> bool {
            let p = PermutationMatrix::<u64>::identity(size);
            let p_as_array: Vec<usize> = p.into();
            let expected = (0 .. size).collect::<Vec<usize>>();
            p_as_array == expected
        }
    }

    quickcheck! {
        fn property_dim_is_equal_to_array_dimensions(array: ValidPermutationArray) -> bool {
            let ValidPermutationArray(array) = array;
            let n = array.len();
            let p = PermutationMatrix::<u32>::from_array(array).unwrap();
            p.dim() == n
        }
    }

    quickcheck! {
        fn property_inverse_of_inverse_is_original(p: PermutationMatrix<u32>) -> bool {
            p == p.inverse().inverse()
        }
    }

    quickcheck! {
        fn property_inverse_composes_to_identity(p: PermutationMatrix<u32>) -> bool {
            // Recall that P * P_inv = I and P_inv * P = I
            let n = p.dim();
            let pinv = p.inverse();
            let mut p_pinv_composition = PermutationMatrix::identity(n);
            let mut pinv_p_composition = PermutationMatrix::identity(n);
            p.compose(&pinv, &mut p_pinv_composition);
            pinv.compose(&p, &mut pinv_p_composition);
            assert_eq!(p_pinv_composition, PermutationMatrix::identity(n));
            assert_eq!(pinv_p_composition, PermutationMatrix::identity(n));
            true
        }
    }
}
