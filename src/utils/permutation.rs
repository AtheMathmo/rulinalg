use error::{Error, ErrorKind};

/// An abstract permutation of an ordered set.
///
/// Given an ordered set X of cardinality N, `Permutation` is an efficient representation
/// of a permutation of this set. More concretely, if X is an array,
/// it maps each index to a new (unique) index in the permuted array.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Permutation {
    // Represents a permutation map
    // { 1, ..., N } -> { 1, ..., N }
    // For each index in the vector,
    perm: Vec<usize>
}

impl Permutation {
    /// The cardinality of the sets that the permutation can be applied to.
    ///
    /// If the permutation permutes sets of cardinality `N`, then `cardinality()` is equal to `N`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::utils::Permutation;
    ///
    /// let p = Permutation::identity(4);
    /// assert_eq!(p.cardinality(), 4);
    /// ```
    pub fn cardinality(&self) -> usize {
        self.perm.len()
    }

    /// Returns the identity permutation.
    ///
    /// That is, it returns the unique permutation such that
    /// when applied to an ordered set X yields X itself.
    pub fn identity(n: usize) -> Self {
        Permutation {
            perm: (0 .. n).collect()
        }
    }

    /// Swaps indices in the permutation.
    ///
    /// If P sends `i` to `a`, and `j` to `b`,
    /// then P sends `i` to `b` and `j` to `a` after this operation.
    pub fn swap(&mut self, i: usize, j: usize) {
        self.perm.swap(i, j);
    }

    /// Maps an index from the original set to the permuted set.
    ///
    /// If the permutation P sends `i` to `j`, then this function
    /// returns `j`.
    pub fn map_index(&self, i: usize) -> usize {
        self.perm[i]
    }

    /// Constructs a `Permutation` from an array.
    ///
    /// # Errors
    /// The supplied N-length array must satisfy the following:
    ///
    /// - Each element must be in the half-open range [0, N).
    /// - Each element must be unique.
    pub fn from_array<A: Into<Vec<usize>>>(array: A) -> Result<Permutation, Error> {
        let p = Permutation {
            perm: array.into()
        };
        validate_permutation(&p.perm).map(|_| p)
    }

    /// Constructs a `Permutation` from an array, without checking the validity of
    /// the supplied permutation.
    ///
    /// However, to ease development, a `debug_assert` with regards to the validity
    /// is still performed.
    ///
    /// Use of this function is generally discouraged unless the checked version
    /// has been proven to be a performance bottleneck.
    pub fn from_array_unchecked<A: Into<Vec<usize>>>(array: A) -> Permutation {
        let p = Permutation {
            perm: array.into()
        };
        debug_assert!(validate_permutation(&p.perm).is_ok(), "Permutation is not valid");
        p
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
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::utils::Permutation;
    ///
    /// let p = Permutation::from_array(vec![1, 0, 2]).unwrap();
    /// let mut x = vec![0, 1, 2];
    /// p.permute_by_swap(|i, j| x.swap(i, j));
    ///
    /// assert_eq!(x, vec![1, 0, 2]);
    /// ```
    pub fn permute_by_swap<S>(mut self, mut swap: S) where S: FnMut(usize, usize) -> () {
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
        for i in 0 .. self.perm.len() {
            let mut target = self.perm[i];
            while i != target {
                // When resolving a cycle, we resolve each index in the cycle
                // by repeatedly moving the current item into the target position,
                // and item in the target position into the current position.
                // By repeating this until we hit the start index,
                // we effectively resolve the entire cycle.
                let new_target = self.perm[target];
                swap(i, target);
                self.perm[target] = target;
                target = new_target;
            }
            self.perm[i] = i;
        }
    }

    /// The inverse of this permutation.
    ///
    /// More precisely, if the current permutation `perm` sends index i to j, then
    /// `perm.inverse()` sends j to i.
    pub fn inverse(&self) -> Permutation {
        let mut inv: Vec<usize> = vec![0; self.cardinality()];

        for (source, target) in self.perm.iter().cloned().enumerate() {
            inv[target] = source;
        }

        Permutation {
            perm: inv
        }
    }
}

impl From<Permutation> for Vec<usize> {
    fn from(p: Permutation) -> Vec<usize> {
        p.perm
    }
}

fn validate_permutation(perm: &[usize]) -> Result<(), Error> {
    use std::collections::HashSet;

    let ref n = perm.len();
    let all_in_bounds = perm.iter().all(|x| x < n);

    // Note: If we ever use itertools, this could be replaced with a one-liner.
    let all_unique = {
        let mut visited = HashSet::new();
        let mut unique = true;
        for i in perm {
            if visited.contains(&i) {
                unique = false;
                break;
            }
            visited.insert(i);
        }
        unique
    };

    if !all_in_bounds && !all_unique {
        Err(Error::new(ErrorKind::InvalidPermutation,
            "Supplied permutation array has both elements out of bounds and duplicate elements."))
    } else if !all_in_bounds {
        Err(Error::new(ErrorKind::InvalidPermutation,
            "Supplied permutation array has elements out of bounds."))
    } else if !all_unique {
        Err(Error::new(ErrorKind::InvalidPermutation,
            "Supplied permutation array duplicate elements."))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Permutation;

    #[test]
    fn permutation_permute_by_swap_on_empty_array() {
        let mut x = Vec::<char>::new();
        let p = Permutation::from_array(Vec::new()).unwrap();
        p.permute_by_swap(|i, j| x.swap(i, j));
    }

    #[test]
    fn permutation_permute_by_swap_on_arbitrary_array() {
        let mut x = vec!['a', 'b', 'c', 'd'];
        let p = Permutation::from_array(vec![0, 2, 3, 1]).unwrap();

        p.clone().permute_by_swap(|i, j| x.swap(i, j));
        assert_eq!(x, vec!['a', 'd', 'b', 'c']);
    }

    #[test]
    fn permutation_permute_by_swap_identity_on_arbitrary_array() {
        let mut x = vec!['a', 'b', 'c', 'd'];
        let p = Permutation::from_array(vec![0, 1, 2, 3]).unwrap();
        p.clone().permute_by_swap(|i, j| x.swap(i, j));
        assert_eq!(x, vec!['a', 'b', 'c', 'd']);
    }

    #[test]
    fn permutation_into_vec() {
        let index_map = vec![0, 2, 3, 1];
        let p = Permutation::from_array(index_map.clone()).unwrap();
        let p_as_vec: Vec<usize> = p.into();
        assert_eq!(index_map, p_as_vec);
    }

    #[test]
    fn permutation_cardinality() {
        let p = Permutation::from_array(vec![0, 2, 1]).unwrap();
        assert_eq!(p.cardinality(), 3);
    }

    #[test]
    fn permutation_mapped_index() {
        let p = Permutation::from_array(vec![0, 2, 1]).unwrap();
        assert_eq!(p.map_index(0), 0);
        assert_eq!(p.map_index(1), 2);
        assert_eq!(p.map_index(2), 1);
    }

    #[test]
    fn permutation_swap() {
        let mut p = Permutation::from_array(vec![0, 2, 1]).unwrap();

        p.swap(1, 2);
        assert_eq!(Vec::from(p.clone()), vec![0, 1, 2]);

        p.swap(0, 2);
        assert_eq!(Vec::from(p), vec![2, 1, 0]);
    }

    #[test]
    fn permutation_from_array_if_invalid() {
        assert!(Permutation::from_array(vec![1]).is_err());
        assert!(Permutation::from_array(vec![0, 0]).is_err());
        assert!(Permutation::from_array(vec![0, 1, 1]).is_err());
        assert!(Permutation::from_array(vec![0, 2]).is_err());
    }
}
