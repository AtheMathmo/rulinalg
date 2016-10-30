# Change Log

This document will be used to keep track of changes made between release versions. I'll do my best to note any breaking changes!

# 0.3.5

### New Contributors

- [gcollura](https://github.com/gcollura)

### Breaking Changes

- None

### Features

- Added new `iter_diag` and `iter_diag_mut` functions to `BaseMatrix`
and `BaseMatrixMut` respectively.

### Bug Fixes

- The `matrix!` macro now works on empty matrices.

### Minor Changes

- Some refactoring of `decomposition` module.
- More lenient error handling on triangular solvers.
They no longer `assert!` that a matrix is triangular.
- All tests are now using `matrix!` macro and other
tidier constructors.

# 0.3.4

### New Contributors

- [andrewcsmith](https://github.com/andrewcsmith)
- [nwtnian](https://github.com/nwtnian)

### Breaking Changes

- Removed the `MachineEpsilon` trait. The same functionality
now exists in [num](https://github.com/rust-num/num).

### Features

- Implemented `From`/`Into` for traits for `Vec` and `Vector`.

### Bug Fixes

- `det()` now returns `0` instead of panicking if `Matrix` is singular.

### Minor Changes

- None

## 0.3.3

### New Contributors

- [Andlon](https://github.com/Andlon)
- [regexident](https://github.com/regexident)
- [tokahuke](https://github.com/tokahuke)

### Breaking Changes

- None

### Features

- SVD now returns singular values in descending order.
- Implemented a new `matrix!` macro for creating (small) matrices.
- Added a `from_fn` constructor for `Matrix`.
- Implementing `IndexMut` for `Vector`.
- Added `iter` and `iter_mut` for `Vector`.
- Implemented `IntoIter` for `Vector`.

### Bug Fixes

- Fixed bug with SVD convergence (would loop endlessly).
- Singular values from SVD are now non-negative.

### Minor Changes

- None

## 0.3.2

### New Contributors

- [eugene-bulkin](https://github.com/eugene-bulkin)

### Breaking Changes

- `Matrix::variance` now returns a `Result`.

### Features

- Added `swap_rows` and `swap_cols` function to `BaseMatrixMut`.

### Minor Changes

- Implemented `Display` for `Vector`.

## 0.3.1

### New Contributors

- [scholtzan](https://github.com/scholtzan)
- [theotherphil](https://github.com/theotherphil)

### Breaking Changes

- None

### Features

- None

### Minor Changes

- Improved documentation for `sum_rows` and `sum_cols` functions.
- Generalized signature of `select_rows` and `select_cols`. These functions now
take an `ExactSizeIterator` instead of a slice.

## 0.3.0

This is a large release which refactors most of the `matrix` module.
We modify the `BaseSlice` trait to encompass `Matrix` functionality too - hence
renaming it `BaseMatrix`. The motivation behind this is to allow us to be generic
over `Matrix`/`MatrixSlice`/`MatrixSliceMut`.

### Breaking Changes

- Refactor `BaseSlice` trait as `BaseMatrix`. Implement this trait for `Matrix` too.
- Much of the `Matrix` functionality is now implemented behind the `BaseMatrix` trait. 
It will need to be `use`d to access this functionality.

### Features

- Add a new `BaseMatrixMut` trait for `Matrix` and `MatrixSliceMut`.
- Many methods which were previously for `Matrix` only or for `MatrixSlice(Mut)` only now
work with both!

### Minor Changes

- Fixing a bug in the `sub_slice` method.
- Modifying some unsafe code to use equivalent iterators instead.
- More benchmarks for wider performance regression coverage.

## 0.2.2

### Breaking Changes

-None

### Features

- Vector and Matrix now derive the `Eq` trait.
- Vector and Matrix now derive the `Hash` trait.

### Minor Changes

- None

## 0.2.1

### New Contributors

- [brendan-rius](https://github.com/brendan-rius)
- [tafia](https://github.com/tafia)

### Breaking Changes

- None

### Features

- Adding new `get_row_*` methods for all `Matrix` types. Includes
mutable and unchecked `get` functions.

### Minor Changes

- None

## 0.2.0

### Breaking Changes

- Upper Hessenberg decomposition now consumes the input `Matrix` (instead of cloning at the start).

### Features

- Added Bidiagonal decomposition.
- Added Singular Value Decomposition.

### Minor Changes

- Fixed a bug where `get_unchecked_mut` returned `&T` instead of `&mut T`.

## 0.1.0

This release marks the separation of rulinalg from [rusty-machine](https://github.com/AtheMathmo/rusty-machine).

Rulinalg is now its own crate!