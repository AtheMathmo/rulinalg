use matrix::BaseMatrixMut;
use libnum::Zero;

pub fn nullify_lower_triangular_part<T, M>(matrix: &mut M)
    where T: Zero, M: BaseMatrixMut<T> {
    for (i, mut row) in matrix.row_iter_mut().enumerate() {
        for element in row.iter_mut().take(i) {
            *element = T::zero();
        }
    }
}

pub fn nullify_upper_triangular_part<T, M>(matrix: &mut M)
    where T: Zero, M: BaseMatrixMut<T> {
    for (i, mut row) in matrix.row_iter_mut().enumerate() {
        for element in row.iter_mut().skip(i + 1) {
            *element = T::zero();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::nullify_lower_triangular_part;
    use super::nullify_upper_triangular_part;

    #[test]
    fn nullify_lower_triangular_part_examples() {
        let mut x = matrix![1.0, 2.0, 3.0;
                            4.0, 5.0, 6.0;
                            7.0, 8.0, 9.0];
        nullify_lower_triangular_part(&mut x);
        assert_matrix_eq!(x, matrix![
            1.0, 2.0, 3.0;
            0.0, 5.0, 6.0;
            0.0, 0.0, 9.0
        ]);
    }

    #[test]
    fn nullify_upper_triangular_part_examples() {
        let mut x = matrix![1.0, 2.0, 3.0;
                            4.0, 5.0, 6.0;
                            7.0, 8.0, 9.0];
        nullify_upper_triangular_part(&mut x);
        assert_matrix_eq!(x, matrix![
            1.0, 0.0, 0.0;
            4.0, 5.0, 0.0;
            7.0, 8.0, 9.0
        ]);
    }
}
