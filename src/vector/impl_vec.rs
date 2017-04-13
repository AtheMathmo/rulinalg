use std::ops::{Mul, Add};
use libnum::{One, Zero, Float, FromPrimitive};
use std::fmt;

use utils;

use super::Vector;

impl<T> Vector<T> {
    /// Constructor for Vector struct.
    ///
    /// Requires the vector data.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::vector::Vector;
    ///
    /// let vec = Vector::new(vec![1.0,2.0,3.0,4.0]);
    /// ```
    pub fn new<U: Into<Vec<T>>>(data: U) -> Vector<T> {
        let our_data = data.into();
        let size = our_data.len();

        Vector {
            size: size,
            data: our_data,
        }
    }

    /// Constructor for Vector struct that takes a function `f`
    /// and constructs a new vector such that `V_i = f(i)`,
    /// where `i` is the index.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::Vector;
    ///
    /// let v = Vector::from_fn(4, |x| x * 3);
    /// assert_eq!(v, vector![0, 3, 6, 9]);
    /// # }
    /// ```
    pub fn from_fn<F>(size: usize, mut f: F) -> Vector<T>
        where F: FnMut(usize) -> T
    {

        let data: Vec<T> = (0..size).into_iter().map(|x| f(x)).collect();

        Vector {
            size: size,
            data: data,
        }
    }

    /// Consumes the Vector and returns the Vec of data.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T> Into<Vec<T>> for Vector<T> {
    fn into(self) -> Vec<T> {
        self.data
    }
}

impl<T: fmt::Display> fmt::Display for Vector<T> {
    /// Displays the Vector.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "["));
        for (i, datum) in self.data.iter().enumerate() {
            match f.precision() {
                Some(places) => {
                    try!(write!(f, " {:.*}", places, datum));
                }
                None => {
                    try!(write!(f, " {}", datum));
                }
            }
            if i < self.data.len() - 1 {
                try!(write!(f, ","));
            }
        }
        write!(f, "]")
    }
}

impl<T: Clone> Clone for Vector<T> {
    /// Clones the Vector.
    fn clone(&self) -> Vector<T> {
        Vector {
            size: self.size,
            data: self.data.clone(),
        }
    }
}

impl<T: Copy> Vector<T> {
    /// Applies a function to each element in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::Vector;
    /// fn add_two(a: f64) -> f64 {
    ///     a + 2f64
    /// }
    ///
    /// let a = vector![0.; 4];
    ///
    /// let b = a.apply(&add_two);
    ///
    /// assert_eq!(b, vector![2.0; 4]);
    /// # }
    /// ```
    pub fn apply(mut self, f: &Fn(T) -> T) -> Vector<T> {
        for val in &mut self.data {
            *val = f(*val);
        }
        self
    }
}

impl<T: Copy + PartialOrd> Vector<T> {
    /// Find the argmax of the Vector.
    ///
    /// Returns the index of the largest value in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::Vector;
    ///
    /// let a = vector![1.0,2.0,0.0,5.0];
    /// let b = a.argmax();
    /// assert_eq!(b.0, 3);
    /// assert_eq!(b.1, 5.0);
    /// # }
    /// ```
    pub fn argmax(&self) -> (usize, T) {
        utils::argmax(&self.data)
    }

    /// Find the argmin of the Vector.
    ///
    /// Returns the index of the smallest value in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::Vector;
    ///
    /// let a = vector![1.0, 2.0, 0.0, 5.0];
    /// let b = a.argmin();
    /// assert_eq!(b.0, 2);
    /// assert_eq!(b.1, 0.0);
    /// # }
    /// ```
    pub fn argmin(&self) -> (usize, T) {
        utils::argmin(&self.data)
    }
}

impl<T: Clone + Zero> Vector<T> {
    /// Constructs Vector of all zeros.
    ///
    /// Requires the size of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::vector::Vector;
    ///
    /// let vec = Vector::<f64>::zeros(10);
    /// ```
    pub fn zeros(size: usize) -> Vector<T> {
        Vector {
            size: size,
            data: vec![T::zero(); size],
        }
    }
}

impl<T: Clone + One> Vector<T> {
    /// Constructs Vector of all ones.
    ///
    /// Requires the size of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::vector::Vector;
    ///
    /// let vec = Vector::<f64>::ones(10);
    /// ```
    pub fn ones(size: usize) -> Vector<T> {
        Vector {
            size: size,
            data: vec![T::one(); size],
        }
    }
}

impl<T: Copy + Zero + Mul<T, Output = T> + Add<T, Output = T>> Vector<T> {
    /// Compute dot product with specified Vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::Vector;
    ///
    /// let a = vector![1.0, 2.0, 3.0, 4.0];
    /// let b = vector![2.0; 4];
    ///
    /// let c = a.dot(&b);
    /// assert_eq!(c, 20.0);
    /// # }
    /// ```
    pub fn dot(&self, v: &Vector<T>) -> T {
        utils::dot(&self.data, &v.data)
    }
}

impl<T: Copy + Zero + Add<T, Output = T>> Vector<T> {
    /// The sum of the vector.
    ///
    /// Returns the sum of all elements in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::Vector;
    ///
    /// let a = vector![1.0, 2.0, 3.0, 4.0];
    ///
    /// let c = a.sum();
    /// assert_eq!(c, 10.0);
    /// # }
    /// ```
    pub fn sum(&self) -> T {
        utils::unrolled_sum(&self.data[..])
    }
}

impl<T: Float + FromPrimitive> Vector<T> {
    /// The mean of the vector.
    ///
    /// Returns the arithmetic mean of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::Vector;
    ///
    /// let a = vector![1.0, 2.0, 3.0, 4.0];
    ///
    /// let c = a.mean();
    /// assert_eq!(c, 2.5);
    /// # }
    /// ```
    pub fn mean(&self) -> T {
        let sum = self.sum();
        sum / FromPrimitive::from_usize(self.size).unwrap()
    }

    /// The variance of the vector.
    ///
    /// Returns the unbiased sample variance of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate rulinalg; fn main() {
    /// use rulinalg::vector::Vector;
    ///
    /// let a = vector![1.0, 2.0, 3.0, 4.0];
    ///
    /// let c = a.variance();
    /// assert_eq!(c, 5.0 / 3.0);
    /// # }
    /// ```
    pub fn variance(&self) -> T {
        let m = self.mean();
        let mut var = T::zero();

        for u in &self.data {
            var = var + (*u - m) * (*u - m);
        }

        var / FromPrimitive::from_usize(self.size - 1).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{BaseVector, BaseVectorMut, Vector};
    use norm::Euclidean;

    #[test]
    fn test_display() {
        let v = vector![1, 2, 3, 4];
        assert_eq!(format!("{}", v), "[ 1, 2, 3, 4]");

        let v2 = vector![3.3, 4.0, 5.0, 6.0];
        assert_eq!(format!("{}", v2), "[ 3.3, 4, 5, 6]");
        assert_eq!(format!("{:.1}", v2), "[ 3.3, 4.0, 5.0, 6.0]");
    }

    #[test]
    fn test_equality() {
        let v = vector![1, 2, 3, 4];
        let v_redux = v.clone();
        assert_eq!(v, v_redux);
    }

    #[test]
    fn create_vector_new() {
        let a = vector![1.0; 12];

        assert_eq!(a.size(), 12);

        for i in 0..12 {
            assert_eq!(a[i], 1.0);
        }
    }

    #[test]
    fn create_vector_new_from_slice() {
        let data_vec: Vec<u32> = vec![1, 2, 3];
        let data_slice: &[u32] = &data_vec[..];
        let from_vec = Vector::new(data_vec.clone());
        let from_slice = Vector::new(data_slice);
        assert_eq!(from_vec, from_slice);
    }

    #[test]
    fn create_vector_from_fn() {
        let v1 = Vector::from_fn(3, |x| x + 1);
        assert_eq!(v1, vector![1, 2, 3]);

        let v2 = Vector::from_fn(3, |x| x as f64);
        assert_eq!(v2, vector![0., 1., 2.]);

        let mut z = 0;
        let v3 = Vector::from_fn(3, |x| {
            z += 1;
            x + z
        });
        assert_eq!(v3, vector![0 + 1, 1 + 2, 2 + 3]);

        let v4 = Vector::from_fn(3, move |x| x + 1);
        assert_eq!(v4, vector![1, 2, 3]);

        let v5 = Vector::from_fn(0, |x| x);
        assert_eq!(v5, Vector::new(vec![]));
    }

    #[test]
    fn create_vector_zeros() {
        let a = Vector::<f32>::zeros(7);

        assert_eq!(a.size(), 7);

        for i in 0..7 {
            assert_eq!(a[i], 0.0);
        }
    }

    #[test]
    fn vector_dot_product() {
        let a = vector![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vector![3.0; 6];

        let c = a.dot(&b);

        assert_eq!(c, 63.0);
    }

    #[test]
    fn vector_euclidean_norm() {
        let a = vector![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = a.norm(Euclidean);

        assert_eq!(b, (1. + 4. + 9. + 16. + 25. + 36. as f32).sqrt());
    }

    #[test]
    fn vector_iteration() {
        let our_vec = vec![2i32, 7, 1, 8, 2, 8];
        let our_vector = Vector::new(our_vec.clone());
        let our_vector_again = our_vector.clone();

        // over Vector (consuming)
        let mut our_recovered_vec = Vec::new();
        for i in our_vector {
            our_recovered_vec.push(i);
        }
        assert_eq!(our_recovered_vec, our_vec);

        // over &Vector
        let mut our_refcovered_vec = Vec::new();
        for i in &our_vector_again {
            our_refcovered_vec.push(*i);
        }
        assert_eq!(our_refcovered_vec, our_vec);
    }

    #[test]
    fn vector_from_iter() {
        let v1: Vector<usize> = (2..5).collect();
        let exp1 = vector![2, 3, 4];
        assert_eq!(v1, exp1);

        let orig: Vec<f64> = vec![2., 3., 4.];
        let v2: Vector<f64> = orig.iter().map(|x| x + 1.).collect();
        let exp2 = vector![3., 4., 5.];
        assert_eq!(v2, exp2);
    }

    #[test]
    fn vector_get_unchecked() {
        let v1 = vector![1, 2, 3];
        unsafe {
            assert_eq!(v1.get_unchecked(1), &2);
        }

        let mut v2 = vector![1, 2, 3];

        unsafe {
            let elem = v2.get_unchecked_mut(1);
            *elem = 4;
        }
        assert_eq!(v2, vector![1, 4, 3]);
    }

    #[test]
    fn vector_mul_f32_elemwise() {
        let a = vector![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vector![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let exp = vector![2.0, 6.0, 12.0, 20.0, 30.0, 42.0];

        // Allocating new memory
        let c = &a.elemul(&b);
        assert_eq!(c, &exp);

        // Allocating new memory
        let c = a.elemul(&b);
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_mul_int_elemwise() {
        let a = vector![1, 2, 3, 4];
        let b = vector![2, 4, 6, 8];

        let exp = vector![2, 8, 18, 32];

        // Allocating new memory
        let c = &a.elemul(&b);
        assert_eq!(c, &exp);

        // Allocating new memory
        let c = a.elemul(&b);
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_div_f32_elemwise() {
        let a = vector![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vector![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let exp = vector![1. / 2., 2. / 3., 3. / 4., 4. / 5., 5. / 6., 6. / 7.];

        // Allocating new memory
        let c = &a.elediv(&b);
        assert_eq!(c, &exp);

        // Allocating new memory
        let c = a.elediv(&b);
        assert_eq!(c, exp);
    }

    #[test]
    fn vector_div_int_elemwise() {
        let a = vector![2, 4, 6, 8];
        let b = vector![2, 2, 3, 3];

        let exp = vector![1, 2, 2, 2];

        // Allocating new memory
        let c = &a.elediv(&b);
        assert_eq!(c, &exp);

        // Allocating new memory
        let c = a.elediv(&b);
        assert_eq!(c, exp);
    }
}
