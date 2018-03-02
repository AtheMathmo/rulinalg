//! The vector module.
//!
//! Currently contains all code
//! relating to the vector linear algebra struct.

mod impl_ops;
mod impl_vec;

/// The Vector struct.
///
/// Can be instantiated with any type.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Vector<T> {
    size: usize,
    data: Vec<T>,
}

#[cfg(test)]
mod tests {
    #[cfg(all(feature = "serde", test))]
    fn serde_test() {
        use serde_test::{Token, assert_tokens};

        let vec = vector![1., 2., 3., 4.];

        assert_tokens(&vec, &[
            Token::Struct{name: "Vector", len:2},

            Token::Str("size"),
            Token::U64(4),

            Token::Str("data"),
            Token::Seq{len:Some(4)},
            Token::F64(1.),
            Token::F64(2.),
            Token::F64(3.),
            Token::F64(4.),
            Token::SeqEnd,

            Token::StructEnd
        ]);
    }
}
