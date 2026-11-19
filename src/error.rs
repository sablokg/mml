use std::fmt;

/*
Gaurav Sablok
gsablok@proton.me
*/

#[derive(Debug)]
pub enum MicrobiomeError {
    InvalidInput(String),
    Model(String),
    Io(String),
}

pub type Result<T> = std::result::Result<T, MicrobiomeError>;

impl fmt::Display for MicrobiomeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(m) => write!(f, "invalid input: {m}"),
            Self::Model(m) => write!(f, "model error: {m}"),
            Self::Io(m) => write!(f, "io error: {m}"),
        }
    }
}

impl std::error::Error for MicrobiomeError {}

impl From<std::io::Error> for MicrobiomeError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

pub(crate) fn model_err<E: fmt::Display>(e: E) -> MicrobiomeError {
    MicrobiomeError::Model(e.to_string())
}
