use ndarray::{Array1, ArrayView1, ArrayView2};

pub mod linear;

#[derive(Clone, Debug)]
pub enum Error {
    RegressorNotFit,
    InvalidTrainingData,
    DidNotConverge,
}

pub trait Regressor {
    fn fit<'a>(&mut self, x: ArrayView2<'a, f64>, y: ArrayView1<'a, f64>) -> Result<(), Error>;
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, Error>;
}
