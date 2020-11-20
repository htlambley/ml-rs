use crate::Error;
use ndarray::{Array1, ArrayView1, ArrayView2};

pub mod linear;

/// This trait represents a *regression model*, used to predict a continuous
/// value fit from some training data. Contrast this with the `Classifier`
/// trait in the `classification` module, which assigns a discrete class
/// (in other words, predicts the class) of some data after being fit
/// upon some training data.
///
/// Regression tasks are highly varied, and a particularly common approach
/// is linear regression (sometimes utilising regularisation on the
/// weights). It is also common to attempt to fit non-linear functions
/// such as polynomials to the data where this is believed to be a more
/// appropriate model.
pub trait Regressor {
    fn fit<'a>(&mut self, x: ArrayView2<'a, f64>, y: ArrayView1<'a, f64>) -> Result<(), Error>;
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, Error>;
}
