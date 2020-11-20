use crate::Error;
use ndarray::{Array2, ArrayView2};

/// Contains the [`pca::PrincipalComponentAnalysis`] transformer which can perform
/// prinicpal component analysis (PCA) on a data matrix.
pub mod pca;

/// Represents a *transformer*, which performs some operation on
/// the data, such as dimensionality reduction or scaling. A machine learning
/// pipeline may involve loading data to memory, applying some transformations
/// to simplify the data, then fitting an appropriate model.
pub trait Transformer {
    fn fit(&mut self, x: ArrayView2<f64>) -> Result<(), Error>;
    fn transform(&self, x: ArrayView2<f64>) -> Result<Array2<f64>, Error>;
    fn fit_transform(&mut self, x: ArrayView2<f64>) -> Result<Array2<f64>, Error> {
        self.fit(x)?;
        self.transform(x)
    }
}
