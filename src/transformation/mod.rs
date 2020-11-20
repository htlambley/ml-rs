use crate::Error;
use ndarray::{Array2, ArrayView2};

pub mod pca;

/// This trait represents a *transformer*, which performs some operation on
/// the data, such as dimensionality reduction or scaling. A machine learning
/// pipeline may involve loading data to memory, applying some transformations
/// to simplify the data, then fitting an appropriate model.
///
/// # Details
/// Transformers in this library can be regarded as various maps on data
/// matrices in $\mathcal{X}^n$. A transformer $T$ is generally a function
/// $$T \colon \mathcal{X}^n \to \mathcal{Z}^n$$,
/// where $\mathcal{Z}$ is some other data space that we transform to.
///
/// A *dimensionality reduction* transformer is a transformation where
/// $\mathcal{X} = \mathbb{R}^m$, and $\mathcal{Z} = \mathbb{R}^p$, with
/// $p < m$. We could perform trivial dimensionality reduction by deleting
/// certain components of each sample vector, or we could perform a more
/// nuanced transformation. Among the most famous dimensionality reduction
/// procedures is *principal component analysis* as proposed by Pearson (1901).
///
/// # Prior Art
/// The idea of a common API for classification, regression and transformation
/// is used to great success in scikit-learn, a Python machine learning library.
///
/// # References
/// Pedregosa et al, *Scikit-learn: Machine Learning in Python*, J. Machine
/// Learning Research 12, 2011, pp. 2825–2830.
pub trait Transformer {
    fn fit(&mut self, x: ArrayView2<f64>) -> Result<(), Error>;
    fn transform(&self, x: ArrayView2<f64>) -> Result<Array2<f64>, Error>;
    fn fit_transform(&mut self, x: ArrayView2<f64>) -> Result<Array2<f64>, Error> {
        self.fit(x)?;
        self.transform(x)
    }
}
