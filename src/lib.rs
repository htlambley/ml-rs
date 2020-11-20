/// A variety of classification models. The `linear` module includes
/// linear models such as `LogisticRegression`, which performs logistic
/// regression on the data.
pub mod classification;

/// A collection of metrics to measure the performance of classification
/// and regression models.
///
/// After developing a model, these metrics allow you to evaluate the
/// performance and decide whether to refine, reject or accept the model.
pub mod metrics;
pub mod preprocessing;
pub mod regression;
pub mod transformation;

use thiserror::Error;

#[derive(Clone, Debug, Error)]
pub enum Error {
    #[error("attempted to use before calling `fit()`: try fitting with appropriate training data before usage")]
    UseBeforeFit,
    #[error("provided training data was invalid")]
    InvalidTrainingData,
    #[error("attempted to solve optimisation problem, but the optimiser encountered an error")]
    OptimiserError,
    #[error("an error occurred during the training process")]
    TrainingError,
}
