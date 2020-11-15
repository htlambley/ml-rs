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
