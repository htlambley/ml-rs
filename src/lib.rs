#[warn(missing_docs)]

/// This module contains a variety of classification models to use with numeric
/// data. Classification tasks aim to construct a model 
/// $h \colon \mathcal{X} \to \{0, 1, \dots, n - 1\}$ to distinguish between
/// $n$ classes of data from the data space $\mathcal{X}$, which is typically
/// $\mathbb{R}^m$. Classification is a *supervised learning* task which 
/// requires some pre-labelled training data sampled independently from
/// the data distribution.
///
/// # Context
/// When designing a classifier, we start with some training data 
/// $(X_1, Y_1), \dots, (X_p, Y_p)$, and choose a *model*, which determines the
/// collection $\mathcal{H}$ of classifiers that we want to choose from. 
/// Generally, we then proceed by trying to find the classifier in 
/// $\mathcal{H}$ that minimises the error over the training data, using some
/// suitable algorithm. We then evaluate the performance of the model on some
/// new data sampled from the data distribution in order to estimate the 
/// generalisation error. 
///
/// This library supports the procedure by providing several models, listed
/// below, which can be fit on data, and some tools in the `metrics` module
/// to evaluate the performance of models on new data. The steps to take are:
/// - Load the dataset into memory as a *data matrix* $X$ and an array of 
///   *labels* $y$.
/// - Choose a suitable model and fit it (see the `Classifier` trait) on 
///   *training data*.
/// - Use a scoring function from `metrics` such as the accuracy score to
///   evaluate the performance on some *test data* that is distinct from
///   the training data.
///
/// An overview of model selection can be found in [1]. Bibliographic 
/// references to the models provided by the library are provided where
/// appropriate in the documentation of the respective classifier.
///
/// ## Statistical Learning Theory
/// This section can freely be omitted, but provides interesting mathematical 
/// formalism which explains why the procedures we use are justified.
///
/// We begin with a data space $\mathcal{X}$, and the corresponding *target 
/// space* $\mathcal{Y} = \{0, 1, \dots, n - 1\}$. We assume that the data
/// pairs $(x, y) \in \mathcal{X} \times \mathcal{Y}$ emerge frome some
/// probability disribution $\mathcal{P}$ on $\mathcal{X} \times \mathcal{Y}$,
/// and that the training data are *independent and identically distributed*
/// (i.i.d.) samples from $\mathcal{P}$. The goal is to learn the label of 
/// any sample $x \in \mathcal{X}$: in other words, we would like to know
/// the conditional probability $\mathbb{P} [ Y = y \mid X \in A]$ for any
/// subset $A$ of $\mathcal{X}$. We would expect that if the training data
/// are i.i.d., then we should be able to make a good estimation of the 
/// conditional probability if we have sufficient data.
///
/// We choose a function to measure the *risk*, $R(h)$, that a given classifier
/// $h$ makes an error. This is taken over the entire distribution with respect
/// to some *loss function*, so if $(X, Y)$ are sampled from $\mathcal{P}$, 
/// then
/// $$ R(h) = \mathbb{E} [ L(h(X), Y) ]. $$
/// We can estimate the risk over the entire distribution by the *empirical 
/// risk*, given some training data $T = \{ (X_1, Y_1), \dots, (X_p, Y_p) \}$:
/// $$R_\mathrm{E}(h; T) = \frac1p \sum_{i = 1}^p L(h(X_i), Y_i).$$
/// Provided that the training data are indeed i.i.d., the expected value of 
/// the empirical risk is the (generalisation) risk $R(h)$, so the empirical
/// risk serves as an estimate of generalisation error. If the data are not
/// i.i.d. then the empirical risk may not be a good estimate of the 
/// true risk, leading to poor performance on unseen data.
///
/// Choosing a model amounts to selecting a *hypothesis class* $\mathcal{H}$,
/// which is a collection of functions which we consider as candidates. The
/// *empirical risk minimisation* problem is to find the classifier in 
/// $\mathcal{H}$ that best fits the data:
/// $$ \text{ERM problem: } \argmin_{h \in \mathcal{H}} R_\mathrm{E}(h; T). $$
/// 
/// The above is a standard characterisation of statistical learning theory.
/// A much broader book on the topic is [2].
///
/// # Models
/// Currently, this library supports the following models.
/// ## Trivial Models
/// - [`TrivialClassifier`](struct.TrivialClassifier.html)
/// - [`MajorityClassifier`](struct.MajorityClassifier.html)
/// ## Logistic Regression (in [`linear`](linear/))
/// These models currently only support binary classification. They are
/// appropriate where a linear function of the features would be a good
/// predictor of the probability of lying in the positive class.
///
/// - [`LogisticRegression`](linear/struct.LogisticRegression.html)
/// - [`IRLSLogisticRegression`](linear/struct.IRLSLogisticRegression.html).
/// These classifiers differ only in the algorithm used to fit the model, and
/// depending on configuration, one may be significantly faster than the other
/// during the fitting process.
///
/// # Examples
/// For examples, see the classifiers above, which are provided with 
/// usage examples.
///
/// # References
/// [1] Hastie et al, *The Elements of Statistical Learning: Data Mining,
/// Inference and Prediction*, Springer, New York, NY, 2001, 1st ed, 
/// ch. 7.
///
/// [2] Vapnik, *The Nature of Statistical Learning Theory*, Springer, New
/// York, NY, 1999, 1st ed. 
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
    #[error("an error occurred during the fitting process")]
    FittingError,
}
