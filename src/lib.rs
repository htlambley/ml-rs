//! # ml-rs
//! This crate implements various machine learning algorithms in Rust,
//! built on the [`ndarray`](https://github.com/rust-ndarray/ndarray)
//! crate.
//!
//! Currently, there is support for classification and regression models,
//! transformation (including principal component analysis) and metrics
//! to evaluate model performance.
//!
//! # Setup
//! This crate uses BLAS/LAPACK to accelerate the linear algebra operations
//! needed. This crate is built on `ndarray-linalg`, which supports the
//! following backends:
//! - [Netlib](https://www.netlib.org/)
//! - [OpenBLAS](https://www.openblas.net/)
//! - [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html).
//!
//! You can choose which library you wish to link to. Roughly speaking, you
//! should expect Netlib to be slower than the other two choices, and if you
//! are deploying on Intel processors, MKL is likely to perform best (though
//! the library is not open-source).
//!
//! On Debian systems, a working configuration can be obtained by installing
//! the packages `libopenblas-base`, `libopenblas-dev` and `liblapacke-dev`,
//! and adding
//! ```toml
//! openblas-src = { version = "0.7", features = ["system"] }
//! ```
//! to your `Cargo.toml`, which will link to the system OpenBLAS.
//! # Quick Start Guide
//! - Load data from CSV to a 2-dimensional array:
//! ```no_run
//! use ml_rs::preprocessing::CsvReader;
//! use ndarray::Array2;
//! use std::fs::File;
//!
//! let csv_file = File::open("data.csv").unwrap();
//! let reader = CsvReader::new(&csv_file);
//! // Pass the number of rows and columns expected
//! let n_columns = 5;
//! let n_rows = 1000;
//! let data: Array2<f64> = reader.read(n_rows, n_columns);
//! ```
//! - Separate into data matrix and target vector:
//! ```
//! # use ndarray::array;
//! # let data = array![[-1.0, 1.0, 1.0, -1.0, 3.0], [2.0, 4.0, 5.0, 1.0, 6.0]];
//! use ndarray::{Axis, s};
//! // Choose the last column of the `data` array to be the target
//! let feature_col_index = 4;
//! let x = data.slice(s![.., 0..feature_col_index]);
//! let y = data.index_axis(Axis(1), feature_col_index);
//! ```
//! - Perform regression:
//! ```no_run
//! # use ndarray::array;
//! # let _x = array![[-1.0, 1.0, 1.0, -1.0], [2.0, 4.0, 5.0, 1.0]];
//! # let _y = array![3.0, 6.0];
//! # let x = _x.view();
//! # let y = _y.view();
//! use ml_rs::regression::linear::LinearRegression;
//! use ml_rs::regression::Regressor;
//! let mut lm = LinearRegression::new();
//! // Fit a linear regression model to the data, unwrapping to check for errors
//! lm.fit(x, y).unwrap();
//! // Get the predicted values for `x` given by the regression model
//! let y_pred = lm.predict(x);
//! ```
//! - Measure the performance of a model:
//! ```
//! # use ndarray::array;
//! # let _y = array![3.0, 6.0];
//! # let y = _y.view();
//! # let y_pred = array![3.0, 6.0];
//! use ml_rs::metrics::accuracy_score;
//! // We own `y_pred`, so we need to return a view, which means we
//! // don't consume it when calculating accuracy.
//! let train_accuracy = accuracy_score(y, y_pred.view());
//! println!("Training set accuracy: {}%", train_accuracy * 100.0);
//! ```
//! Classification works very similarly to regression: for an example,
//! see the [`classification`] module.

#[warn(missing_docs)]

/// A variety of supervised classification models to use with numeric data.
///
/// Classification tasks aim to construct a model
/// $h \colon \mathcal{X} \to \\{0, 1, \dots, n - 1\\}$ to distinguish between
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
/// An overview of model selection can be found in \[1\]. Bibliographic
/// references to the models provided by the library are provided where
/// appropriate in the documentation of the respective classifier.
///
/// ## Statistical Learning Theory
/// This section can freely be omitted, but provides interesting mathematical
/// formalism which explains why the procedures we use are justified.
///
/// We begin with a data space $\mathcal{X}$, and the corresponding *target
/// space* $\mathcal{Y} = \\{0, 1, \dots, n - 1\\}$. We assume that the data
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
/// risk*, given some training data $T = \\{ (X_1, Y_1), \dots, (X_p, Y_p) \\}$:
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
/// $$ \argmin_{h \in \mathcal{H}} R_\mathrm{E}(h; T). $$
///
/// The above is a standard characterisation of statistical learning theory.
/// A much broader book on the topic is \[2\].
///
/// # Models
/// Currently, this library supports the following models.
/// ## Trivial Models
/// - [`classification::TrivialClassifier`]
/// - [`classification::MajorityClassifier`].   
/// ## Logistic Regression (in `linear`)
/// These models currently only support binary classification. They are
/// appropriate where a linear function of the features would be a good
/// predictor of the probability of lying in the positive class.
///
/// - [`classification::linear::LogisticRegression`].
///
/// Multiple solvers are provided, as can be viewed on the main documentation
/// page; it is advisable to try all the options to see which perform best.
///
/// # Examples
/// For examples, see the classifiers above, which are provided with
/// usage examples.
///
/// # References
/// \[1\] Hastie et al, *The Elements of Statistical Learning: Data Mining,
/// Inference and Prediction*, Springer, New York, NY, 2001, 1st ed,
/// ch. 7.
///
/// \[2\] Vapnik, *The Nature of Statistical Learning Theory*, Springer, New
/// York, NY, 1999, 1st ed.
pub mod classification;

/// A collection of metrics to measure the performance of classification
/// and regression models.
///
/// After developing a model, these metrics allow you to evaluate the
/// performance and decide whether to refine, reject or accept the model.
pub mod metrics;
/// Utilities including loading data from CSV files to arrays to input into
/// models.
pub mod preprocessing;
/// A collection of supervised regression models to predict continuous
/// variables from data.
///
/// Regression tasks look to construct a regressor
/// $h \colon \mathcal{X} \to \mathcal{Y}$, where instead of $\mathcal{Y}$
/// being a finite set as in classification, we have $\mathcal{Y}$ being a
/// continuum, e.g. an interval $[a, b]$.
///
/// # Context
/// The context is largely the same as in the classification case (and reading
/// the [`classification`] module documentation should prove helpful). The
/// main change is the the space $\mathcal{Y}$ which is now continuous, so some
/// classification models do not have corresponding regression models, whereas
/// others (such as least squares regression) find applications in both
/// classification and regression.
///
/// # Models
/// Currently, the following regressors are supported.
/// ## Linear Regression
/// These models are appropriate when the target $Y$ is believed to be some
/// linear function $f(X)$ of the feature vector $X$.
/// - [`regression::linear::LinearRegression`].
pub mod regression;
/// Procedures to perform scaling, dimensionality reduction and other
/// transformations on the data before input into a model.
///
/// # Details
/// Transformers in this library can be regarded as various maps on data
/// matrices in $\mathcal{X}^n$. A transformer $T$ is generally a function
/// $$T \colon \mathcal{X}^n \to \mathcal{Z}^n,$$
/// where $\mathcal{Z}$ is some other data space that we transform to.
///
/// A *dimensionality reduction* transformer is a transformation where
/// $\mathcal{X} = \mathbb{R}^m$, and $\mathcal{Z} = \mathbb{R}^p$, with
/// $p < m$. We could perform trivial dimensionality reduction by deleting
/// certain components of each sample vector, or we could perform a more
/// nuanced transformation. Among the most famous dimensionality reduction
/// procedures is *principal component analysis* as proposed by Pearson (1901).
/// This is implemented in ml-rs as [`transformation::pca::PrincipalComponentAnalysis`].
///
/// # Prior Art
/// The idea of a common API for classification, regression and transformation
/// is used to great success in scikit-learn, a Python machine learning library.
///
/// # References
/// Pedregosa et al, *Scikit-learn: Machine Learning in Python*, J. Machine
/// Learning Research 12, 2011, pp. 2825–2830.
pub mod transformation;

use thiserror::Error;

/// The main error type which represents an error in a model or transformer.
/// As there are many commonalities between classifiers, regressors and
/// transformers, this general error is returned during the use of any of
/// these objects.
#[derive(Clone, Debug, Error)]
pub enum Error {
    /// A model or transformer that required fitting was not fit before trying
    /// to use (e.g. calling `predict()` before `fit()`).
    #[error("attempted to use before calling `fit()`: try fitting with appropriate training data before usage")]
    UseBeforeFit,
    /// The training data provided was invalid in some sense. Check the
    /// assumptions for the model used. Common problems are:
    /// - the data matrix `x` and the label array `y` are different lengths
    /// - the data matrix `x` or label array `y` are empty.
    #[error("provided training data was invalid")]
    InvalidTrainingData,
    /// The model being used requires an optimisation problem to be solved,
    /// but when passing the problem to `argmin`, an error occurred.
    /// This is most likely an internal error that is not caused by the
    /// user input: file an issue if this occurs without obvious cause.
    #[error("attempted to solve optimisation problem, but the optimiser encountered an error")]
    OptimiserError,
    /// The model had an error during the fitting process. Common problems are:
    /// - the fitting process involved a call to LAPACK which failed.
    ///
    /// This may be an internal error, or a problem on your system.
    #[error("an error occurred during the fitting process")]
    FittingError,
}
