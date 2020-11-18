use super::{labels_binary, Classifier, Error, ProbabilityBinaryClassifier};
use argmin::core;
use argmin::core::{ArgminOp, Executor};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::BFGS;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

/// A classifier implementing the logistic regression model. Logistic
/// regression models can be used for binary classification problems and may
/// be extended through multinomial logistic regression.
///
/// # Model
/// This model is appropriate if you have a collection of samples `x`
/// with labels `y` equal to `0` and `1`. In other words, this classifier
/// is suitable for **binary classification** tasks.
///
/// Roughly speaking, the logistic regression model tries to fit a linear
/// model to predict the probability of each class. As we require probability
/// estimates to be in [0, 1], the linear model is used to predict the
/// *log-odds* instead.
///
/// The model is fit using *maximum likelihood estimation* to find the optimal
/// weights. There is no closed form to find the maximum likelihood estimator
/// weights so this is solved numerically using the BFGS optimiser from
/// `argmin`, which is a quasi-Newton optimisation method using second-order
/// derivatives.
///
/// # Configuration
/// `max_iter` — sets the maximum permitted number of iterations used in
/// gradient descent to find weights. Larger values will typically lead to
/// better convergence but takes longer. The default is 100.
///
/// # Examples
/// Fitting a logistic regression classifier and making a prediction.
/// ```no_run
/// use ndarray::array;
/// use ml_rs::classification::Classifier;
/// use ml_rs::classification::linear::LogisticRegression;
///
/// let x = array![[1.0, 2.0, 3.0], [1.0, 4.0, 3.0]];
/// let y = array![0, 1];
///
/// let mut clf = LogisticRegression::new();
/// clf.fit(x.view(), y.view()).unwrap();
///
/// let x_test = array![[1.0, 2.0, 3.0]];
/// assert_eq!(clf.predict(x_test.view()).unwrap(), array![0]);
/// ```
#[derive(Clone)]
pub struct LogisticRegression {
    weights: Option<Array1<f64>>,
    pub max_iter: u64,
}

struct LogisticRegressionProblem<'a, 'b> {
    train_x: ArrayView2<'a, f64>,
    train_y: ArrayView1<'b, f64>,
}

/// A numerical approximation of f(x) = log(1 + exp(x)). This suffers from
/// numerical errors for moderately large x (e.g. x > 700) as e^700 approaches
/// the maximum representable f64 value.
///
/// We can try to address this problem using approximations in the limiting
/// case. The limit f(x) as x tends to infinity is x, and the limit as it
/// tends to negative infinity is e^x.
///
/// The issue is explored in greater detail at
/// http://sachinashanbhag.blogspot.com/2014/05/numerically-approximation-of-log-1-expy.html
/// with suggested cutoff values.
fn approx_log_exp(x: f64) -> f64 {
    if x > 40. {
        x
    } else if x < -10. {
        x.exp()
    } else {
        (1. + x.exp()).ln()
    }
}

impl<'a, 'b> ArgminOp for LogisticRegressionProblem<'a, 'b> {
    type Param = Array1<f64>;
    type Float = f64;
    type Output = f64;
    type Hessian = Array2<f64>;
    type Jacobian = ();

    fn apply(&self, w: &Self::Param) -> Result<Self::Output, core::Error> {
        // The log-likelihood is given by the expression
        // $$ \ell(w) = \langle y, Xw \rangle - \sum \log(1 + \exp((Xw)_i)). $$
        // We wish to maximise the log-likelihood, i.e. minimise the negative
        // log-likelihood.

        let xw = self.train_x.dot(w);
        let mut log_exp_xw = xw.clone();
        log_exp_xw.par_mapv_inplace(approx_log_exp);
        let log_likelihood = self.train_y.dot(&xw) - log_exp_xw.sum();
        Ok(-log_likelihood)
    }

    fn gradient(&self, w: &Self::Param) -> Result<Self::Param, core::Error> {
        // The gradient of the log-likelihood is given by the expression
        // $$ \nabla \ell(w) = X^T(y - \sigma(Xw)). $$
        // We require the gradient of the negative log-likelihood so we
        // negate at the end.

        let mut xw = self.train_x.dot(w);
        xw.par_mapv_inplace(sigmoid);
        let inner = -xw + self.train_y;
        let gradient = self.train_x.t().dot(&inner);
        Ok(-gradient)
    }
}

impl LogisticRegression {
    pub fn new() -> LogisticRegression {
        LogisticRegression {
            weights: None,
            max_iter: 100,
        }
    }
}

impl Default for LogisticRegression {
    fn default() -> LogisticRegression {
        Self::new()
    }
}

impl Classifier for LogisticRegression {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<usize>) -> Result<(), Error> {
        if x.nrows() != y.len() || !labels_binary(y) {
            return Err(Error::InvalidTrainingData);
        }

        // Map y to an Array<f64> for convenience when defining the log-
        // likelihood and gradient functions, and define the optimisation
        // problem.
        let train_y = y.mapv(|x| x as f64);
        let best_param: Result<Array1<f64>, Error> = {
            let cost = LogisticRegressionProblem {
                train_x: x,
                train_y: train_y.view(),
            };

            let param_size = x.ncols();
            // Set up a gradient descent using More–Thuente line search.
            let line_search = MoreThuenteLineSearch::new();
            let init_hessian = Array2::eye(param_size);
            let solver = BFGS::new(init_hessian, line_search);
            // Generate a random initial point in [-1, 1]^n and hope this is
            // reasonable.
            let x_0 = Array1::random(param_size, Uniform::new(-1.0, 1.0));
            let executor = Executor::new(cost, solver, x_0).max_iters(self.max_iter);
            // Execute the optimiser and save the best weights found.
            executor
                .run()
                .map(|result| {
                    let state = result.state;
                    state.best_param.to_owned()
                })
                .map_err(|_| Error::DidNotConverge)
        };

        self.weights = Some(best_param?);
        Ok(())
    }

    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<usize>, Error> {
        let probabilties = self.predict_probability(x)?;
        Ok(probabilties.iter().map(|x| (x.round() as usize)).collect())
    }
}

impl ProbabilityBinaryClassifier for LogisticRegression {
    fn predict_probability(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, Error> {
        // TODO: return an iterator so the consumer can map if necessary
        if let Some(weights) = &self.weights {
            // Estimate the probability for each sample, and return 1 if p > 0.5, 0 otherwise.
            let mut xw = x.dot(weights);
            xw.par_mapv_inplace(sigmoid);
            Ok(xw)
        } else {
            Err(Error::ClassifierNotFit)
        }
    }
}

/// The sigmoid function, also called the logistic function, given by
/// f(x) = 1 / (1 + exp(-x)).
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod test {
    use super::super::{Classifier, Error};
    use super::LogisticRegression;
    use ndarray::{array, Array1, Array2};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_unfit_logistic_regression() {
        let clf = LogisticRegression::new();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        match clf.predict(x.view()) {
            Err(Error::ClassifierNotFit) => (),
            _ => panic!("Classifier did not return correct error"),
        }
    }

    #[test]
    fn test_logistic_regression_different_sizes() {
        let mut clf = LogisticRegression::new();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0];
        match clf.fit(x.view(), y.view()) {
            Err(Error::InvalidTrainingData) => (),
            _ => panic!("Classifier did not return correct error"),
        }
    }

    #[test]
    fn test_fit_logistic_regression() {
        let mut clf = LogisticRegression::new();
        let x = array![[1.0, 2.0], [1.0, 3.0], [3.0, 4.0], [3.0, 5.0]];
        let y = array![0, 0, 1, 1];
        clf.fit(x.view(), y.view()).unwrap();
        assert_eq!(array![0, 0, 1, 1], clf.predict(x.view()).unwrap());
    }

    #[test]
    fn test_logistic_regression_non_binary_labels() {
        let mut clf = LogisticRegression::new();
        let x = array![[1.0, 2.0], [1.0, 3.0], [3.0, 4.0], [3.0, 5.0]];
        let y = array![0, 0, 1, 2];
        match clf.fit(x.view(), y.view()) {
            Err(Error::InvalidTrainingData) => (),
            _ => panic!("Classifier did not return correct error"),
        }
    }

    #[test]
    fn test_single_logistic_regression() {
        let mut clf = LogisticRegression::new();
        let x = array![[1.0, 2.0, 3.0],];
        let y = array![0];
        clf.fit(x.view(), y.view()).unwrap();
        assert_eq!(array![0], clf.predict(x.view()).unwrap());
    }

    #[test]
    fn test_fit_logistic_regression_random() {
        let mut clf = LogisticRegression::new();
        let n_rows = 2000;
        let n_features = 5;
        let x = Array2::random((n_rows, n_features), Uniform::new(-1.0, 1.0));
        let y = Array1::random(n_rows, Uniform::new(0, 2));
        clf.fit(x.view(), y.view()).unwrap();
    }

    #[test]
    fn test_fit_logistic_regression_random_large() {
        let mut clf = LogisticRegression::new();
        let n_rows = 100000;
        let n_features = 5;
        let x = Array2::random((n_rows, n_features), Uniform::new(-1.0, 1.0));
        let y = Array1::random(n_rows, Uniform::new(0, 2));
        clf.fit(x.view(), y.view()).unwrap();
    }
}
