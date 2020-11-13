use ndarray::{Array1, Axis, ArrayView1, ArrayView2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use argmin::prelude::*;
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use super::Classifier;

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
/// weights so this is solved numerically using gradient descent. 
///
/// # Configuration
/// `max_iter` — sets the maximum permitted number of iterations used in 
/// gradient descent to find weights. Larger values will typically lead to
/// better convergence but takes longer. The default is 1000.
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
/// clf.fit(x.view(), y.view());
/// 
/// let x_test = array![[1.0, 2.0, 3.0]];
/// assert_eq!(clf.predict(x_test.view()), array![0]);
/// ```
#[derive(Clone)]
pub struct LogisticRegression {
    weights: Option<Array1<f64>>,
    pub max_iter: u64
}

struct LogisticRegressionProblem<'a, 'b> {
    train_x: ArrayView2<'a, f64>,
    train_y: ArrayView1<'b, f64>
}

impl<'a, 'b> ArgminOp for LogisticRegressionProblem<'a, 'b> {
    type Param = Array1<f64>;
    type Float = f64;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let mut sum = 0.0;
        for (x_i, y_i) in self.train_x.outer_iter().zip(self.train_y.iter()) {
            let linear_combination = p.dot(&x_i);
            sum += y_i * linear_combination - (1.0 + linear_combination.exp()).ln();
        }   
        Ok(-sum)
    }

    fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
        let mut gradient = Array1::zeros(p.len());
        for j in 0..p.len() {
            let mut sum = 0.0;
            for i in 0..self.train_x.nrows() {
                let x_i = self.train_x.index_axis(Axis(0), i);
                let inner = self.train_y[i] - logistic_probability(p.view(), x_i);
                sum += inner * x_i[j];
            }
            gradient[j] = sum;
        }
        Ok(-gradient)
    }
}


impl LogisticRegression {
    pub fn new() -> LogisticRegression {
        LogisticRegression {
            weights: None,
            max_iter: 1000
        }
    }
}

impl Classifier for LogisticRegression {
    fn fit<'a>(&mut self, x: ArrayView2<'a, f64>, y: ArrayView1<'a, usize>) {
        assert!(x.nrows() == y.len(), "Failed to fit `LogisticRegression` classifier: must have same number of samples in `x` and `y`.");

        // Map y to an Array<f64> for convenience when defining the log-
        // likelihood and gradient functions, and define the optimisation 
        // problem.
        let train_y = y.mapv(|x| x as f64);
        let cost = LogisticRegressionProblem {
            train_x: x,
            train_y: train_y.view()
        };

        // Set up a gradient descent using More–Thuente line search.
        let line_search = MoreThuenteLineSearch::new();
        let solver = SteepestDescent::new(line_search);
        // Generate a random initial point in [-1, 1]^n and hope this is 
        // reasonable.
        let x_0 = Array1::random(x.ncols(), Uniform::new(-1.0, 1.0));
        let executor = Executor::new(cost, solver, x_0)
            .max_iters(self.max_iter);
        // Execute the optimiser and save the best weights found.
        // TODO: return a result here rather than unwrapping. 
        let res = executor.run().unwrap();
        let state = res.state;
        self.weights = Some(state.best_param);
    }

    fn predict(&self, x: ArrayView2<f64>) -> Array1<usize> {
        if let Some(weights) = &self.weights {
            // Estimate the probability for each sample, and return 1 if p > 0.5, 0 otherwise.
            x.outer_iter().map(|row| logistic_probability(weights.view(), row).round() as usize).collect::<Array1<usize>>()
        } else {
            panic!("LogisticRegression classifier must be fit before usage. Use `classifier.fit(x, y)` before usage.")
        }
    }
}   

/// Calculate the probability that Y = 1 given the data X and weights for a 
/// logistic regression model.
fn logistic_probability(weights: ArrayView1<f64>, x: ArrayView1<f64>) -> f64 {
    let linear_combination: f64 = weights.dot(&x);
    1.0 / (1.0 + (-linear_combination).exp())
}

/// Calculates the log-odds of a probability p in (0, 1), defined by 
/// log(p/(1 - p)).
///
/// # Arguments
/// `p` - a probability value that must satisfy 0 < p < 1.
///
/// # Examples
/// ```
/// use ml_rs::classification::linear::log_odds;
/// let p = 0.5;
/// assert_eq!(log_odds(p), 0.0);
/// ```
pub fn log_odds(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

#[cfg(test)]
mod test {
    use super::{logistic_probability, LogisticRegression};
    use super::super::Classifier;   
    use ndarray::{Array1, Array2, array};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    #[test]
    fn test_logistic_probability() {
        let weights = array![1.0, 2.0, 1.0];
        let x = array![1.0, 3.0, 1.0];
        assert_eq!(logistic_probability(weights.view(), x.view()), 1.0 / (1.0 + (-8.0f64).exp()));
    }

    #[test]
    #[should_panic(expected = "LogisticRegression classifier must be fit before usage. Use `classifier.fit(x, y)` before usage.")]
    fn test_unfit_logistic_regression() {
        let clf = LogisticRegression::new();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        clf.predict(x.view());
    }

    #[test]
    #[should_panic(expected = "Failed to fit `LogisticRegression` classifier: must have same number of samples in `x` and `y`.")]
    fn test_logistic_regression_different_sizes() {
        let mut clf = LogisticRegression::new();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0];
        clf.fit(x.view(), y.view());
    }

    #[test]
    fn test_fit_logistic_regression() {
        let mut clf = LogisticRegression::new();
        let x = array![[1.0, 2.0], [1.0, 3.0], [3.0, 4.0], [3.0, 5.0]];
        let y = array![0, 0, 1, 1];
        clf.fit(x.view(), y.view());
        assert_eq!(array![0, 0, 1, 1], clf.predict(x.view()));
    }

    #[test]
    fn test_single_logistic_regression() {
        let mut clf = LogisticRegression::new();
        let x = array![[1.0, 2.0, 3.0],];
        let y = array![0];
        clf.fit(x.view(), y.view());
        assert_eq!(array![0], clf.predict(x.view()));
    }

    #[test]
    fn test_fit_logistic_regression_random() {
        let mut clf = LogisticRegression::new();
        let n_rows = 200;
        let n_features = 5;
        let x = Array2::random((n_rows, n_features), Uniform::new(-1.0, 1.0));
        let y = Array1::random(n_rows, Uniform::new(0, 2));
        clf.fit(x.view(), y.view());
    }
}