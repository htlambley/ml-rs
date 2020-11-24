use super::{labels_binary, Classifier, ProbabilityBinaryClassifier};
use crate::Error;
use argmin::core;
use argmin::core::{ArgminOp, Executor};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::BFGS;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::LeastSquaresSvd;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

/// Represents a backend for the [`LogisticRegression`] classifier which
/// calculates appropriate weights. See [`BFGSSolver`] and [`IRLSSolver`]
/// for concrete implementations of this trait.
pub trait LogisticRegressionSolver {
    /// Given data matrix `x` and label array `y` (with labels converted to
    /// float), calculate suitable weights according to the chosen algorithm.
    fn fit_weights(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Array1<f64>, Error>;
}

/// Solver for the [`LogisticRegression`] classifier implementing the
/// Broyden–Fletcher–Goldfarb–Shanno (BFGS) method.
///
/// This solver fits the model using *maximum likelihood estimation* to find
/// the optimal weights. There is no closed form to find the maximum likelihood
/// estimator weights, so this solver obtains weights numerically using the
/// Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm. This is a quasi-Newton
/// iterative method, and further details are available in \[1\].
///
/// The optimisation algorithm is handled by the `argmin` library.
/// # References
/// \[1\] Nocedal and Wright, *Numerical Optimization*, Springer, New York,
/// NY, 2nd ed, 2006, pp. 136–144.
#[derive(Clone, Debug)]
pub struct BFGSSolver {
    max_iter: u64,
}

impl Default for BFGSSolver {
    fn default() -> BFGSSolver {
        BFGSSolver { max_iter: 100 }
    }
}

impl BFGSSolver {
    /// Creates a new [`BFGSSolver`] with the given maximum number of 
    /// iterations.
    pub fn with_max_iter(max_iter: u64) -> BFGSSolver {
        BFGSSolver {
            max_iter
        }
    }
}

impl LogisticRegressionSolver for BFGSSolver {
    fn fit_weights(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Array1<f64>, Error> {
        let cost = LogisticRegressionProblem {
            train_x: x,
            train_y: y,
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
            .map_err(|_| Error::OptimiserError)
    }
}

/// Solver for the [`LogisticRegression`] classifier implementing the
/// iteratively reweighted least squares (IRLS) method.
///
/// This solver attempts to obtain a maximum likelihood estimator for the
/// weights. In contrast to [`BFGSSolver`], this solver repeatedly solves
/// a least squares problem (which has a closed form solution) in order
/// to obtain an approximation for the weights.
///
/// # Details
///
/// Let:
/// - $X$ be the data matrix with corresponding labels $y$;
/// - $w_i$ be the $i$th approxmation for the weights;
/// - $p$ the vector with entries $p_j = p(x_j; w_i)$;
/// - $W$ the diagonal matrix with entries $W_{jj} = p(x_j; w_i)(1 - p(x_j; w_i))$;
/// - $z = Xw_i  + W^{-1} (y - p)$.
///
/// The next approximation is given by solving the *weighted least squares*
/// (WLS) problem
/// $$w_{i + 1} = \argmin_w (z - Xw)^T W (z - Xw),$$
/// where this more obviously a least squares problem by rewriting the
/// expression as
/// $$w_{i + 1} = \argmin_w \lVert W^{\frac12} (z - Xw) \rVert,$$
/// recalling that the traditional least squares problem is to minimise
/// $\lVert Ax - b \rVert$ over all $x$.
///
/// Internally, we solve the least squares problem
/// $$w_{i + 1} = \argmin_w \lVert b - Cw \rVert,$$
/// where $b = W^{\frac12} z$ and $C = W^{\frac12} X$ using singular value
/// decompositions.
///
/// # References
/// Hastie et al, *The Elements of Statistical Learning: Data Mining,
/// Inference and Prediction*, Springer, New York, NY, 2001, 1st ed,
/// pp. 95–100.
#[derive(Clone, Debug)]
pub struct IRLSSolver {
    max_iter: usize,
}

impl Default for IRLSSolver {
    fn default() -> IRLSSolver {
        IRLSSolver { max_iter: 10 }
    }
}

impl IRLSSolver {
    /// Creates a new [`IRLSSolver`] with the given maximum number of 
    /// iterations.
    pub fn with_max_iter(max_iter: usize) -> IRLSSolver {
        IRLSSolver {
            max_iter
        }
    }
}

impl LogisticRegressionSolver for IRLSSolver {
    fn fit_weights(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Array1<f64>, Error> {
        let mut weights: Array1<f64> = Array1::zeros(x.ncols());
        for _ in 0..self.max_iter {
            println!("T");
            let xw: Array1<f64> = x.dot(&weights);
            let mut p = xw.clone();
            p.par_mapv_inplace(sigmoid);
            // Rather than directly calculating W, then inverting and taking
            // square roots, we note that W is a diagonal matrix with non-
            // negative entries, so its inverse is the reciprocal of the
            // non-zero diagonal elements. Its square root is simply the square
            // root of each diagonal element.
            // We first calculate the diagonal elements of W.
            let mut w_diag: Array1<f64> = p.clone();
            w_diag.par_mapv_inplace(|v| {
                // p * (1 - p)
                v * (1.0 - v)
            });
            let y_sub_p = -p + &y;
            let mut w_inv = w_diag.clone();
            w_inv.par_mapv_inplace(|v| if v > 0.0 { 1.0 / v } else { 0.0 });
            // W^{-1}(y - p)
            let prod: Array1<f64> = w_inv
                .iter()
                .zip(y_sub_p.iter())
                .map(|(x, y)| x * y)
                .collect();
            let z: Array1<f64> = xw + prod;
            // w_diag is now W^{1/2}.
            w_diag.par_mapv_inplace(|v| v.sqrt());
            // W^{1/2} z
            let w_sqrt_z = w_diag.iter().zip(z.iter()).map(|(a, b)| a * b).collect();
            // a will represent W^{1/2} X
            let mut a = x.into_owned();
            a.outer_iter_mut()
                .zip(w_diag.iter())
                .for_each(|(mut row, w_i)| row.iter_mut().for_each(|el| *el *= w_i));

            let sol = a
                .least_squares(&w_sqrt_z)
                .map_err(|_| Error::FittingError)?
                .solution;
            weights = sol;
        }
        Ok(weights)
    }
}

/// A classifier implementing the logistic regression model. Logistic
/// regression models can be used for binary classification problems and may
/// be extended through multinomial logistic regression.
///
/// **Important**: you need to choose the solver you want to use in order
/// to use this model. See the 'Fitting' section below for help.
///
/// # Model
/// This model is appropriate if you have a collection of samples `x`
/// with labels `y` equal to `0` and `1`. In other words, this classifier
/// is suitable for **binary classification** tasks.
///
/// Roughly speaking, the logistic regression model tries to fit a linear
/// model to predict the probability of each class. As we require probability
/// estimates to be in $[0, 1]$, the linear model is used to predict the
/// *log-odds* instead.
///
/// ## Interpretation as a neural network
/// In the language of neural networks, a binary logistic regression classifier
/// is a single-neuron network where the neuron uses sigmoid activation.
/// The model is trained by minimising the cross-entropy (which can be shown
/// to be equivalent to maximising the log-likelihood). The ideas underlying
/// logistic regression are much older than those in modern neural network
/// theory, and the fitting process would typically be restrained to some
/// form of gradient descent in the neural network context.
///
/// ## Fitting
/// Multiple methods of fitting `LogisticRegression` are provided, so you can
/// choose whichever one you prefer or performs best on your system. The
/// current choices are:
/// - [`BFGSSolver`]
///
/// Numerically obtains the weights using a quasi-Newton iterative algorithm,
/// known as the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm.
///
/// - [`IRLSSolver`]
///
/// Obtains weights using a procedure known as iteratively reweighted
/// least squares, as given by Hastie et al.
///
/// In general, [`IRLSSolver`] is slightly faster, but check this on your
/// system as it may depend on the BLAS/LAPACK you link to.
///
/// # Examples
/// Fitting a logistic regression classifier and making a prediction, using the
/// BFGS solver.
/// ```no_run
/// use ndarray::array;
/// use ml_rs::classification::Classifier;
/// use ml_rs::classification::linear::{BFGSSolver, LogisticRegression};
///
/// let x = array![[1.0, 2.0, 3.0], [1.0, 4.0, 3.0]];
/// let y = array![0, 1];
///
/// let solver = BFGSSolver::default();
/// let mut clf = LogisticRegression::new(solver);
/// clf.fit(x.view(), y.view()).unwrap();
///
/// let x_test = array![[1.0, 2.0, 3.0]];
/// assert_eq!(clf.predict(x_test.view()).unwrap(), array![0]);
/// ```
pub struct LogisticRegression<T: LogisticRegressionSolver> {
    weights: Option<Array1<f64>>,
    solver: T,
    /// The decision threshold used to classify a sample as negative (class 0)
    /// or positive (class 1). If the probability estimate from
    /// `predict_probability` is greater than `threshold`, `predict` will
    /// output class 1 for that sample.
    ///
    /// The default is 0.5, which classifies a sample as class 1 if the
    /// probability that it is class 1 is greater than a half.
    pub threshold: f64,
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

impl<T: LogisticRegressionSolver> LogisticRegression<T> {
    /// Creates a new `LogisticRegression` classifier which must be fit on the
    /// data in order to find suitable weights.
    pub fn new(solver: T) -> LogisticRegression<T> {
        LogisticRegression {
            weights: None,
            threshold: 0.5,
            solver,
        }
    }
}

impl<T: LogisticRegressionSolver + Default> Default for LogisticRegression<T> {
    fn default() -> LogisticRegression<T> {
        let solver = T::default();
        Self::new(solver)
    }
}

impl<T: LogisticRegressionSolver> Classifier for LogisticRegression<T> {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<usize>) -> Result<(), Error> {
        if x.nrows() != y.len() || !labels_binary(y) {
            return Err(Error::InvalidTrainingData);
        }

        // Map y to an Array<f64> for convenience in the fitting process.
        let train_y = y.mapv(|x| x as f64);
        // Pass to internal solver and obtain best weights.
        let best_param = self.solver.fit_weights(x, train_y.view())?;
        self.weights = Some(best_param);
        Ok(())
    }

    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<usize>, Error> {
        let probabilties = self.predict_probability(x)?;
        // Using the probability estimates, choose class 1 if the estimate
        // is greater than `self.threshold`.
        Ok(probabilties
            .iter()
            .map(|x| if *x > self.threshold { 1 } else { 0 })
            .collect())
    }
}

impl<T: LogisticRegressionSolver> ProbabilityBinaryClassifier for LogisticRegression<T> {
    fn predict_probability(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, Error> {
        // TODO: return an iterator so the consumer can map if necessary
        if let Some(weights) = &self.weights {
            // Estimate the probability for each sample.
            let mut xw = x.dot(weights);
            xw.par_mapv_inplace(sigmoid);
            Ok(xw)
        } else {
            Err(Error::UseBeforeFit)
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
    use super::{BFGSSolver, IRLSSolver, LogisticRegression};
    use ndarray::{array, Array1, Array2};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_unfit_logistic_regression() {
        let solver = BFGSSolver::default();
        let clf = LogisticRegression::new(solver);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        match clf.predict(x.view()) {
            Err(Error::UseBeforeFit) => (),
            _ => panic!("Classifier did not return correct error"),
        }
    }

    #[test]
    fn test_logistic_regression_different_sizes() {
        let solver = BFGSSolver::default();
        let mut clf = LogisticRegression::new(solver);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0];
        match clf.fit(x.view(), y.view()) {
            Err(Error::InvalidTrainingData) => (),
            _ => panic!("Classifier did not return correct error"),
        }
    }

    #[test]
    fn test_fit_logistic_regression() {
        let solver = BFGSSolver::default();
        let mut clf = LogisticRegression::new(solver);
        let x = array![[1.0, 2.0], [1.0, 3.0], [3.0, 4.0], [3.0, 5.0]];
        let y = array![0, 0, 1, 1];
        clf.fit(x.view(), y.view()).unwrap();
        assert_eq!(array![0, 0, 1, 1], clf.predict(x.view()).unwrap());
    }

    #[test]
    fn test_fit_logistic_regression_irls() {
        let solver = IRLSSolver::default();
        let mut clf = LogisticRegression::new(solver);
        let x = array![[1.0, 2.0], [1.0, 3.0], [3.0, 4.0], [3.0, 5.0]];
        let y = array![0, 0, 1, 1];
        clf.fit(x.view(), y.view()).unwrap();
        assert_eq!(array![0, 0, 1, 1], clf.predict(x.view()).unwrap());
    }

    #[test]
    fn test_logistic_regression_non_binary_labels() {
        let solver = BFGSSolver::default();
        let mut clf = LogisticRegression::new(solver);
        let x = array![[1.0, 2.0], [1.0, 3.0], [3.0, 4.0], [3.0, 5.0]];
        let y = array![0, 0, 1, 2];
        match clf.fit(x.view(), y.view()) {
            Err(Error::InvalidTrainingData) => (),
            _ => panic!("Classifier did not return correct error"),
        }
    }

    #[test]
    fn test_single_logistic_regression() {
        let solver = BFGSSolver::default();
        let mut clf = LogisticRegression::new(solver);
        let x = array![[1.0, 2.0, 3.0],];
        let y = array![0];
        clf.fit(x.view(), y.view()).unwrap();
        assert_eq!(array![0], clf.predict(x.view()).unwrap());
    }

    #[test]
    fn test_fit_logistic_regression_random() {
        let solver = BFGSSolver::default();
        let mut clf = LogisticRegression::new(solver);
        let n_rows = 2000;
        let n_features = 5;
        let x = Array2::random((n_rows, n_features), Uniform::new(-1.0, 1.0));
        let y = Array1::random(n_rows, Uniform::new(0, 2));
        clf.fit(x.view(), y.view()).unwrap();
    }

    #[test]
    fn test_fit_logistic_regression_random_large() {
        let solver = BFGSSolver::default();
        let mut clf = LogisticRegression::new(solver);
        let n_rows = 100000;
        let n_features = 5;
        let x = Array2::random((n_rows, n_features), Uniform::new(-1.0, 1.0));
        let y = Array1::random(n_rows, Uniform::new(0, 2));
        clf.fit(x.view(), y.view()).unwrap();
    }

    #[test]
    fn test_fit_irls_logistic_regression_random_large() {
        let solver = IRLSSolver::default();
        let mut clf = LogisticRegression::new(solver);
        let n_rows = 100000;
        let n_features = 5;
        let x = Array2::random((n_rows, n_features), Uniform::new(-1.0, 1.0));
        let y = Array1::random(n_rows, Uniform::new(0, 2));
        clf.fit(x.view(), y.view()).unwrap();
    }

    #[test]
    fn test_unfit_irls_logistic_regression() {
        let solver = IRLSSolver::default();
        let clf = LogisticRegression::new(solver);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        match clf.predict(x.view()) {
            Err(Error::UseBeforeFit) => (),
            _ => panic!("Classifier did not return correct error"),
        }
    }
}
