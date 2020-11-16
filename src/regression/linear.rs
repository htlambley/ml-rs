use super::Regressor;
use ndarray::{Array1, ArrayView1, ArrayView2};
use ndarray_linalg::LeastSquaresSvd;
/// Fits a linear model to data using the ordinary least squares method.
///
/// # Model
/// This method is appropriate when you wish to model your response variable Y
/// as a linear combination of the features,
/// \hat{Y} = \alpha_1 X_1 + ... + \alpha_n X_n.
/// based on the hypothesis that each training data (x_i, y_i) is in fact
/// of the form y_i = \alpha_1 x_1 + ... + \alpha_n x_n + \varepsilon_i,
/// for some random noise \varepsilon_i.
///
/// This model fits a line that minimises the sum of the squares
/// (y_i - \hat{y}_i)^2 over the training data. This can be done through
/// a closed form calculation by forming each x_i as a row in a matrix X
/// and solving the least squares problem using the singular value
/// decomposition of X.
///
/// This model does not currently support an intercept term. For now, add
/// an additional feature to your data which is a constant 1, and this will
/// allow the model to fit a suitable intercept term.
///
/// # Examples
/// ```no_run
/// use ml_rs::regression::Regressor;
/// use ml_rs::regression::linear::LinearRegression;
/// use ndarray::array;
///
/// let x = array![[1.0, 0.0, 0.0], [2.0, 1.0, 2.0], [3.0, 2.0, 1.0]];
/// let y = array![0.0, 1.0, 2.0];
/// let mut regressor = LinearRegression::new();
/// regressor.fit(x.view(), y.view());
/// let predictions = regressor.predict(x.view());
/// ```
#[derive(Clone, Default)]
pub struct LinearRegression {
    weights: Option<Array1<f64>>,
}

impl LinearRegression {
    pub fn new() -> LinearRegression {
        LinearRegression { weights: None }
    }
}

impl Regressor for LinearRegression {
    fn fit<'a>(&mut self, x: ArrayView2<'a, f64>, y: ArrayView1<'a, f64>) {
        assert!(x.nrows() == y.len(), "`LinearRegression` could not be fit: `x` and `y` must have the same number of samples.");
        let w = x
            .least_squares(&y)
            .expect("Could not solve least squares problem.");
        self.weights = Some(w.solution);
    }

    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        if let Some(weights) = &self.weights {
            x.dot(weights)
        } else {
            panic!("`LinearRegression` must be fit before usage. Call `regressor.fit(x, y)` before attempting to make predictions.");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::Regressor;
    use super::LinearRegression;
    use ndarray::{array, Array1, Array2};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_linear_regression_simple() {
        let x = array![[1., 2.], [3., 4.]];
        let y = array![0., 1.];
        let mut regressor = LinearRegression::new();
        regressor.fit(x.view(), y.view());
        let x_test = array![[2., 3.]];
        let y_pred = regressor.predict(x_test.view());
        assert!((y_pred[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_fit_linear_regression_random_large() {
        let mut regressor = LinearRegression::new();
        let n_rows = 100000;
        let n_features = 500;
        let x = Array2::random((n_rows, n_features), Uniform::new(-1.0, 1.0));
        let y = Array1::random(n_rows, Uniform::new(0., 2.));
        regressor.fit(x.view(), y.view());
    }

    #[test]
    #[should_panic(
        expected = "`LinearRegression` could not be fit: `x` and `y` must have the same number of samples."
    )]
    fn test_fit_linear_regression_different_lengths() {
        let x = array![[1., 2.], [3., 4.]];
        let y = array![0., 1., 2.];
        let mut regressor = LinearRegression::new();
        regressor.fit(x.view(), y.view());
    }
}
