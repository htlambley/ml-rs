use ndarray::{Array1, ArrayView1, ArrayView2};
use super::Classifier;

/// A classifier implementing the logistic regression model. Logistic 
/// regression models can be used for binary classification problems and may
/// be extended through multinomial logistic regression.
///
/// # Model
/// TODO
/// # Examples
/// TODO
#[derive(Clone, Default)]
pub struct LogisticRegression {
    weights: Option<LogisticRegressionWeights>
}

#[derive(Clone)]
struct LogisticRegressionWeights {
    bias: f64,
    weights: Array1<f64>
}

impl LogisticRegression {
    pub fn new() -> LogisticRegression {
        LogisticRegression {
            weights: None
        }
    }
}

impl Classifier for LogisticRegression {
    /// Fits a logistic regression model to the data by choosing the maximum
    /// likelihood estimators for the weight and bias terms.
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<usize>) {
        unimplemented!();
    }

    fn predict(&self, x: ArrayView2<f64>) -> Array1<usize> {
        if let Some(weights) = &self.weights {
            // TODO: review threshold
            x.outer_iter().map(|row| logistic_probability(weights, row).round() as usize).collect::<Array1<usize>>()
        } else {
            panic!("LogisticRegression classifier must be fit before usage. Use `classifier.fit(x, y)` before usage.")
        }
    }
}


/// Calculate the probability that Y = 1 given the data X and weights for a 
/// logistic regression model.
fn logistic_probability(weights: &LogisticRegressionWeights, x: ArrayView1<f64>) -> f64 {
    let linear_combination: f64 = weights.bias + weights.weights.dot(&x);
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
    use super::{logistic_probability, LogisticRegression, LogisticRegressionWeights};
    use super::super::Classifier;   
    use ndarray::array;
    #[test]
    fn test_logistic_probability() {
        let w = LogisticRegressionWeights {
            bias: 1.0,
            weights: array![1.0, 2.0, 1.0]
        };
        let x = array![1.0, 3.0, 1.0];
        assert_eq!(logistic_probability(&w, x.view()), 1.0 / (1.0 + (-9.0f64).exp()));
    }

    #[test]
    #[should_panic(expected = "LogisticRegression classifier must be fit before usage. Use `classifier.fit(x, y)` before usage.")]
    fn test_unfit_logistic_regression() {
        let clf = LogisticRegression::new();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        clf.predict(x.view());
    }
}