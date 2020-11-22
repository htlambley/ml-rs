use ndarray::ArrayView1;
use thiserror::Error;

/// Metrics particularly suitable for binary classification problems, such
/// as precision, recall and false/true positive rate.
pub mod binary;

/// Calculate the accuracy of an array of predictions `y_pred` from a
/// classifier against the true values given in `y_true`.
///
/// The **accuracy** of a classifier is the percentage of predictions that are
/// correct in the array `y_pred`. It lies in $[0, 1]$, with 1.0
/// representing perfect accuracy.
///
/// # Arguments
/// `y_true` - a view to a 1D array containing the true labels for the samples
/// `y_pred` - a view to a 1D array containing the predicted labels (as
/// generated, for example, through `classifier.predict()`).
///
/// `y_true` and `y_pred` must have the same length.
pub fn accuracy_score<T: PartialEq>(
    y_true: ArrayView1<T>,
    y_pred: ArrayView1<T>,
) -> Result<f64, MetricError> {
    let n_true = y_true.len();
    let n_pred = y_pred.len();
    if n_true != n_pred {
        return Err(MetricError::PredictionLengthsDifferent { n_true, n_pred });
    }

    if n_true == 0 {
        return Err(MetricError::NoPredictions);
    }

    let n_correct = y_true
        .iter()
        .zip(y_pred.iter())
        .fold(0, |acc, (x, y)| if x == y { acc + 1 } else { acc });

    Ok((n_correct as f64) / (n_true as f64))
}

/// Errors that may occur when computing a metric on model output.
///
/// This error will be returned by any metric in this module, although some
/// errors are specific to metrics on binary classification (see [`binary`]).
#[derive(Clone, Debug, Error)]
pub enum MetricError {
    /// Either `y_true` or `y_pred` was empty and did not contain any elements,
    /// so no meaningful metric can be calculated.
    #[error("at least one of the prediction arrays was empty")]
    NoPredictions,
    /// The arrays `y_true` and `y_pred` had different lengths, as returned
    /// in `n_true` and `n_pred`. This is not valid because it means
    /// the predictions probably do not correspond to the data and there
    /// is a mistake somewhere.
    #[error("the prediction and true value arrays had different lengths")]
    PredictionLengthsDifferent { n_true: usize, n_pred: usize },
    /// For metrics in [`binary`], `y_true` and `y_pred` must be an array of 0
    /// and 1 values only. If `y_true` contains values outside of the permitted
    /// choices, convert these to binary if possible, or if there are more than
    /// two classes, try a non-binary classification method. If `y_pred`
    /// contains invalid values, you need to switch to a binary classifier.
    #[error(
        "one of the arrays passed was not an array of binary labels: ensure values are all 0 or 1"
    )]
    NotBinary,
}

#[cfg(test)]
mod tests {
    use super::{accuracy_score, MetricError};
    use ndarray::{array, Array1};

    #[test]
    fn test_accuracy_none_equal() {
        let y_true = array![1, 2, 1];
        let y_pred = array![0, 1, 2];
        let acc = accuracy_score(y_true.view(), y_pred.view()).unwrap();
        assert_eq!(acc, 0.0);
    }

    #[test]
    fn test_accuracy_all_equal() {
        let y_true = array![1, 2, 3];
        let acc = accuracy_score(y_true.view(), y_true.view()).unwrap();
        assert_eq!(acc, 1.0);
    }

    #[test]
    fn test_accuracy_differing_sizes_right() {
        let y_true = array![1, 2, 1];
        let y_pred = array![0, 1, 2, 2];
        match accuracy_score(y_true.view(), y_pred.view()) {
            Err(MetricError::PredictionLengthsDifferent { n_true, n_pred }) => {
                assert_eq!(n_true, 3);
                assert_eq!(n_pred, 4);
            }
            _ => panic!("Incorrect error returned."),
        }
    }

    #[test]
    fn test_accuracy_differing_sizes_left() {
        let y_true = array![1, 2, 1, 5];
        let y_pred = array![0, 1, 2];
        match accuracy_score(y_true.view(), y_pred.view()) {
            Err(MetricError::PredictionLengthsDifferent { n_true, n_pred }) => {
                assert_eq!(n_true, 4);
                assert_eq!(n_pred, 3);
            }
            _ => panic!("Incorrect error returned."),
        }
    }

    #[test]
    fn test_accuracy_no_preds() {
        let y_true: Array1<f64> = array![];
        let y_pred: Array1<f64> = array![];
        match accuracy_score(y_true.view(), y_pred.view()) {
            Err(MetricError::NoPredictions) => {}
            _ => panic!("Incorrect error returned."),
        }
    }
}
