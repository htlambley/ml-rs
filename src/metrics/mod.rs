use ndarray::ArrayView1;

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
pub fn accuracy_score<T: PartialEq>(y_true: ArrayView1<T>, y_pred: ArrayView1<T>) -> f64 {
    let n_true = y_true.len();
    let n_pred = y_pred.len();
    if n_true != n_pred {
        panic!("Prediction arrays must be of same length. y_true has length {} but y_pred has length {}.", n_true, n_pred);
    }

    assert!(
        n_true > 0,
        "`y_true` has length zero. Ensure that `y_true` has at least one prediction."
    );

    let n_correct = y_true
        .iter()
        .zip(y_pred.iter())
        .fold(0, |acc, (x, y)| if x == y { acc + 1 } else { acc });

    (n_correct as f64) / (n_true as f64)
}

#[cfg(test)]
mod tests {
    use super::accuracy_score;
    use ndarray::{Array1, array};

    #[test]
    fn test_accuracy_none_equal() {
        let y_true = array![1, 2, 1];
        let y_pred = array![0, 1, 2];
        assert_eq!(accuracy_score(y_true.view(), y_pred.view()), 0.0);
    }

    #[test]
    fn test_accuracy_all_equal() {
        let y_true = array![1, 2, 3];
        assert_eq!(accuracy_score(y_true.view(), y_true.view()), 1.0);
    }

    #[test]
    #[should_panic(
        expected = "Prediction arrays must be of same length. y_true has length 3 but y_pred has length 4."
    )]
    fn test_accuracy_differing_sizes_right() {
        let y_true = array![1, 2, 1];
        let y_pred = array![0, 1, 2, 2];
        accuracy_score(y_true.view(), y_pred.view());
    }

    #[test]
    #[should_panic(
        expected = "Prediction arrays must be of same length. y_true has length 4 but y_pred has length 3."
    )]
    fn test_accuracy_differing_sizes_left() {
        let y_true = array![1, 2, 1, 5];
        let y_pred = array![0, 1, 2];
        accuracy_score(y_true.view(), y_pred.view());
    }

    #[test]
    #[should_panic(
        expected = "`y_true` has length zero. Ensure that `y_true` has at least one prediction."
    )]
    fn test_accuracy_no_preds() {
        let y_true: Array1<f64> = array![];
        let y_pred: Array1<f64> = array![];
        accuracy_score(y_true.view(), y_pred.view());
    }
}
