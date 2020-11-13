use ndarray::ArrayView1;

/// Calculate the accuracy of an array of predictions `y_pred` from a 
/// classifier against the true values given in `y_true`.
///
/// The **accuracy** of a classifier is the percentage of labels that are
/// correct in the prediction array `y_pred`. It lies in [0, 1], with 1 
/// representing perfect accuracy.
///
/// # Arguments
/// `y_true` - a view to a 1D array containing the true labels for the samples
/// `y_pred` - a view to a 1D array containing the predicted labels (as 
/// generated, for example, through `classifier.predict()`).
/// 
/// `y_true` and `y_pred` must have the same length. 
pub fn accuracy_score(y_true: ArrayView1<usize>, y_pred: ArrayView1<usize>) -> f64 {
    let len_y_true = y_true.len();
    let len_y_pred = y_pred.len();
    if len_y_true != len_y_pred {
        panic!("Prediction arrays must be of same length. y_true has length {} but y_pred has length {}.", len_y_true, len_y_pred)
    }

    let mut n_correct = 0;
    let mut total = 0;
    for (true_class, pred_class) in y_true.iter().zip(y_pred.iter()) {
        if true_class == pred_class {
            n_correct += 1;
        }
        total += 1;
    }
    (n_correct as f64) / (total as f64)
}

#[cfg(test)]
mod tests {
    use super::accuracy_score;
    use ndarray::array;

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
}
