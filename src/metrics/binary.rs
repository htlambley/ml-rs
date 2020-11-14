use crate::classification::labels_binary;
use ndarray::ArrayView1;

pub fn precision_recall_score(
    y_true: ArrayView1<usize>,
    y_pred: ArrayView1<usize>,
) -> (Option<f64>, Option<f64>) {
    assert!(y_true.len() == y_pred.len(), "`precision_recall_score` called for label vectors of different lengths. Ensure that `y_true` and `y_pred` have the same length.");
    assert!(
        y_true.len() > 0,
        "`y_true` has no predictions. Ensure that `y_true` does not have length 0."
    );
    assert!(
        labels_binary(y_true),
        "`y_true` must be an array of binary labels (in {0, 1})."
    );
    assert!(
        labels_binary(y_pred),
        "`y_pred` must be an array of binary labels (in {0, 1})."
    );

    // The number of cases that were correctly predicted to be positive
    let mut true_positives = 0;
    // The number of cases predicted to be positive in y_pred
    let mut predicted_positive = 0;
    // The number of positive cases in the true dataset
    let mut total_positive = 0;

    for (true_value, predicted_value) in y_true.iter().zip(y_pred.iter()) {
        total_positive += true_value;
        predicted_positive += predicted_value;
        true_positives += true_value * predicted_value;
    }

    let precision = if predicted_positive > 0 {
        Some((true_positives as f64) / (predicted_positive as f64))
    } else {
        None
    };

    let recall = if total_positive > 0 {
        Some((true_positives as f64) / (total_positive as f64))
    } else {
        None
    };

    (precision, recall)
}

#[cfg(test)]
mod tests {
    use super::precision_recall_score;
    use ndarray::array;

    #[test]
    fn test_precision_recall() {
        let y_true = array![1, 0, 1, 1, 0];
        let y_pred = array![1, 0, 1, 1, 1];
        let (precision, recall) = precision_recall_score(y_true.view(), y_pred.view());
        assert_eq!(precision, Some(0.75));
        assert_eq!(recall, Some(1.0));
    }

    #[test]
    #[should_panic(
        expected = "`precision_recall_score` called for label vectors of different lengths. Ensure that `y_true` and `y_pred` have the same length."
    )]
    fn test_precision_recall_differing_lengths() {
        let y_true = array![1, 0, 1, 1, 0, 1];
        let y_pred = array![1, 0, 1, 1, 1];
        precision_recall_score(y_true.view(), y_pred.view());
    }

    #[test]
    #[should_panic(
        expected = "`y_true` has no predictions. Ensure that `y_true` does not have length 0."
    )]
    fn test_precision_recall_zero_length() {
        let y_true = array![];
        let y_pred = array![];
        precision_recall_score(y_true.view(), y_pred.view());
    }

    #[test]
    #[should_panic(expected = "`y_true` must be an array of binary labels (in {0, 1}).")]
    fn test_precision_recall_nonbinary_true() {
        let y_true = array![1, 0, 1, 1, 2];
        let y_pred = array![1, 0, 1, 1, 1];
        precision_recall_score(y_true.view(), y_pred.view());
    }

    #[test]
    #[should_panic(expected = "`y_pred` must be an array of binary labels (in {0, 1}).")]
    fn test_precision_recall_nonbinary_pred() {
        let y_true = array![1, 0, 1, 1, 1];
        let y_pred = array![1, 0, 1, 1, 2];
        precision_recall_score(y_true.view(), y_pred.view());
    }

    #[test]
    fn test_precision_recall_zero_true_preds() {
        let y_true = array![1, 0, 1, 1, 1];
        let y_pred = array![0, 0, 0, 0, 0];
        let (precision, recall) = precision_recall_score(y_true.view(), y_pred.view());
        assert_eq!(precision, None);
        assert_eq!(recall, Some(0.0));
    }

    #[test]
    fn test_precision_recall_zero_true() {
        let y_true = array![0, 0, 0, 0, 0];
        let y_pred = array![0, 0, 0, 0, 1];
        let (precision, recall) = precision_recall_score(y_true.view(), y_pred.view());
        assert_eq!(precision, Some(0.0));
        assert_eq!(recall, None);
    }

    #[test]
    fn test_precision_recall_half() {
        let y_true = array![0, 1, 0, 1];
        let y_pred = array![0, 1, 1, 0];
        let (precision, recall) = precision_recall_score(y_true.view(), y_pred.view());
        assert_eq!(precision, Some(0.5));
        assert_eq!(recall, Some(0.5));
    }
}
