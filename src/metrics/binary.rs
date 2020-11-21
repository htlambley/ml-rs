use crate::classification::labels_binary;
use ndarray::ArrayView1;

/// Calculates the precision and recall of a binary classifier given the true
/// and predicted classes for some test data.
///
/// Precision and recall only make sense for **binary classification**
/// problems, so `y_true` and `y_pred` must contain only labels in $\\{0, 1\\}$.
/// The class with label 1 will be regarded as the "positive" class, and the
/// class with label 0 will be regarded as the "negative" class.
///
/// # Arguments
/// `y_true` - the true classes of the data
/// `y_pred` - the classes predicted by the binary classifier
///
/// # Interpretation
/// A **true positive** is a case where the classifier has predicted case 1 on
/// the data, and the true class was indeed 1. A **false positive** is one
/// which was predicted to be class 1, but was actually in class 0. True and
/// false negatives are defined likewise.
///
/// The **precision** of a classifier on some data $(x_1, y_1), \dots, (x_n, y_n)$
/// is the proportion of positive predictions that were true positives. This
/// lies in the interval $[0, 1]$. A perfect classifier should have precision 1.0,
/// because every positive prediction will be a true positive case. A precision
/// of 0.0 means that every positive case predicted by the classifier was actually
/// a negative case.
///
/// The **recall** of a classifier is the proportion of the positive cases that
/// the classifier identifies correctly: a recall of 0.5 means that half of the
/// positive cases in the data are predicted to be positive by the classifier.
/// This also lies in $[0, 1]$ with 1.0 being the best possible score.
///
/// # Returns
/// A pair representing the precision and recall respectively. In the
/// degenerate case that the classifier makes no positive predictions,
/// the precision is undefined and `None` will be returned. Likewise,
/// if there are no positive cases in the data, the recall is undefined
/// and `None` will be returned.
pub fn precision_recall_score(
    y_true: ArrayView1<usize>,
    y_pred: ArrayView1<usize>,
) -> (Option<f64>, Option<f64>) {
    assert!(y_true.len() == y_pred.len(), "`precision_recall_score` called for label vectors of different lengths. Ensure that `y_true` and `y_pred` have the same length.");
    assert!(
        !y_true.is_empty(),
        "`y_true` has no predictions. Ensure that `y_true` does not have length 0."
    );
    assert!(
        labels_binary(y_true),
        "`y_true` must be an array of binary labels (0 or 1)."
    );
    assert!(
        labels_binary(y_pred),
        "`y_pred` must be an array of binary labels (0 or 1)."
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
    #[should_panic(expected = "`y_true` must be an array of binary labels (0 or 1).")]
    fn test_precision_recall_nonbinary_true() {
        let y_true = array![1, 0, 1, 1, 2];
        let y_pred = array![1, 0, 1, 1, 1];
        precision_recall_score(y_true.view(), y_pred.view());
    }

    #[test]
    #[should_panic(expected = "`y_pred` must be an array of binary labels (0 or 1).")]
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
