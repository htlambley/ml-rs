use super::MetricError;
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
/// If successful, a pair representing the precision and recall respectively.
/// In the degenerate case that the classifier makes no positive predictions,
/// the precision is undefined and `None` will be returned. Likewise,
/// if there are no positive cases in the data, the recall is undefined
/// and `None` will be returned.
///
/// If invalid input is passed, a suitable error is returned.
pub fn precision_recall_score(
    y_true: ArrayView1<usize>,
    y_pred: ArrayView1<usize>,
) -> Result<(Option<f64>, Option<f64>), MetricError> {
    let n_true = y_true.len();
    let n_pred = y_pred.len();

    if n_true != n_pred {
        return Err(MetricError::PredictionLengthsDifferent { n_true, n_pred });
    }

    if n_true == 0 {
        return Err(MetricError::NoPredictions);
    }

    if !labels_binary(y_true) || !labels_binary(y_pred) {
        return Err(MetricError::NotBinary);
    }

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

    Ok((precision, recall))
}

#[cfg(test)]
mod tests {
    use super::super::MetricError;
    use super::precision_recall_score;
    use ndarray::array;

    #[test]
    fn test_precision_recall() {
        let y_true = array![1, 0, 1, 1, 0];
        let y_pred = array![1, 0, 1, 1, 1];
        let (precision, recall) = precision_recall_score(y_true.view(), y_pred.view()).unwrap();
        assert_eq!(precision, Some(0.75));
        assert_eq!(recall, Some(1.0));
    }

    #[test]
    fn test_precision_recall_differing_lengths() {
        let y_true = array![1, 0, 1, 1, 0, 1];
        let y_pred = array![1, 0, 1, 1, 1];
        match precision_recall_score(y_true.view(), y_pred.view()) {
            Err(MetricError::PredictionLengthsDifferent { n_true, n_pred }) => {
                assert_eq!(n_true, 6);
                assert_eq!(n_pred, 5);
            }
            _ => panic!("Incorrect error returned"),
        }
    }

    #[test]
    fn test_precision_recall_zero_length() {
        let y_true = array![];
        let y_pred = array![];
        match precision_recall_score(y_true.view(), y_pred.view()) {
            Err(MetricError::NoPredictions) => {}
            _ => panic!("Incorrect error returned"),
        }
    }

    #[test]
    fn test_precision_recall_nonbinary_true() {
        let y_true = array![1, 0, 1, 1, 2];
        let y_pred = array![1, 0, 1, 1, 1];
        match precision_recall_score(y_true.view(), y_pred.view()) {
            Err(MetricError::NotBinary) => {}
            _ => panic!("Incorrect error returned"),
        }
    }

    #[test]
    fn test_precision_recall_nonbinary_pred() {
        let y_true = array![1, 0, 1, 1, 1];
        let y_pred = array![1, 0, 1, 1, 2];
        match precision_recall_score(y_true.view(), y_pred.view()) {
            Err(MetricError::NotBinary) => {}
            _ => panic!("Incorrect error returned"),
        }
    }

    #[test]
    fn test_precision_recall_zero_true_preds() {
        let y_true = array![1, 0, 1, 1, 1];
        let y_pred = array![0, 0, 0, 0, 0];
        let (precision, recall) = precision_recall_score(y_true.view(), y_pred.view()).unwrap();
        assert_eq!(precision, None);
        assert_eq!(recall, Some(0.0));
    }

    #[test]
    fn test_precision_recall_zero_true() {
        let y_true = array![0, 0, 0, 0, 0];
        let y_pred = array![0, 0, 0, 0, 1];
        let (precision, recall) = precision_recall_score(y_true.view(), y_pred.view()).unwrap();
        assert_eq!(precision, Some(0.0));
        assert_eq!(recall, None);
    }

    #[test]
    fn test_precision_recall_half() {
        let y_true = array![0, 1, 0, 1];
        let y_pred = array![0, 1, 1, 0];
        let (precision, recall) = precision_recall_score(y_true.view(), y_pred.view()).unwrap();
        assert_eq!(precision, Some(0.5));
        assert_eq!(recall, Some(0.5));
    }
}
