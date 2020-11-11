use ndarray::ArrayView1;

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
