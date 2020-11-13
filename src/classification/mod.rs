use ndarray::{Array1, ArrayView1, ArrayView2};
use std::collections::HashMap;

pub mod linear;

pub trait Classifier {
    fn fit<'a>(&mut self, x: ArrayView2<'a, f64>, y: ArrayView1<'a, usize>);
    fn predict(&self, x: ArrayView2<f64>) -> Array1<usize>;
}

/// A trivial classifier that is initialised with a class label and outputs
/// that label for any sample it is given.
///
/// This may be useful to demonstrate the performance of the most naive
/// models, which in the case of highly imbalanced classes may have a "good"
/// accuracy score. For example, if 95% of your data has class 0, then the
/// `TrivialClassifier` with class 0 may be expected to be 95% accurate newly
/// sampled data from that distribution.
///
/// The trivial classifier does not require fitting as it does not learn 
/// from the dataset.
///
/// A slightly more advanced version of the `TrivialClassifier` is the 
/// `MajorityClassifier`, which learns the most common class and outputs
/// this for every sample.
#[derive(Clone)]
pub struct TrivialClassifier {
    class: usize,
}

impl TrivialClassifier {
    pub fn new(class: usize) -> TrivialClassifier {
        TrivialClassifier { class }
    }
}

impl Classifier for TrivialClassifier {
    fn fit(&mut self, _: ArrayView2<f64>, _: ArrayView1<usize>) {}
    fn predict(&self, x: ArrayView2<f64>) -> Array1<usize> {
        Array1::ones(x.nrows()) * self.class
    }
}

/// A classifier which learns the most common class and predicts this class
/// for all unseen data.
///
/// # Examples
/// ```
/// use ml_rs::classification::{Classifier, MajorityClassifier};
/// use ndarray::array;
/// let x = array![[0.], [1.], [2.]];
/// let y = array![0, 0, 1];
///
/// let mut classifier = MajorityClassifier::new();
/// classifier.fit(x.view(), y.view());
/// let y_pred = classifier.predict(x.view());
/// ```
#[derive(Clone, Default)]
pub struct MajorityClassifier {
    class: Option<usize>,
}

impl MajorityClassifier {
    pub fn new() -> MajorityClassifier {
        MajorityClassifier { class: None }
    }
}

impl Classifier for MajorityClassifier {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<usize>) {
        let x_len = x.nrows();
        let y_len = y.len();
        assert!(x_len == y_len, "Training features and labels must be of equal length. x has {} rows but y has {} rows.", x_len, y_len);

        let mut frequency_map: HashMap<usize, usize> = HashMap::new();
        for class in y {
            let class_frequency = frequency_map.entry(*class).or_insert(0);
            *class_frequency += 1;
        }

        let mut max_frequency = 0;
        let mut max_class = 0;
        for (class, frequency) in frequency_map {
            if frequency > max_frequency {
                max_frequency = frequency;
                max_class = class;
            }
        }

        self.class = Some(max_class);
    }

    fn predict(&self, x: ArrayView2<f64>) -> Array1<usize> {
        if let Some(class) = self.class {
            Array1::ones(x.nrows()) * class
        } else {
            panic!("MajorityClassifier must be fit on data before usage. Use `classifier.fit(x, y)` before trying to predict on data.")
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Classifier, MajorityClassifier, TrivialClassifier};
    use ndarray::array;

    #[test]
    fn test_trivial_classifier_predictions() {
        let clf = TrivialClassifier::new(0);
        let x = array![[1.0, 2.0], [1.0, 3.0]];
        assert_eq!(clf.predict(x.view()), array![0, 0]);
    }

    #[test]
    #[should_panic(
        expected = "MajorityClassifier must be fit on data before usage. Use `classifier.fit(x, y)` before trying to predict on data."
    )]
    fn test_majority_classifier_unfit() {
        let clf = MajorityClassifier::new();
        let x = array![[1.0, 2.0], [1.0, 3.0]];
        clf.predict(x.view());
    }

    #[test]
    fn test_majority_classifier_predictions() {
        let mut clf = MajorityClassifier::new();

        let x = array![[1.0, 2.0], [1.0, 3.0], [4.0, 1.0]];
        let y = array![1, 1, 0];

        clf.fit(x.view(), y.view());
        assert_eq!(clf.predict(x.view()), array![1, 1, 1]);
    }

    #[test]
    #[should_panic(
        expected = "Training features and labels must be of equal length. x has 3 rows but y has 4 rows."
    )]
    fn test_majority_classifier_differing_sizes() {
        let mut clf = MajorityClassifier::new();

        let x = array![[1.0, 2.0], [1.0, 3.0], [4.0, 1.0]];
        let y = array![1, 1, 0, 1];
        clf.fit(x.view(), y.view());
    }
}
