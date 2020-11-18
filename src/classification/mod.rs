use ndarray::{Array1, ArrayView1, ArrayView2};
use std::collections::HashMap;

pub mod linear;

/// Convenience function to verify whether an array of labels can be used
/// in a binary classifier.
pub fn labels_binary(y: ArrayView1<usize>) -> bool {
    for label in y {
        if *label > 1 {
            return false;
        }
    }
    true
}

#[derive(Clone, Debug)]
pub enum Error {
    ClassifierNotFit,
    InvalidTrainingData,
    DidNotConverge,
}

pub trait Classifier {
    fn fit<'a>(&mut self, x: ArrayView2<'a, f64>, y: ArrayView1<'a, usize>) -> Result<(), Error>;
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<usize>, Error>;
}

/// A binary classifier that can return calibrated probability estimates in the
/// range [0, 1] for a given sample.
///
/// Any classifier that implements the `ProbabilityBinaryClassifier` trait must
/// also implement the `Classifier` trait. The `predict()` method of the
/// `Classifier` trait is suitable for use when you only need class predictions
/// rather than confidence levels.
///
/// Many classifiers use some sort of decision threshold, but this trait is
/// only applied to classifiers where you can treat the returned values
/// as a probability estimate that has been calibrated in some sense.
/// For example, support vector machines do not (naturally) return calibrated
/// probabilities, whereas logistic regression classifiers do.
///
/// # Examples
/// ```no_run
/// use ml_rs::classification::linear::LogisticRegression;
/// use ml_rs::classification::{Classifier, ProbabilityBinaryClassifier};
/// use ndarray::array;
///
/// let x = array![[1.0, 2.0], [3.0, 4.0]];
/// let y = array![0, 1];
/// let mut clf = LogisticRegression::new();
/// clf.fit(x.view(), y.view());
/// let y_prob = clf.predict_probability(x.view());
/// ```
pub trait ProbabilityBinaryClassifier: Classifier {
    fn predict_probability(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, Error>;
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
    fn fit(&mut self, _: ArrayView2<f64>, _: ArrayView1<usize>) -> Result<(), Error> {
        Ok(())
    }

    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<usize>, Error> {
        Ok(Array1::ones(x.nrows()) * self.class)
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
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<usize>) -> Result<(), Error> {
        let x_len = x.nrows();
        let y_len = y.len();
        if x_len != y_len {
            return Err(Error::InvalidTrainingData);
        }

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
        Ok(())
    }

    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<usize>, Error> {
        if let Some(class) = self.class {
            Ok(Array1::ones(x.nrows()) * class)
        } else {
            Err(Error::ClassifierNotFit)
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Classifier, Error, MajorityClassifier, TrivialClassifier};
    use ndarray::array;

    #[test]
    fn test_trivial_classifier_predictions() {
        let clf = TrivialClassifier::new(0);
        let x = array![[1.0, 2.0], [1.0, 3.0]];
        assert_eq!(clf.predict(x.view()).unwrap(), array![0, 0]);
    }

    #[test]
    fn test_majority_classifier_unfit() {
        let clf = MajorityClassifier::new();
        let x = array![[1.0, 2.0], [1.0, 3.0]];
        match clf.predict(x.view()) {
            Err(Error::ClassifierNotFit) => (),
            _ => panic!("Classifier did not return correct error"),
        }
    }

    #[test]
    fn test_majority_classifier_predictions() {
        let mut clf = MajorityClassifier::new();

        let x = array![[1.0, 2.0], [1.0, 3.0], [4.0, 1.0]];
        let y = array![1, 1, 0];

        clf.fit(x.view(), y.view()).unwrap();
        assert_eq!(clf.predict(x.view()).unwrap(), array![1, 1, 1]);
    }

    #[test]
    fn test_majority_classifier_differing_sizes() {
        let mut clf = MajorityClassifier::new();

        let x = array![[1.0, 2.0], [1.0, 3.0], [4.0, 1.0]];
        let y = array![1, 1, 0, 1];
        match clf.fit(x.view(), y.view()) {
            Err(Error::InvalidTrainingData) => (),
            _ => panic!("Classifier did not return correct error"),
        }
    }
}
