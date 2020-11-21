use crate::Error;
use ndarray::{Array1, ArrayView1, ArrayView2};
use std::collections::HashMap;

/// Classifiers based on linear regression (which, despite its name, is a
/// classification model).
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

/// Represents a classifier that can be fit on numeric data and
/// outputs a discrete prediction of the correct class.
///
/// A broad variety of models exist for solving binary classification problems.
/// These often make distinct assumptions which should be considered carefully
/// when applying these to a particular problem.
///
/// Models that can also provide probability estimates will implement the
/// [`ProbabilityBinaryClassifier`] trait as well as this trait.
///
/// See the module-level documentation for a broad overview of classification
/// with ml-rs.
pub trait Classifier {
    /// Fits the classifier to the given data matrix `x` and labels `y`. This
    /// does not support *online learning* and running this on a classifier
    /// that has already been fit will lose the previously learned parameters.
    ///
    /// # Arguments
    /// - `x`: a view to a 2-dimensional data matrix where each row corresponds
    ///        to a sample. The data matrix is unchanged and only a view, which
    ///        can be obtained by the `.view()` method of an array, is needed.
    ///
    /// - `y`: a view to a 1-dimensional array containing the labels for each
    ///        row of `x`. Labels are required to be non-negative integers,
    ///        so non-integer labels will need to be transformed before use.
    ///
    /// # Returns
    /// This method returns a result that should be checked before attempting
    /// to make predictions. If fitting was successful, `Ok(())` is returned,
    /// and a corresponding `Error` is returned if a problem occurred.
    fn fit<'a>(&mut self, x: ArrayView2<'a, f64>, y: ArrayView1<'a, usize>) -> Result<(), Error>;
    /// Makes a prediction for the sample in each row of the data matrix `x`,
    /// returning the corresponding labels as an `Array1<usize>`.
    ///
    /// Most classifiers require fitting before use with the `fit()` method,
    /// and failing to do so will lead to an error being returned.
    ///
    /// # Arguments
    /// - `x`: a view to a 2-dimensional data matrix where each row corresponds
    ///        to a sample. The data matrix is unchanged and only a view, which
    ///        can be obtained by the `.view()` method of an array, is needed.
    ///
    /// # Returns
    /// If successful, the `Result` value is `Ok` and contains an `Array1` with
    /// each element corresponding to the prediction for the corresponding row
    /// in the data matrix. Otherwise, an `Error` is returned.
    ///
    /// # Examples
    /// ```no_run
    /// use ndarray::array;
    /// use ml_rs::classification::{Classifier, MajorityClassifier};
    ///
    /// // Use some dummy training data
    /// let x_train = array![[1.0, 2.0], [3.0, 4.0]];
    /// let y_train = array![0, 0];
    ///
    /// // Create a classifier and fit, unwrapping to ensure to error occurs.
    /// let mut classifier = MajorityClassifier::new();
    /// classifier.fit(x_train.view(), y_train.view()).unwrap();
    ///
    /// // Use further unseen test data
    /// let x_test = array![[-5.0, 1.0]];
    /// // Make a prediction on `x_test`, which has 1 row, so we expect one prediction.
    /// let y_pred = classifier.predict(x_test.view()).unwrap();
    /// assert_eq!(y_pred, array![0]);
    /// ```
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<usize>, Error>;
}

/// A binary classifier that can return calibrated probability estimates in the
/// range $[0, 1]$ for a given sample.
///
/// Any classifier that implements the [`ProbabilityBinaryClassifier`] trait must
/// also implement the [`Classifier`] trait. The `predict()` method of the
/// [`Classifier`] trait is suitable for use when you only need class predictions
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
/// use ml_rs::classification::linear::{BFGSSolver, LogisticRegression};
/// use ml_rs::classification::{Classifier, ProbabilityBinaryClassifier};
/// use ndarray::array;
///
/// let x = array![[1.0, 2.0], [3.0, 4.0]];
/// let y = array![0, 1];
/// let mut clf = LogisticRegression::<BFGSSolver>::default();
/// clf.fit(x.view(), y.view());
/// let y_prob = clf.predict_probability(x.view());
/// ```
pub trait ProbabilityBinaryClassifier: Classifier {
    /// Makes an estimate of the probability that each sample in the data
    /// matrix `x` is in class 1 (the *positive class*). Values close to 1.0
    /// indicate that the classifier is highly confident that the sample is
    /// in the positive class given the training data.
    ///
    /// # Arguments
    /// - `x`: a view to a 2-dimensional data matrix where each row corresponds
    ///        to a sample. The data matrix is unchanged and only a view, which
    ///        can be obtained by the `.view()` method of an array, is needed.
    ///
    /// # Returns
    /// As noted in the trait description, the returned values are calibrated
    /// probability estimates in the interval $[0, 1]$, but you should bear
    /// in mind that this is the probability estimate given the modelling
    /// assumptions and training data. If the training data are not i.i.d.
    /// or the modelling assumptions are incorrect, you may receive
    /// poor results.
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
/// from the dataset. The class predicted depends only on the value passed
/// in `new()`.
///
/// A slightly more advanced version of the [`TrivialClassifier`] is the
/// [`MajorityClassifier`], which learns the most common class and outputs
/// this for every sample.
#[derive(Clone)]
pub struct TrivialClassifier {
    class: usize,
}

impl TrivialClassifier {
    /// Creates a new `TrivialClassifier` which will always return the class
    /// passed in the argument `class`.
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
/// The classifier will learn the majority class in the training data and
/// assumes that this training data is a *class-balanced* sample i.i.d. from
/// the true data distribution. If the training data has a different majority
/// class to the unseen data, the performance of this classifier will be
/// particularly poor.
///
/// This classifier is not intended for use in serious applications, and just
/// serves as a baseline for a model that does not use any information from the
/// unseen features at all. If a more advanced model is performing worse than
/// the [`MajorityClassifier`], you should consider whether there are serious
/// problems in the modelling assumptions for the advanced model that mean that
/// it cannot even learn this naive rule.
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
    /// Creates a new [`MajorityClassifier`] ready to be fit on the data.
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
            Err(Error::UseBeforeFit)
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Classifier, MajorityClassifier, TrivialClassifier};
    use crate::Error;
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
            Err(Error::UseBeforeFit) => (),
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
