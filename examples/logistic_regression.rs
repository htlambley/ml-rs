use ml_rs::classification::Classifier;
use ml_rs::classification::linear::LogisticRegression;
use ml_rs::metrics::accuracy_score;
use ndarray::array;

fn main() {
    let x = array![[0., 3.], [1., 2.], [1., 3.], [0.5, 1.7]];
    let y = array![0, 1, 1, 1];

    // Initialise new LogisticRegression classifier and fit to data
    let mut classifier = LogisticRegression::new();
    // The classifier only needs an `ArrayView` as it does not consume the
    // training data, so we pass with .view()
    classifier.fit(x.view(), y.view());

    let x_test = array![[1., 5.], [0.6, 1.]];
    // .predict() can take a 2D array, and returns a 1D array with the
    // predictions for each row.
    let y_pred = classifier.predict(x_test.view());
    let y_true = array![1, 1];
    let accuracy = accuracy_score(y_true.view(), y_pred.view());
    assert_eq!(accuracy, 1.0);
}