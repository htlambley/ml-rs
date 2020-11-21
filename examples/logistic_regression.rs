use ml_rs::classification::linear::{BFGSSolver, LogisticRegression};
use ml_rs::classification::{Classifier, ProbabilityBinaryClassifier};
use ml_rs::metrics::accuracy_score;
use ml_rs::preprocessing::CsvReader;
use ndarray::{array, s, Array2, Axis};
use std::fs::File;

const BASE: &'static str = env!("CARGO_MANIFEST_DIR");

fn main() {
    let csv_file = File::open(BASE.to_owned() + "/examples/test.csv").unwrap();
    let reader = CsvReader::new(&csv_file);
    let data: Array2<f64> = reader.read(3, 4);
    let x = data.slice(s![.., 0..3]);
    let y = data.index_axis(Axis(1), 3).mapv(|x| x as usize);

    // Initialise new LogisticRegression classifier and fit to data.
    // The `LogisticRegression` classifier provides multiple internal solvers
    // to choose from, as noted in the documentation. Generally `BFGSSolver`
    // is a reasonable choice, and if you're happy with the defaults, can be
    // set up as follows.
    let mut classifier = LogisticRegression::<BFGSSolver>::default();
    // The classifier only needs an `ArrayView` as it does not consume the
    // training data, so we pass with .view()
    // This returns a result that you should check to ensure the fitting
    // process worked as expected.
    classifier.fit(x, y.view()).unwrap();

    let x_test = array![[1., 3., 5.]];

    // .predict() can take a 2D array, and returns a 1D array with the
    // predictions for each row.
    // This also returns a result which should be dealt with appropriately:
    // if you can, use the ? operator for convenience.
    let y_pred = classifier.predict(x_test.view()).unwrap();
    let y_true = array![0];

    // We can measure the accuracy of our predictions using accuracy_score,
    // which returns a value in [0, 1]
    let accuracy = accuracy_score(y_true.view(), y_pred.view());
    assert_eq!(accuracy, 1.0);

    // Logistic regression can also return calibrated probability estimates
    // using `predict_probability()`.
    let y_prob = classifier.predict_probability(x_test.view()).unwrap();
    println!("{}", y_prob);
}
