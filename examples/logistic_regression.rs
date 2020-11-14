use ml_rs::classification::linear::LogisticRegression;
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

    // Initialise new LogisticRegression classifier and fit to data
    let mut classifier = LogisticRegression::new();
    // The classifier only needs an `ArrayView` as it does not consume the
    // training data, so we pass with .view()
    classifier.fit(x, y.view());

    let x_test = array![[1., 3., 5.]];
    // .predict() can take a 2D array, and returns a 1D array with the
    // predictions for each row.
    let y_pred = classifier.predict(x_test.view());
    let y_true = array![0];
    let accuracy = accuracy_score(y_true.view(), y_pred.view());
    assert_eq!(accuracy, 1.0);
    let y_prob = classifier.predict_probability(x_test.view());
    println!("{}", y_prob);
}
