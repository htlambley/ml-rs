use ml_rs::classification::linear::{IRLSSolver, LogisticRegression};
use ml_rs::classification::Classifier;
use ml_rs::metrics::accuracy_score;
use ml_rs::datasets::load_breast_cancer;

fn main() {
    let (x, y) = load_breast_cancer();

    let solver = IRLSSolver::with_max_iter(5);
    let mut classifier = LogisticRegression::new(solver);
    classifier.fit(x.view(), y.view()).unwrap();

    // Test training accuracy
    let y_pred = classifier.predict(x.view()).unwrap();
    println!("{}", y_pred);
    let accuracy = accuracy_score(y.view(), y_pred.view()).unwrap();
    println!("Training set accuracy: {}%", accuracy * 100.);
}
