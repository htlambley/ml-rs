use criterion::{criterion_group, criterion_main, Criterion};
use ml_rs::classification::linear::{BFGSSolver, IRLSSolver, LogisticRegression};
use ml_rs::classification::Classifier;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn bench_logistic_regression(c: &mut Criterion) {
    c.bench_function("BFGS Logistic Regression (large random)", |b| {
        b.iter(|| {
            let solver = BFGSSolver::default();
            let mut clf = LogisticRegression::new(solver);
            let n_rows = 20000;
            let n_features = 5;
            let x = Array2::random((n_rows, n_features), Uniform::new(-1.0, 1.0));
            let y = Array1::random(n_rows, Uniform::new(0, 2));
            clf.fit(x.view(), y.view()).unwrap();
        });
    });

    c.bench_function("IRLS Logistic Regression (large random)", |b| {
        b.iter(|| {
            let solver = IRLSSolver::default();
            let mut clf = LogisticRegression::new(solver);
            let n_rows = 20000;
            let n_features = 5;
            let x = Array2::random((n_rows, n_features), Uniform::new(-1.0, 1.0));
            let y = Array1::random(n_rows, Uniform::new(0, 2));
            clf.fit(x.view(), y.view()).unwrap();
        });
    });
}

criterion_group!(benches, bench_logistic_regression);
criterion_main!(benches);
