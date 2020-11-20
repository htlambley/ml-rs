use criterion::{criterion_group, criterion_main, Criterion};
use ml_rs::transformation::pca::PrincipalComponentAnalysis;
use ml_rs::transformation::Transformer;
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn bench_pca(c: &mut Criterion) {
    c.bench_function("PCA (large random)", |b| {
        b.iter(|| {
            let mut pca = PrincipalComponentAnalysis::new(20);
            let n_rows = 100000;
            let n_features = 50;
            let x = Array2::random((n_rows, n_features), Uniform::new(-1.0, 1.0));
            pca.fit(x.view()).unwrap();
        });
    });
}

criterion_group!(benches, bench_pca);
criterion_main!(benches);
