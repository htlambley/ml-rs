use ndarray::{Array2, ArrayView2};

pub mod pca;

pub trait Transformer {
    fn fit(&mut self, x: ArrayView2<f64>);
    fn transform(&self, x: ArrayView2<f64>) -> Array2<f64>;
    fn fit_transform(&mut self, x: ArrayView2<f64>) -> Array2<f64> {
        self.fit(x);
        self.transform(x)
    }
}
