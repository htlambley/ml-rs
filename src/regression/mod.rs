use ndarray::{Array1, ArrayView1, ArrayView2};

pub mod linear;

pub trait Regressor {
    fn fit<'a>(&mut self, x: ArrayView2<'a, f64>, y: ArrayView1<'a, f64>);
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64>;
}
