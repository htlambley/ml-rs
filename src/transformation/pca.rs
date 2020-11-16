use super::Transformer;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::svd::SVD;

fn column_means(x: ArrayView2<f64>) -> Array1<f64> {
    let n: f64 = x.nrows() as f64;
    x.axis_iter(Axis(1)).map(|col| col.sum() / n).collect()
}

fn subtract_column_means(x: &mut Array2<f64>, means: ArrayView1<f64>) {
    assert!(x.ncols() == means.len());
    x.axis_iter_mut(Axis(1))
        .zip(means.iter())
        .for_each(|(mut col, mean)| col.iter_mut().for_each(|x| *x -= mean))
}

/// A transformer that can perform principal component analysis (PCA) on a data
/// matrix.
///
/// Principal component analysis can be used as a dimensionality reduction
/// technique. It operates by performing an orthonormal change of basis, and
/// optionally we can keep only the first $k$ basis vectors. This has the
/// effect of preserving the "most significant" axes of the data while
/// transitioning into a lower dimensional space.
///
/// # Examples
/// Performing dimensionality reduction on data matrix with 5 features into the
/// two principal components.
/// ```no_run
/// use ml_rs::transformation::Transformer;
/// use ml_rs::transformation::pca::PrincipalComponentAnalysis;
/// use ndarray::array;
///
/// // Create a data matrix and a PCA transformer to transform into two dimensions.
/// let n_dimensions = 2;
/// let x = array![[1.0, 2.0, 3.0, -1.0, 4.0], [-1.0, 3.0, 5.0, 4.0, 0.0], [0.0, -3.0, 2.0, 1.0, 0.1]];
/// let mut pca = PrincipalComponentAnalysis::new(n_dimensions);
/// pca.fit(x.view());
///
/// // Test the PCA transformation on some previously unseen data.
/// let x_new = array![[0.0, -5.0, 3.0, 2.0, 7.0]];
/// let x_transformed = pca.transform(x_new.view());
/// ```
#[derive(Clone)]
pub struct PrincipalComponentAnalysis {
    n_components: usize,
    principal_components: Option<Array2<f64>>,
    means: Option<Array1<f64>>,
}

impl PrincipalComponentAnalysis {
    pub fn new(n_components: usize) -> PrincipalComponentAnalysis {
        assert!(n_components > 0);
        PrincipalComponentAnalysis {
            n_components,
            principal_components: None,
            means: None,
        }
    }
}

impl Transformer for PrincipalComponentAnalysis {
    fn fit(&mut self, x: ArrayView2<f64>) {
        let mut x = x.to_owned();
        let means = column_means(x.view());
        subtract_column_means(&mut x, means.view());

        // Using the singular value decomposition $A = U \Sigma V^T$, we can
        // obtain the PCA of a data matrix X by the expression $XV$. This
        // internally calls LAPACK's dgesvd routine so we choose to only
        // calculate $\Sigma$ and $V^T$.
        // NOTE: ndarray-linalg 0.12.1 has been causing errors calling dgesvd
        // so we have to build with the GitHub main branch of ndarray-linalg.
        let (_, _, v_t) = x
            .svd(false, true)
            .expect("`PrinicpalComponentAnalysis` failed to calculate the SVD.");
        let v_t = v_t.unwrap();
        let v = v_t.t();
        // Keep only the first `self.n_components` columns of $V$, which are
        // known as the principal components.
        let v_k = v.slice(s![.., ..self.n_components]);
        self.principal_components = Some(v_k.to_owned());
        self.means = Some(means);
    }

    fn transform(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let mut x = x.to_owned();
        if let Some(means) = &self.means {
            subtract_column_means(&mut x, means.view());
        } else {
            panic!("`PrincipalComponentAnalysis` transformer must be fit before transforming. Use `transformer.fit(x)` to fit.");
        }

        if let Some(principal_components) = &self.principal_components {
            x.dot(principal_components)
        } else {
            panic!("`PrincipalComponentAnalysis` transformer must be fit before transforming. Use `transformer.fit(x)` to fit.");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::Transformer;
    use super::PrincipalComponentAnalysis;
    use super::{column_means, subtract_column_means};
    use ndarray::{array, Array2};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_centre_data() {
        let mut x = array![[-10., -2.], [10., -4.], [12., 0.]];
        let means = column_means(x.view());
        subtract_column_means(&mut x, means.view());
        assert_eq!(x, array![[-14., 0.], [6., -2.], [8., 2.]]);
    }

    #[test]
    fn test_precentred_data() {
        let x = array![[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [-1.0, -1.0, 2.0]];
        let mut x_copy = x.clone();
        let means = column_means(x_copy.view());
        subtract_column_means(&mut x_copy, means.view());
        assert_eq!(x, x_copy);
    }

    #[test]
    fn test_fit_pca() {
        let x = array![[1.0, 2.0], [3.0, 1.0], [5.0, 3.0]];
        let mut pca = PrincipalComponentAnalysis::new(2);
        pca.fit(x.view());
        let x_transformed = pca.transform(x.view());
        // Test against precomputed values (generated using scikit-learn's PCA)
        assert_eq!(x_transformed.nrows(), 3);
        assert_eq!(x_transformed.ncols(), 2);
        assert!((x_transformed[[0, 0]] - 1.91418405).abs() < 1e-5);
        assert!((x_transformed[[0, 1]] - 0.5795683).abs() < 1e-5);
        assert!((x_transformed[[1, 0]] - 0.28978415).abs() < 1e-5);
        assert!((x_transformed[[1, 1]] + 0.95709203).abs() < 1e-5);
        assert!((x_transformed[[2, 0]] + 2.2039682).abs() < 1e-5);
        assert!((x_transformed[[2, 1]] - 0.37752373).abs() < 1e-5);
    }

    #[test]
    fn test_pca_dimensionality_reduction() {
        let mut pca = PrincipalComponentAnalysis::new(1);
        let x = array![[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]];
        pca.fit(x.view());
        let x_transformed = pca.transform(x.view());
        assert_eq!(x_transformed.ncols(), 1);
        assert_eq!(x_transformed.nrows(), 3);
        assert!((x_transformed[[0, 0]] - 18.13224251).abs() < 1e-5);
        assert!((x_transformed[[1, 0]] - 3.73524997).abs() < 1e-5);
        assert!((x_transformed[[2, 0]] + 21.86749248).abs() < 1e-5);
    }

    #[test]
    fn test_pca_unseen_data() {
        let mut pca = PrincipalComponentAnalysis::new(1);
        let x = array![[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]];
        pca.fit(x.view());
        let x_test = array![[3.0, -1.0]];
        let x_transformed = pca.transform(x_test.view());
        assert_eq!(x_transformed.ncols(), 1);
        assert_eq!(x_transformed.nrows(), 1);
        assert!((x_transformed[[0, 0]] - 20.91547962).abs() < 1e-5);
    }

    #[test]
    fn test_fit_pca_random_large() {
        let mut pca = PrincipalComponentAnalysis::new(20);
        let n_rows = 10000;
        let n_features = 50;
        let x = Array2::random((n_rows, n_features), Uniform::new(-1.0, 1.0));
        pca.fit(x.view());
    }

    #[test]
    #[should_panic(
        expected = "`PrincipalComponentAnalysis` transformer must be fit before transforming. Use `transformer.fit(x)` to fit."
    )]
    fn test_pca_unfit() {
        let pca = PrincipalComponentAnalysis::new(1);
        let x = array![[1.0, 0.0]];
        pca.transform(x.view());
    }
}
