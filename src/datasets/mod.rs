use csv::ReaderBuilder;
use ndarray::{Array1, Array2};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Record {
    clump_thickness: f64,
    cell_size_uniformity: f64,
    cell_shape_uniformity: f64,
    marginal_adhesion: f64,
    single_epi_cell_size: f64,
    bare_nuclei: f64,
    bland_chromatin: f64,
    normal_nucleoli: f64,
    mitoses: f64,
    class: usize,
}

/// Loads the *Breast Cancer Wisconsin (Diagnostic) Data Set* from the
/// Center for Machine Learning and Intelligent Systems at UC Irvine.
///
/// The original source of the dataset is linked [here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).
/// Some preprocessing has been done to remove records with missing data, and
/// the ID number for each sample, but otherwise this is an unedited version of
/// the `breast-cancer-wisconsin.data` file provided by UC Irvine.
///
/// # Returns
/// A 683 x 9 data matrix is returned as an `Array2<f64>`, with the columns
/// corresponding to the following features:
///
/// 1. Clump Thickness
/// 2. Uniformity of Cell Size      
/// 3. Uniformity of Cell Shape     
/// 4. Marginal Adhesion             
/// 5. Single Epithelial Cell Size   
/// 6. Bare Nuclei                 
/// 7. Bland Chromatin            
/// 8. Normal Nucleoli            
/// 9. Mitoses                     
///
/// All features will have integer values between 1.0 and 10.0 inclusive.
///
/// The label array `y` is returned in an `Array1<usize>` with class 0
/// indicating a benign tumour, and class 1 indicating a malignancy.
///
/// **Class balance**: 444 positive (65%), 239 negative (35%).
///
/// # Examples
/// ```
/// use ml_rs::datasets::load_breast_cancer;
/// let (x, y) = load_breast_cancer();
/// ```
///
/// # References
/// This breast cancer databases was obtained from the University of Wisconsin
/// Hospitals, Madison from Dr. William H. Wolberg. See, for example, the
/// following reference.
///
/// O. L. Mangasarian and W. H. Wolberg, *Cancer diagnosis via linear
/// programming*, SIAM News 23.5, September 1990, pp 1 & 18.
pub fn load_breast_cancer() -> (Array2<f64>, Array1<usize>) {
    let dataset = include_str!("breast-cancer-wisconsin.csv");
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(dataset.as_bytes());
    let n_rows = 683;
    let n_cols = 9;
    let mut x = Array2::zeros((n_rows, n_cols));
    let mut y = Array1::zeros(n_rows);
    for (i, result) in reader.deserialize().enumerate() {
        let record: Record = result.unwrap();
        x[[i, 0]] = record.clump_thickness;
        x[[i, 1]] = record.cell_size_uniformity;
        x[[i, 2]] = record.cell_shape_uniformity;
        x[[i, 3]] = record.marginal_adhesion;
        x[[i, 4]] = record.single_epi_cell_size;
        x[[i, 5]] = record.bare_nuclei;
        x[[i, 6]] = record.bland_chromatin;
        x[[i, 7]] = record.normal_nucleoli;
        x[[i, 8]] = record.mitoses;
        y[i] = record.class;
    }

    (x, y)
}

#[cfg(test)]
mod tests {
    use super::load_breast_cancer;

    #[test]
    fn test_breast_cancer_class_balance() {
        let (_, y) = load_breast_cancer();
        let positive_cases = y.sum();
        assert_eq!(positive_cases, 444);
        assert_eq!(y.len(), 683);
    }

    #[test]
    fn test_breast_cancer_vals() {
        let (x, y) = load_breast_cancer();
        x.iter().for_each(|x| assert!(*x != 0.0));
        y.iter().for_each(|y| assert!(*y == 0 || *y == 1));
    }
}
