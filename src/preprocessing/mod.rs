use csv::ReaderBuilder;
use ndarray::Array2;
use ndarray_csv::{Array2Reader, ReadError};
use serde::Deserialize;
use std::fs::File;

/// Loads data from a CSV file to an `Array2<f64>` for use in transformers
/// and models.
///
/// This assumes that the CSV file you wish to load from is a homogeneous 
/// table of float values that can be interpreted as a matrix of `f64`.
///
/// # Notes
/// This can be considered a very rough equivalent to the `pandas` function
/// `read_csv`, with the added restriction that, as we only support homogeneous
/// arrays, all values must be float, and no column names are supported. 
///
/// # Examples
/// ```no_run
/// use std::fs::File;
/// use ml_rs::preprocessing::CsvReader;
/// // Load the CSV file as a `File`.
/// let csv_file = File::open("test.csv").unwrap();
/// // Create a `CsvReader` from the file reference.
/// let reader = CsvReader::new(&csv_file);
/// // The expected number of rows and columns must be specified when reading 
/// // from CSV.
/// let n_rows = 3;
/// let n_cols = 4;
/// // Read `csv_file` to an `Array2<f64>`.
/// let data: Array2<f64> = reader.read(3, 4);
/// ```
pub struct CsvReader<'a> {
    file: &'a File,
    has_headers: bool,
}

impl<'a> CsvReader<'a> {
    /// Create a new `CsvReader` from the file reference passed. The file
    /// will not be consumed as we only read from the CSV file.
    pub fn new(file: &File) -> CsvReader {
        CsvReader {
            file,
            has_headers: false,
        }
    }

    /// Reads the CSV file to an `Array2<f64>`. 
    ///
    /// # Panics
    /// Panics if the CSV is not valid, or contains an incorrect number of
    /// rows or columns.
    pub fn read<T: Sized + for<'de> Deserialize<'de>>(
        &self,
        rows: usize,
        columns: usize,
    ) -> Array2<T> {
        let mut reader = ReaderBuilder::new()
            .has_headers(self.has_headers)
            .from_reader(self.file);
        match reader.deserialize_array2((rows, columns)) {
            Ok(array) => array,
            Err(e) => match e {
                ReadError::Csv(e) => panic!("{:?}", e),
                ReadError::NRows { expected, actual } => panic!(
                    "Incorrect number of rows when reading from CSV. Expected {}, but got {}",
                    expected, actual
                ),
                ReadError::NColumns {
                    expected,
                    actual,
                    at_row_index: _,
                } => panic!(
                    "Incorrect number of rows when reading from CSV. Expected {}, but got {}",
                    expected, actual
                ),
            },
        }
    }
}
