use std::fs::File;
use csv::ReaderBuilder;
use ndarray_csv::{Array2Reader, ReadError};
use ndarray::Array2;
use serde::Deserialize;

pub struct CsvReader<'a> {
    file: &'a File,
    has_headers: bool
}

impl<'a> CsvReader<'a> {
    pub fn new(file: &File) -> CsvReader {
        CsvReader {
            file,
            has_headers: false
        }
    }

    pub fn read<T: Sized + for<'de> Deserialize<'de>>(&self, rows: usize, columns: usize) -> Array2<T> {
        let mut reader = ReaderBuilder::new()
            .has_headers(self.has_headers)
            .from_reader(self.file);
        match reader.deserialize_array2((rows, columns)) {
            Ok(array) => array,
            Err(e) => match e {
                ReadError::Csv(e) => panic!("{:?}", e),
                ReadError::NRows {expected, actual} => panic!("Incorrect number of rows when reading from CSV. Expected {}, but got {}", expected, actual),
                ReadError::NColumns {expected, actual, at_row_index: _} => panic!("Incorrect number of rows when reading from CSV. Expected {}, but got {}", expected, actual)    
            }
        }
    }
}