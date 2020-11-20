# ml-rs
Machine learning library in Rust built on 
[`ndarray`](https://github.com/rust-ndarray/ndarray) and 
[`argmin`](http://argmin-rs.org/).

[**Documentation**](https://htlambley.github.io/ml-rs/ml_rs/)

## Features
- **Classification**
  - Logistic Regression
- **Regression**
  - Linear Regression (Ordinary Least Squares)
- **Classification and Regression Metrics**
  - Accuracy
  - Precision and recall
- **Data Preprocessing**
  - Loading from CSV
- **Transformations**
  - Principal Component Analysis (PCA)

## Supported Platforms & Setup
Currently, `ml-rs` has been tested on Linux (Debian). To use the library, a 
backend for `ndarray_linalg` is required in order to perform the linear algebra
needed. Three options are supported:
- OpenBLAS
- Netlib
- Intel MKL. 

The library has been most thoroughly tested with the OpenBLAS backend. On 
Debian, to set up OpenBLAS, run:
```shell
apt install libopenblas-base libopenblas-dev liblapacke-dev
```
Then, add the following to your binary's `Cargo.toml`:
```toml
openblas-src = { version = "0.7", features = ["system"] }
```
This will link to your system's OpenBLAS library.
