use ndarray::{Array1, ArrayView1};

pub trait Solver {
    fn step(&self, current_x: ArrayView1<f64>) -> Array1<f64>;
    fn step_length(&self) -> f64;
    fn within_tolerance(&self, x_k: ArrayView1<f64>, current_x: ArrayView1<f64>) -> bool;
    fn max_iterations(&self) -> usize;
    fn solve(&mut self, x_0: Array1<f64>) -> Array1<f64> {
        let mut current_x = x_0;
        let mut iterations = 0;
        loop {
            let x_k = self.step(current_x.view());
            
            if self.within_tolerance(x_k.view(), current_x.view()) {
                return x_k;
            }

            current_x = x_k;
            iterations += 1;
            if iterations > self.max_iterations() {
                // TODO: don't panic here, instead return a result, and update doc on this point
                panic!("Solver did not converge within {} iterations. Terminating.", self.max_iterations())
            }
        }
    }
}


/// A simple solver implementing gradient descent to find the minimum of a function.
/// # Method
/// Given an initial guess `x_0`, a function f, and its gradient \nabla f, the 
/// gradient descent algorithm iterates upon the relation:
/// `x_{k + 1} = x_k - \alpha \nabla f(x_k)`
/// for some fixed step length \alpha. 
/// 
/// # Usage
/// To use the `GradientDescent` solver, you must provide a function to 
/// calculate the value of f and its gradient. Using 
/// `GradientDescent::step_length` and `GradientDescent::epsilon``, you can 
/// configure the value \alpha and the criterion used to terminate: the 
/// algorithm stops when |f(x_{k + 1}) - f(x_k)| < epsilon.
///
/// The solver will converge more slowly in general for smaller step length
/// ("learning rate") but excessive values may lead to overshooting the minimum
/// and oscillating between sub-optimal points.
///
/// The solver will perform a fixed number of iterations and panic if 
/// convergence does not occur by the maximum iteration. For non-convex f,
/// unique solutions are not guaranteed, and there are many functions for which
/// gradient descent makes no sense (e.g. f(x) = x).
///
/// # Examples
/// Minimising f(x, y) = x^2 + y^2 + 1. The true minimum is (0, 0).
/// ```
/// use ml_rs::solvers::{Solver, GradientDescent};
/// use ndarray::{Array1, ArrayView1, array};
/// fn f(x: ArrayView1<f64>) -> f64 {
///     x.iter().fold(0.0, |total, x_i| total + x_i * x_i) + 1.0
/// }
///
/// fn gradient(x: ArrayView1<f64>) -> Array1<f64> {
///    2.0 * x.to_owned()
/// }
///
/// let x_0 = array![1.0, -1.0];
/// let mut solver = GradientDescent::new(f, gradient);
/// let solution = solver.solve(x_0);
/// ```
pub struct GradientDescent<F, G> where F: Fn(ArrayView1<f64>) -> f64, G: Fn(ArrayView1<f64>) -> Array1<f64> {
    f: F,
    gradient: G,
    step_length: f64,
    epsilon: f64,
    // TODO: allow configuration of max_iter
    max_iter: usize
}

impl<F, G> GradientDescent<F, G> where F: Fn(ArrayView1<f64>) -> f64, G: Fn(ArrayView1<f64>) -> Array1<f64> {
    pub fn new(f: F, gradient: G) -> GradientDescent<F, G> {
        GradientDescent {
            f, 
            gradient,
            step_length: 0.1,
            epsilon: 1e-5,
            max_iter: 10000
        }
    }

    pub fn step_length(&mut self, length: f64) { 
        assert!(length > 0.0, "Step length for gradient descent must be strictly positive. Try `step_length(eps)` for some `eps > 0.0`.");
        self.step_length = length;
    }

    pub fn epsilon(&mut self, epsilon: f64) {
        assert!(epsilon > 0.0, "Stopping criterion for gradient descent must be strictly positive. Try `epsilon(eps)` for some `eps > 0.0`.");
        self.epsilon = epsilon;
    }
}

impl<F, G> Solver for GradientDescent<F, G> where F: Fn(ArrayView1<f64>) -> f64, G: Fn(ArrayView1<f64>) -> Array1<f64> {
    fn step(&self, current_x: ArrayView1<f64>) -> Array1<f64> {
        current_x.to_owned() - self.step_length * (self.gradient)(current_x.view())
    }

    fn step_length(&self) -> f64 {
        self.step_length
    }

    fn within_tolerance(&self, current_x: ArrayView1<f64>, x_k: ArrayView1<f64>) -> bool {
        ((self.f)(x_k) - (self.f)(current_x)).abs() < self.epsilon
    }
    
    fn max_iterations(&self) -> usize {
        self.max_iter
    }
}

#[cfg(test)]
mod test {
    use super::{Solver, GradientDescent};
    use ndarray::{Array1, ArrayView1, array};
    use ndarray_linalg::norm::Norm;

    #[test]
    fn test_gradient_descent_1d() {
        fn f(x: ArrayView1<f64>) -> f64 {
            x.iter().fold(0.0, |total, x_i| total + x_i * x_i)
        }
    
        fn g(x: ArrayView1<f64>) -> Array1<f64> {
            2.0 * x.to_owned()
        }

        let mut solver = GradientDescent::new(f, g);
        let solution = solver.solve(array![1.]);
        assert!((solution - array![0.0]).norm_l2() < 0.01)
    }

    #[test]
    #[should_panic(expected = "Solver did not converge within 10000 iterations. Terminating.")]
    fn test_gradient_descent_1d_unbounded() {
        fn f(x: ArrayView1<f64>) -> f64 {
            x[0]
        }
    
        fn g(x: ArrayView1<f64>) -> Array1<f64> {
            array![1.0]
        }

        let mut solver = GradientDescent::new(f, g);
        solver.solve(array![1.]);
    }

    #[test]
    #[should_panic(expected = "Step length for gradient descent must be strictly positive. Try `step_length(eps)` for some `eps > 0.0`.")]
    fn test_gradient_descent_set_step() {
        fn f(x: ArrayView1<f64>) -> f64 {
            x[0]
        }
    
        fn g(x: ArrayView1<f64>) -> Array1<f64> {
            array![1.0]
        }

        let mut solver = GradientDescent::new(f, g);
        GradientDescent::step_length(&mut solver, -1.0);
    }

    #[test]
    #[should_panic(expected = "Stopping criterion for gradient descent must be strictly positive. Try `epsilon(eps)` for some `eps > 0.0`.")]
    fn test_gradient_descent_set_eps() {
        fn f(x: ArrayView1<f64>) -> f64 {
            x[0]
        }
    
        fn g(x: ArrayView1<f64>) -> Array1<f64> {
            array![1.0]
        }

        let mut solver = GradientDescent::new(f, g);
        GradientDescent::epsilon(&mut solver, -1.0);
    }

    #[test]
    fn test_gradient_descent_2d() {
        fn f(x: ArrayView1<f64>) -> f64 {
            x.iter().fold(0.0, |total, x_i| total + x_i * x_i)
        }
    
        fn g(x: ArrayView1<f64>) -> Array1<f64> {
            2.0 * x.to_owned()
        }

        let x_0 = array![1.0, -1.0];
        let mut solver = GradientDescent::new(f, g);
        let solution = solver.solve(x_0);
        assert!((solution - array![0.0, 0.0]).norm_l2() < 0.01);
    }
}