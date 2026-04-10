use faer::linalg::solvers::{DenseSolveCore, Solve};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};

struct LogitResult {
    coefficients: faer::Mat<f64>,
    p_values: Vec<f64>,
    aic: f64,
    bic: f64,
    log_likelihood: f64,
    converged: bool,
}

fn fit_logit_internal(
    x_faer: faer::mat::MatRef<'_, f64>,
    y_faer: faer::mat::MatRef<'_, f64>,
    max_iter: usize,
    tol: f64,
) -> Result<LogitResult, String> {
    let n = x_faer.nrows();
    let k = x_faer.ncols();

    let mut beta = faer::Mat::<f64>::zeros(k, 1);
    let mut log_likelihood = f64::NEG_INFINITY;
    let mut converged = false;

    for _ in 0..max_iter {
        let x_beta = x_faer * &beta;
        let p = faer::Mat::<f64>::from_fn(n, 1, |row, _| {
            let value = x_beta[(row, 0)];
            1.0_f64 / (1.0_f64 + (-value).exp())
        });

        let mut current_ll = 0.0_f64;
        for row in 0..n {
            let probability = p[(row, 0)].clamp(1e-15, 1.0 - 1e-15);
            current_ll += y_faer[(row, 0)] * probability.ln()
                + (1.0_f64 - y_faer[(row, 0)]) * (1.0_f64 - probability).ln();
        }

        let mut err = faer::Mat::<f64>::zeros(n, 1);
        for row in 0..n {
            err[(row, 0)] = y_faer[(row, 0)] - p[(row, 0)];
        }
        let gradient = x_faer.transpose() * &err;

        let mut x_weighted = faer::Mat::<f64>::zeros(n, k);
        for row in 0..n {
            let probability = p[(row, 0)];
            let weight = probability * (1.0 - probability);
            for col in 0..k {
                x_weighted[(row, col)] = x_faer[(row, col)] * weight;
            }
        }
        let hessian = x_faer.transpose() * &x_weighted;

        let solve_res = hessian.full_piv_lu().solve(&gradient);
        beta += &solve_res;

        if (current_ll - log_likelihood).abs() < tol {
            log_likelihood = current_ll;
            converged = true;
            break;
        }
        log_likelihood = current_ll;
    }

    let x_beta = x_faer * &beta;
    let p = faer::Mat::<f64>::from_fn(n, 1, |row, _| {
        let value = x_beta[(row, 0)];
        1.0_f64 / (1.0_f64 + (-value).exp())
    });

    let mut x_weighted = faer::Mat::<f64>::zeros(n, k);
    for row in 0..n {
        let probability = p[(row, 0)];
        let weight = probability * (1.0 - probability);
        for col in 0..k {
            x_weighted[(row, col)] = x_faer[(row, col)] * weight;
        }
    }
    let hessian = x_faer.transpose() * &x_weighted;
    let inv_hessian = hessian.full_piv_lu().inverse();

    let mut p_values = Vec::with_capacity(k);
    let normal = Normal::new(0.0, 1.0).map_err(|err| err.to_string())?;

    for idx in 0..k {
        let variance = inv_hessian[(idx, idx)];
        let p_value = if variance > 0.0 {
            let standard_error = variance.sqrt();
            let coefficient = beta[(idx, 0)];
            let z_score = coefficient / standard_error;
            2.0 * (1.0 - normal.cdf(z_score.abs()))
        } else {
            f64::NAN
        };
        p_values.push(p_value);
    }

    let aic = 2.0 * (k as f64) - 2.0 * log_likelihood;
    let bic = (n as f64).ln() * (k as f64) - 2.0 * log_likelihood;

    Ok(LogitResult {
        coefficients: beta,
        p_values,
        aic,
        bic,
        log_likelihood,
        converged,
    })
}

#[pyfunction]
#[pyo3(signature = (x, y, max_iter=25, tol=1e-6))]
fn fit_logistic_regression_numpy(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<PyObject> {
    let x_view = x.as_array();
    let y_view = y.as_array();

    let n = x_view.nrows();
    let k = x_view.ncols();

    if n != y_view.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "X and y must have same number of rows.",
        ));
    }

    let x_faer = faer::Mat::<f64>::from_fn(n, k, |row, col| x_view[(row, col)]);
    let y_faer = faer::Mat::<f64>::from_fn(n, 1, |row, _| y_view[row]);

    match fit_logit_internal(x_faer.as_ref(), y_faer.as_ref(), max_iter, tol) {
        Ok(result) => {
            let dict = PyDict::new(py);
            let mut beta_flat = Vec::with_capacity(k);
            for idx in 0..k {
                beta_flat.push(result.coefficients[(idx, 0)]);
            }

            dict.set_item("coefficients", beta_flat.into_pyobject(py)?)?;
            dict.set_item("p_values", result.p_values)?;
            dict.set_item("aic", result.aic)?;
            dict.set_item("bic", result.bic)?;
            dict.set_item("log_likelihood", result.log_likelihood)?;
            dict.set_item("converged", result.converged)?;

            Ok(dict.into_any().unbind())
        }
        Err(err) => Err(pyo3::exceptions::PyRuntimeError::new_err(err)),
    }
}

#[pyfunction]
#[pyo3(signature = (fixed_x, candidate_features, y, max_iter=25, tol=1e-6))]
fn batch_fit_logistic_regression_numpy(
    py: Python<'_>,
    fixed_x: PyReadonlyArray2<f64>,
    candidate_features: Vec<PyReadonlyArray1<f64>>,
    y: PyReadonlyArray1<f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Vec<PyObject>> {
    let fixed_x_view = fixed_x.as_array();
    let y_view = y.as_array();
    let n = fixed_x_view.nrows();
    let k_fixed = fixed_x_view.ncols();

    if y_view.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fixed_x and y must have the same number of rows.",
        ));
    }

    let y_faer = faer::Mat::<f64>::from_fn(n, 1, |row, _| y_view[row]);

    let candidate_vecs: Vec<Vec<f64>> = candidate_features
        .iter()
        .map(|feature| {
            feature
                .as_slice()
                .map(|slice| slice.to_vec())
                .map_err(|err| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Candidate not contiguous: {}",
                        err
                    ))
                })
        })
        .collect::<PyResult<Vec<_>>>()?;

    for candidate_vec in &candidate_vecs {
        if candidate_vec.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Each candidate feature must match the number of rows in fixed_x.",
            ));
        }
    }

    let results: Vec<LogitResult> = py.allow_threads(|| {
        candidate_vecs
            .par_iter()
            .map(|candidate_vec| {
                let x_combined = faer::Mat::<f64>::from_fn(n, k_fixed + 1, |row, col| {
                    if col < k_fixed {
                        fixed_x_view[(row, col)]
                    } else {
                        candidate_vec[row]
                    }
                });

                match fit_logit_internal(x_combined.as_ref(), y_faer.as_ref(), max_iter, tol) {
                    Ok(result) => result,
                    Err(_) => LogitResult {
                        coefficients: faer::Mat::zeros(k_fixed + 1, 1),
                        p_values: vec![1.0; k_fixed + 1],
                        aic: f64::INFINITY,
                        bic: f64::INFINITY,
                        log_likelihood: f64::NEG_INFINITY,
                        converged: false,
                    },
                }
            })
            .collect()
    });

    let mut py_results = Vec::with_capacity(results.len());
    for result in results {
        let dict = PyDict::new(py);
        dict.set_item("p_value", result.p_values.last().copied().unwrap_or(1.0))?;
        dict.set_item("aic", result.aic)?;
        dict.set_item("bic", result.bic)?;
        dict.set_item("converged", result.converged)?;
        py_results.push(dict.into_any().unbind());
    }

    Ok(py_results)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(pyo3::wrap_pyfunction!(
        fit_logistic_regression_numpy,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        batch_fit_logistic_regression_numpy,
        module
    )?)?;
    Ok(())
}
