use faer::linalg::solvers::{DenseSolveCore, Solve};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};
use std::panic::{catch_unwind, AssertUnwindSafe};

struct LogitResult {
    coefficients: faer::Mat<f64>,
    p_values: Vec<f64>,
    aic: f64,
    bic: f64,
    log_likelihood: f64,
    converged: bool,
}

fn mat_all_finite(matrix: faer::mat::MatRef<'_, f64>) -> bool {
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            if !matrix[(row, col)].is_finite() {
                return false;
            }
        }
    }
    true
}

fn lu_is_effectively_singular(lu: &faer::linalg::solvers::FullPivLu<f64>) -> bool {
    let upper = lu.U();
    let dim = upper.nrows().min(upper.ncols());
    if dim == 0 {
        return true;
    }

    let mut max_diag = 0.0_f64;
    for idx in 0..dim {
        let diag = upper[(idx, idx)].abs();
        if !diag.is_finite() {
            return true;
        }
        if diag > max_diag {
            max_diag = diag;
        }
    }

    if max_diag == 0.0 {
        return true;
    }

    let tolerance = 1e-12_f64 * max_diag.max(1.0);
    for idx in 0..dim {
        if upper[(idx, idx)].abs() <= tolerance {
            return true;
        }
    }

    false
}

fn safe_full_piv_solve(
    hessian: faer::mat::MatRef<'_, f64>,
    gradient: faer::mat::MatRef<'_, f64>,
) -> Result<faer::Mat<f64>, String> {
    catch_unwind(AssertUnwindSafe(|| {
        let lu = hessian.full_piv_lu();
        if lu_is_effectively_singular(&lu) {
            return Err("logistic regression hessian is singular".to_string());
        }

        let solution = lu.solve(gradient);
        if !mat_all_finite(solution.as_ref()) {
            return Err("logistic regression solve produced non-finite values".to_string());
        }

        Ok(solution)
    }))
    .map_err(|_| "logistic regression linear solve panicked".to_string())?
}

fn safe_full_piv_inverse(hessian: faer::mat::MatRef<'_, f64>) -> Result<faer::Mat<f64>, String> {
    catch_unwind(AssertUnwindSafe(|| {
        let lu = hessian.full_piv_lu();
        if lu_is_effectively_singular(&lu) {
            return Err("logistic regression hessian is singular".to_string());
        }

        let inverse = lu.inverse();
        if !mat_all_finite(inverse.as_ref()) {
            return Err("logistic regression inverse produced non-finite values".to_string());
        }

        Ok(inverse)
    }))
    .map_err(|_| "logistic regression inverse panicked".to_string())?
}

fn fit_logit_internal(
    x_faer: faer::mat::MatRef<'_, f64>,
    y_faer: faer::mat::MatRef<'_, f64>,
    max_iter: usize,
    tol: f64,
) -> Result<LogitResult, String> {
    let n = x_faer.nrows();
    let k = x_faer.ncols();

    if !mat_all_finite(x_faer) || !mat_all_finite(y_faer) {
        return Err("logistic regression input contains NaN or inf".to_string());
    }

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
        if !current_ll.is_finite() {
            return Err("logistic regression produced non-finite log-likelihood".to_string());
        }

        let mut err = faer::Mat::<f64>::zeros(n, 1);
        for row in 0..n {
            err[(row, 0)] = y_faer[(row, 0)] - p[(row, 0)];
        }
        let gradient = x_faer.transpose() * &err;
        if !mat_all_finite(gradient.as_ref()) {
            return Err("logistic regression produced non-finite gradient".to_string());
        }

        let mut x_weighted = faer::Mat::<f64>::zeros(n, k);
        for row in 0..n {
            let probability = p[(row, 0)];
            let weight = probability * (1.0 - probability);
            for col in 0..k {
                x_weighted[(row, col)] = x_faer[(row, col)] * weight;
            }
        }
        let hessian = x_faer.transpose() * &x_weighted;
        if !mat_all_finite(hessian.as_ref()) {
            return Err("logistic regression produced non-finite hessian".to_string());
        }

        let solve_res = safe_full_piv_solve(hessian.as_ref(), gradient.as_ref())?;
        beta += &solve_res;
        if !mat_all_finite(beta.as_ref()) {
            return Err("logistic regression produced non-finite coefficients".to_string());
        }

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
    if !mat_all_finite(p.as_ref()) {
        return Err("logistic regression produced non-finite probabilities".to_string());
    }

    let mut x_weighted = faer::Mat::<f64>::zeros(n, k);
    for row in 0..n {
        let probability = p[(row, 0)];
        let weight = probability * (1.0 - probability);
        for col in 0..k {
            x_weighted[(row, col)] = x_faer[(row, col)] * weight;
        }
    }
    let hessian = x_faer.transpose() * &x_weighted;
    if !mat_all_finite(hessian.as_ref()) {
        return Err("logistic regression produced non-finite hessian".to_string());
    }
    let inv_hessian = safe_full_piv_inverse(hessian.as_ref())?;

    let mut p_values = Vec::with_capacity(k);
    let normal = Normal::new(0.0, 1.0).map_err(|err| err.to_string())?;

    for idx in 0..k {
        let variance = inv_hessian[(idx, idx)];
        if !variance.is_finite() || variance <= 0.0 {
            return Err("logistic regression hessian inverse is not positive definite".to_string());
        }

        let standard_error = variance.sqrt();
        let coefficient = beta[(idx, 0)];
        if !coefficient.is_finite() || !standard_error.is_finite() {
            return Err("logistic regression diagnostics are non-finite".to_string());
        }
        let z_score = coefficient / standard_error;
        let p_value = 2.0 * (1.0 - normal.cdf(z_score.abs()));
        if !p_value.is_finite() {
            return Err("logistic regression produced non-finite p-values".to_string());
        }
        p_values.push(p_value);
    }

    let aic = 2.0 * (k as f64) - 2.0 * log_likelihood;
    let bic = (n as f64).ln() * (k as f64) - 2.0 * log_likelihood;
    if !aic.is_finite() || !bic.is_finite() || !log_likelihood.is_finite() {
        return Err("logistic regression information criteria are non-finite".to_string());
    }

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
        let p_value = if result.converged {
            result.p_values.last().copied().unwrap_or(1.0)
        } else {
            1.0
        };
        dict.set_item("p_value", p_value)?;
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
