use std::cmp::Ordering;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;

fn calculate_chi2_yates(n1: f64, e1: f64, n2: f64, e2: f64) -> f64 {
    let total_n = n1 + n2;
    let total_e = e1 + e2;
    let total_ne = total_n - total_e;

    if total_n == 0.0 {
        return 0.0;
    }

    let e1_expected = (n1 * total_e / total_n).max(1e-9);
    let e2_expected = (n2 * total_e / total_n).max(1e-9);
    let ne1_expected = (n1 * total_ne / total_n).max(1e-9);
    let ne2_expected = (n2 * total_ne / total_n).max(1e-9);

    ((e1 - e1_expected).abs() - 0.5).powi(2) / e1_expected
        + ((e2 - e2_expected).abs() - 0.5).powi(2) / e2_expected
        + ((n1 - e1 - ne1_expected).abs() - 0.5).powi(2) / ne1_expected
        + ((n2 - e2 - ne2_expected).abs() - 0.5).powi(2) / ne2_expected
}

fn compute_chi_squares_for_bins(bins: &[(f64, f64, f64)]) -> Vec<f64> {
    if bins.len() < 2 {
        return vec![];
    }

    bins.windows(2)
        .map(|pair| calculate_chi2_yates(pair[0].1, pair[0].2, pair[1].1, pair[1].2))
        .collect()
}

fn calculate_single_chi_merge(
    feature: &[f64],
    target: &[i64],
    n_bins: usize,
    threshold: f64,
) -> Vec<f64> {
    if feature.is_empty() {
        return vec![];
    }

    let mut data: Vec<(f64, i64)> = feature
        .iter()
        .zip(target.iter())
        .map(|(&feature_value, &target_value)| (feature_value, target_value))
        .collect();
    data.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));

    let mut bins: Vec<(f64, f64, f64)> = Vec::new();
    if !data.is_empty() {
        let mut curr_val = data[0].0;
        let mut curr_n = 0.0;
        let mut curr_e = 0.0;

        for (feature_value, target_value) in data {
            if feature_value == curr_val {
                curr_n += 1.0;
                if target_value == 1 {
                    curr_e += 1.0;
                }
            } else {
                bins.push((curr_val, curr_n, curr_e));
                curr_val = feature_value;
                curr_n = 1.0;
                curr_e = if target_value == 1 { 1.0 } else { 0.0 };
            }
        }
        bins.push((curr_val, curr_n, curr_e));
    }

    while bins.len() > n_bins {
        let chi_squares = compute_chi_squares_for_bins(&bins);
        if chi_squares.is_empty() {
            break;
        }

        let (min_idx, min_chi2) = chi_squares
            .iter()
            .enumerate()
            .min_by(|left, right| left.1.partial_cmp(right.1).unwrap_or(Ordering::Equal))
            .map(|(idx, value)| (idx, *value))
            .unwrap();

        if min_chi2 >= threshold {
            break;
        }

        let (val1, n1, e1) = bins[min_idx];
        let (_, n2, e2) = bins[min_idx + 1];
        bins[min_idx] = (val1, n1 + n2, e1 + e2);
        bins.remove(min_idx + 1);
    }

    let final_vals: Vec<f64> = bins.iter().map(|bin| bin.0).collect();
    if final_vals.len() < 2 {
        return vec![];
    }

    final_vals
        .windows(2)
        .map(|window| (window[0] + window[1]) / 2.0)
        .collect()
}

#[pyfunction]
fn calculate_chi_merge_numpy(
    py: Python<'_>,
    feature: PyReadonlyArray1<f64>,
    target: PyReadonlyArray1<i64>,
    n_bins: usize,
    threshold: f64,
) -> PyResult<Vec<f64>> {
    let feature_slice = feature.as_slice().map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!("feature not contiguous: {}", err))
    })?;
    let target_slice = target.as_slice().map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!("target not contiguous: {}", err))
    })?;

    if feature_slice.len() != target_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Feature and target must have same length.",
        ));
    }

    py.allow_threads(|| {
        Ok(calculate_single_chi_merge(
            feature_slice,
            target_slice,
            n_bins,
            threshold,
        ))
    })
}

#[pyfunction]
fn calculate_batch_chi_merge_numpy(
    py: Python<'_>,
    features: Vec<PyReadonlyArray1<f64>>,
    target: PyReadonlyArray1<i64>,
    n_bins: usize,
    threshold: f64,
) -> PyResult<Vec<Vec<f64>>> {
    let target_slice = target.as_slice().map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!("target not contiguous: {}", err))
    })?;
    let target_vec = target_slice.to_vec();

    let feature_vecs: Vec<Vec<f64>> = features
        .iter()
        .map(|feature| feature.as_slice().map(|slice| slice.to_vec()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| {
            pyo3::exceptions::PyValueError::new_err(format!("feature not contiguous: {}", err))
        })?;

    py.allow_threads(|| {
        Ok(feature_vecs
            .into_par_iter()
            .map(|feature_vec| {
                calculate_single_chi_merge(&feature_vec, &target_vec, n_bins, threshold)
            })
            .collect())
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(pyo3::wrap_pyfunction!(calculate_chi_merge_numpy, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        calculate_batch_chi_merge_numpy,
        module
    )?)?;
    Ok(())
}
