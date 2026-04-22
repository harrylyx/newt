use std::collections::HashMap;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::shared::{bin_index, build_quantile_edges, iv_from_counts};

fn calculate_single_iv(feature: &[Option<f64>], target: &[i64], bins: usize, _epsilon: f64) -> f64 {
    let values: Vec<f64> = feature.iter().flatten().copied().collect();
    if values.is_empty() {
        return 0.0;
    }

    let edges = build_quantile_edges(&values, bins);
    let bucket_count = edges.len();
    let mut good_counts = vec![0.0_f64; bucket_count];
    let mut bad_counts = vec![0.0_f64; bucket_count];

    for (value, label) in feature.iter().zip(target.iter()) {
        let index = bin_index(*value, &edges);
        if *label == 1 {
            bad_counts[index] += 1.0;
        } else {
            good_counts[index] += 1.0;
        }
    }

    let total_good: f64 = good_counts.iter().sum();
    let total_bad: f64 = bad_counts.iter().sum();
    if total_good == 0.0 || total_bad == 0.0 {
        return 0.0;
    }

    iv_from_counts(&good_counts, &bad_counts)
}

fn calculate_single_categorical_iv(feature: &[Option<String>], target: &[i64]) -> f64 {
    let mut counts: HashMap<String, (f64, f64)> = HashMap::new();

    for (value, label) in feature.iter().zip(target.iter()) {
        let key = value.clone().unwrap_or_else(|| "__MISSING__".to_string());
        let entry = counts.entry(key).or_insert((0.0_f64, 0.0_f64));
        if *label == 1 {
            entry.1 += 1.0;
        } else {
            entry.0 += 1.0;
        }
    }

    if counts.is_empty() {
        return 0.0;
    }

    let (good_counts, bad_counts): (Vec<f64>, Vec<f64>) =
        counts.values().map(|(good, bad)| (*good, *bad)).unzip();

    iv_from_counts(&good_counts, &bad_counts)
}

fn calculate_single_iv_f64(feature: &[f64], target: &[i64], bins: usize, _epsilon: f64) -> f64 {
    let values: Vec<f64> = feature
        .iter()
        .filter(|value| !value.is_nan())
        .copied()
        .collect();
    if values.is_empty() {
        return 0.0;
    }

    let edges = build_quantile_edges(&values, bins);
    let bucket_count = edges.len();
    let mut good_counts = vec![0.0_f64; bucket_count];
    let mut bad_counts = vec![0.0_f64; bucket_count];

    for (value, label) in feature.iter().zip(target.iter()) {
        let index = if value.is_nan() {
            edges.len() - 1
        } else {
            let mut idx = edges.len() - 2;
            for (offset, edge) in edges[1..edges.len() - 1].iter().enumerate() {
                if *value < *edge {
                    idx = offset;
                    break;
                }
            }
            idx
        };
        if *label == 1 {
            bad_counts[index] += 1.0;
        } else {
            good_counts[index] += 1.0;
        }
    }

    let total_good: f64 = good_counts.iter().sum();
    let total_bad: f64 = bad_counts.iter().sum();
    if total_good == 0.0 || total_bad == 0.0 {
        return 0.0;
    }

    iv_from_counts(&good_counts, &bad_counts)
}

#[pyfunction]
fn calculate_batch_iv(
    features: Vec<Vec<Option<f64>>>,
    target: Vec<i64>,
    bins: usize,
    epsilon: f64,
) -> PyResult<Vec<f64>> {
    for feature in &features {
        if feature.len() != target.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Feature length must match target length.",
            ));
        }
    }

    Ok(features
        .par_iter()
        .map(|feature| calculate_single_iv(feature, &target, bins, epsilon))
        .collect())
}

#[pyfunction]
fn calculate_batch_iv_numpy(
    py: Python<'_>,
    features: Vec<PyReadonlyArray1<f64>>,
    target: PyReadonlyArray1<i64>,
    bins: usize,
    epsilon: f64,
) -> PyResult<Vec<f64>> {
    let target_slice = target.as_slice().map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!("target not contiguous: {}", err))
    })?;

    let feature_vecs: Vec<Vec<f64>> = features
        .iter()
        .map(|feature| {
            let slice = feature.as_slice().map_err(|err| {
                pyo3::exceptions::PyValueError::new_err(format!("feature not contiguous: {}", err))
            })?;
            Ok(slice.to_vec())
        })
        .collect::<PyResult<Vec<_>>>()?;

    for feature_vec in &feature_vecs {
        if feature_vec.len() != target_slice.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Feature length must match target length.",
            ));
        }
    }

    let target_vec = target_slice.to_vec();
    py.allow_threads(|| {
        Ok(feature_vecs
            .par_iter()
            .map(|feature_vec| calculate_single_iv_f64(feature_vec, &target_vec, bins, epsilon))
            .collect())
    })
}

#[pyfunction]
fn calculate_categorical_iv(feature: Vec<Option<String>>, target: Vec<i64>) -> PyResult<f64> {
    if feature.len() != target.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Feature length must match target length.",
        ));
    }

    if target.iter().any(|label| *label != 0 && *label != 1) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Target values must be binary (0/1).",
        ));
    }

    Ok(calculate_single_categorical_iv(&feature, &target))
}

#[pyfunction]
fn calculate_batch_categorical_iv(
    features: Vec<Vec<Option<String>>>,
    target: Vec<i64>,
) -> PyResult<Vec<f64>> {
    if target.iter().any(|label| *label != 0 && *label != 1) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Target values must be binary (0/1).",
        ));
    }

    for feature in &features {
        if feature.len() != target.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Feature length must match target length.",
            ));
        }
    }

    Ok(features
        .par_iter()
        .map(|feature| calculate_single_categorical_iv(feature, &target))
        .collect())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(pyo3::wrap_pyfunction!(calculate_batch_iv, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(calculate_batch_iv_numpy, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(calculate_categorical_iv, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        calculate_batch_categorical_iv,
        module
    )?)?;
    Ok(())
}
