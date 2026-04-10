use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::shared::{count_with_edges, count_with_edges_f64, psi_from_counts, quantile_linear};

fn build_percentile_edges(values: &[f64], buckets: usize) -> Vec<f64> {
    if values.is_empty() {
        return vec![f64::NEG_INFINITY, f64::INFINITY];
    }
    let mut sorted_values = values.to_vec();
    sorted_values
        .sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let bucket_count = buckets.max(1);
    let mut breakpoints: Vec<f64> = (0..=bucket_count)
        .map(|idx| quantile_linear(&sorted_values, idx as f64 / bucket_count as f64))
        .collect();
    breakpoints.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    breakpoints.dedup();
    if breakpoints.len() < 2 {
        return vec![f64::NEG_INFINITY, f64::INFINITY];
    }
    if let Some(first) = breakpoints.first_mut() {
        *first = f64::NEG_INFINITY;
    }
    if let Some(last) = breakpoints.last_mut() {
        *last = f64::INFINITY;
    }
    breakpoints
}

#[pyfunction]
fn calculate_psi_batch_from_edges(
    edges: Vec<f64>,
    expected_counts: Vec<f64>,
    groups: Vec<Vec<Option<f64>>>,
    include_missing_bucket: bool,
    epsilon: f64,
) -> PyResult<Vec<f64>> {
    if edges.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Edges must contain at least 2 values.",
        ));
    }
    let expected_len = if include_missing_bucket {
        edges.len()
    } else {
        edges.len() - 1
    };
    if expected_counts.len() != expected_len {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Expected counts length is incompatible with edges and missing-bucket setting.",
        ));
    }

    Ok(groups
        .par_iter()
        .map(|group| {
            let actual_counts = count_with_edges(group, &edges, include_missing_bucket);
            psi_from_counts(&expected_counts, &actual_counts, epsilon)
        })
        .collect())
}

#[pyfunction]
fn calculate_psi_batch_from_edges_numpy(
    py: Python<'_>,
    edges: PyReadonlyArray1<f64>,
    expected_counts: PyReadonlyArray1<f64>,
    groups: Vec<PyReadonlyArray1<f64>>,
    include_missing_bucket: bool,
    epsilon: f64,
) -> PyResult<Vec<f64>> {
    let edges_slice = edges.as_slice().map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!("edges not contiguous: {}", err))
    })?;
    let expected_slice = expected_counts.as_slice().map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!("expected_counts not contiguous: {}", err))
    })?;
    if edges_slice.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Edges must contain at least 2 values.",
        ));
    }
    let expected_len = if include_missing_bucket {
        edges_slice.len()
    } else {
        edges_slice.len() - 1
    };
    if expected_slice.len() != expected_len {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Expected counts length is incompatible with edges and missing-bucket setting.",
        ));
    }

    let group_vecs: Vec<Vec<f64>> = groups
        .iter()
        .map(|group| group.as_slice().map(|slice| slice.to_vec()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| {
            pyo3::exceptions::PyValueError::new_err(format!("group not contiguous: {}", err))
        })?;

    py.allow_threads(|| {
        Ok(group_vecs
            .par_iter()
            .map(|group_slice| {
                let actual_counts =
                    count_with_edges_f64(group_slice, edges_slice, include_missing_bucket);
                psi_from_counts(expected_slice, &actual_counts, epsilon)
            })
            .collect())
    })
}

#[pyfunction]
fn calculate_feature_psi_pairs_numpy(
    py: Python<'_>,
    expected_groups: Vec<PyReadonlyArray1<f64>>,
    actual_groups: Vec<PyReadonlyArray1<f64>>,
    buckets: usize,
    include_missing_bucket: bool,
    epsilon: f64,
) -> PyResult<Vec<f64>> {
    if expected_groups.len() != actual_groups.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "expected_groups and actual_groups must have the same length.",
        ));
    }

    let expected_vecs: Vec<Vec<f64>> = expected_groups
        .iter()
        .map(|group| group.as_slice().map(|slice| slice.to_vec()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "expected group not contiguous: {}",
                err
            ))
        })?;
    let actual_vecs: Vec<Vec<f64>> = actual_groups
        .iter()
        .map(|group| group.as_slice().map(|slice| slice.to_vec()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| {
            pyo3::exceptions::PyValueError::new_err(format!("actual group not contiguous: {}", err))
        })?;

    py.allow_threads(|| {
        Ok((0..expected_vecs.len())
            .into_par_iter()
            .map(|idx| {
                let expected_values = &expected_vecs[idx];
                let actual_values = &actual_vecs[idx];
                let clean_expected: Vec<f64> = expected_values
                    .iter()
                    .copied()
                    .filter(|value| !value.is_nan())
                    .collect();
                let edges = build_percentile_edges(&clean_expected, buckets.max(1));
                let expected_counts =
                    count_with_edges_f64(expected_values, &edges, include_missing_bucket);
                let actual_counts =
                    count_with_edges_f64(actual_values, &edges, include_missing_bucket);
                psi_from_counts(&expected_counts, &actual_counts, epsilon)
            })
            .collect())
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(pyo3::wrap_pyfunction!(
        calculate_psi_batch_from_edges,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        calculate_psi_batch_from_edges_numpy,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        calculate_feature_psi_pairs_numpy,
        module
    )?)?;
    Ok(())
}
