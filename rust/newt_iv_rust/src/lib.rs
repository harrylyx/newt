use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use faer::linalg::solvers::{Solve, DenseSolveCore};
use statrs::distribution::{ContinuousCDF, Normal};



fn build_edges(values: &[f64], bins: usize) -> Vec<f64> {
    if values.is_empty() {
        return vec![f64::NEG_INFINITY, f64::INFINITY];
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap());
    sorted.dedup();

    if sorted.len() <= 1 {
        return vec![f64::NEG_INFINITY, f64::INFINITY];
    }

    let unique_bins = bins.min(sorted.len());
    let mut edges = Vec::with_capacity(unique_bins + 1);
    let all_values = values.to_vec();
    let mut all_sorted = all_values;
    all_sorted.sort_by(|left, right| left.partial_cmp(right).unwrap());

    for index in 0..=unique_bins {
        let quantile = index as f64 / unique_bins as f64;
        let position = ((all_sorted.len() - 1) as f64 * quantile).round() as usize;
        edges.push(all_sorted[position]);
    }

    edges.sort_by(|left, right| left.partial_cmp(right).unwrap());
    edges.dedup();
    if edges.len() < 2 {
        return vec![f64::NEG_INFINITY, f64::INFINITY];
    }

    if let Some(first) = edges.first_mut() {
        *first = f64::NEG_INFINITY;
    }
    if let Some(last) = edges.last_mut() {
        *last = f64::INFINITY;
    }
    edges
}

fn bin_index(value: Option<f64>, edges: &[f64]) -> usize {
    match value {
        None => edges.len() - 1,
        Some(actual) => {
            for (index, edge) in edges[1..edges.len() - 1].iter().enumerate() {
                if actual < *edge {
                    return index;
                }
            }
            edges.len() - 2
        }
    }
}

fn iv_from_counts(good_counts: &[f64], bad_counts: &[f64]) -> f64 {
    let total_good_raw: f64 = good_counts.iter().sum();
    let total_bad_raw: f64 = bad_counts.iter().sum();
    if total_good_raw == 0.0 || total_bad_raw == 0.0 {
        return 0.0;
    }

    let smoothed_good: Vec<f64> = good_counts.iter().map(|value| value.max(1.0)).collect();
    let smoothed_bad: Vec<f64> = bad_counts.iter().map(|value| value.max(1.0)).collect();

    smoothed_good
        .iter()
        .zip(smoothed_bad.iter())
        .map(|(good, bad)| {
            let dist_good = good / total_good_raw;
            let dist_bad = bad / total_bad_raw;
            (dist_good - dist_bad) * (dist_good / dist_bad).ln()
        })
        .sum()
}

fn calculate_single_iv(feature: &[Option<f64>], target: &[i64], bins: usize, _epsilon: f64) -> f64 {
    let values: Vec<f64> = feature.iter().flatten().copied().collect();
    if values.is_empty() {
        return 0.0;
    }

    let edges = build_edges(&values, bins);
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

fn count_with_edges(
    values: &[Option<f64>],
    edges: &[f64],
    include_missing_bucket: bool,
) -> Vec<f64> {
    let non_missing_bins = edges.len() - 1;
    let missing_index = non_missing_bins;
    let mut counts = if include_missing_bucket {
        vec![0.0_f64; non_missing_bins + 1]
    } else {
        vec![0.0_f64; non_missing_bins]
    };

    for value in values {
        match value {
            None => {
                if include_missing_bucket {
                    counts[missing_index] += 1.0;
                }
            }
            Some(actual) => {
                let mut index = non_missing_bins - 1;
                for (offset, edge) in edges[1..non_missing_bins].iter().enumerate() {
                    if *actual < *edge {
                        index = offset;
                        break;
                    }
                }
                counts[index] += 1.0;
            }
        }
    }

    counts
}

fn psi_from_counts(expected_counts: &[f64], actual_counts: &[f64], epsilon: f64) -> f64 {
    let expected_total: f64 = expected_counts.iter().sum();
    let actual_total: f64 = actual_counts.iter().sum();
    if expected_total == 0.0 || actual_total == 0.0 {
        return f64::NAN;
    }

    expected_counts
        .iter()
        .zip(actual_counts.iter())
        .map(|(expected, actual)| {
            let expected_pct = (expected / expected_total).max(epsilon);
            let actual_pct = (actual / actual_total).max(epsilon);
            (actual_pct - expected_pct) * (actual_pct / expected_pct).ln()
        })
        .sum()
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
fn calculate_categorical_iv(feature: Vec<Option<String>>, target: Vec<i64>) -> PyResult<f64> {
    if feature.len() != target.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Feature length must match target length.",
        ));
    }

    let mut counts: HashMap<String, (f64, f64)> = HashMap::new();

    for (value, label) in feature.iter().zip(target.iter()) {
        if *label != 0 && *label != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Target values must be binary (0/1).",
            ));
        }

        let key = value.clone().unwrap_or_else(|| "__MISSING__".to_string());
        let entry = counts.entry(key).or_insert((0.0_f64, 0.0_f64));
        if *label == 1 {
            entry.1 += 1.0;
        } else {
            entry.0 += 1.0;
        }
    }

    if counts.is_empty() {
        return Ok(0.0);
    }

    let (good_counts, bad_counts): (Vec<f64>, Vec<f64>) =
        counts.values().map(|(good, bad)| (*good, *bad)).unzip();

    Ok(iv_from_counts(&good_counts, &bad_counts))
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
    let edges_slice = edges.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("edges not contiguous: {}", e))
    })?;
    let expected_slice = expected_counts.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("expected_counts not contiguous: {}", e))
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

    // Collect slices outside par_iter (PyReadonlyArray borrows Python GIL)
    let group_vecs: Vec<Vec<f64>> = groups
        .iter()
        .map(|group| group.as_slice().map(|slice| slice.to_vec()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("group not contiguous: {}", e))
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

/// Count values into bins from a raw f64 slice where NaN represents missing.
fn count_with_edges_f64(values: &[f64], edges: &[f64], include_missing_bucket: bool) -> Vec<f64> {
    let non_missing_bins = edges.len() - 1;
    let missing_index = non_missing_bins;
    let mut counts = if include_missing_bucket {
        vec![0.0_f64; non_missing_bins + 1]
    } else {
        vec![0.0_f64; non_missing_bins]
    };

    for value in values {
        if value.is_nan() {
            if include_missing_bucket {
                counts[missing_index] += 1.0;
            }
        } else {
            let mut index = non_missing_bins - 1;
            for (offset, edge) in edges[1..non_missing_bins].iter().enumerate() {
                if *value < *edge {
                    index = offset;
                    break;
                }
            }
            counts[index] += 1.0;
        }
    }

    counts
}

#[pyfunction]
fn calculate_batch_iv_numpy(
    py: Python<'_>,
    features: Vec<PyReadonlyArray1<f64>>,
    target: PyReadonlyArray1<i64>,
    bins: usize,
    epsilon: f64,
) -> PyResult<Vec<f64>> {
    let target_slice = target.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("target not contiguous: {}", e))
    })?;

    // Copy feature data out of GIL-bound numpy arrays
    let feature_vecs: Vec<Vec<f64>> = features
        .iter()
        .map(|feature| {
            let slice = feature.as_slice().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("feature not contiguous: {}", e))
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

    let target_vec: Vec<i64> = target_slice.to_vec();
    py.allow_threads(|| {
        Ok(feature_vecs
            .par_iter()
            .map(|feature_vec| calculate_single_iv_f64(feature_vec, &target_vec, bins, epsilon))
            .collect())
    })
}

/// Calculate IV from a raw f64 slice where NaN represents missing.
fn calculate_single_iv_f64(feature: &[f64], target: &[i64], bins: usize, _epsilon: f64) -> f64 {
    let values: Vec<f64> = feature.iter().filter(|v| !v.is_nan()).copied().collect();
    if values.is_empty() {
        return 0.0;
    }

    let edges = build_edges(&values, bins);
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

fn _binary_metrics_exact(labels: &[u8], scores: &[f64]) -> (f64, f64) {
    let total_bad = labels.iter().filter(|&&label| label == 1).count() as f64;
    let total_good = labels.len() as f64 - total_bad;
    if total_bad <= 0.0 || total_good <= 0.0 {
        return (f64::NAN, f64::NAN);
    }

    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_by(|left, right| {
        scores[*right]
            .partial_cmp(&scores[*left])
            .unwrap_or(Ordering::Equal)
    });

    let mut cum_bad = 0.0_f64;
    let mut cum_good = 0.0_f64;
    let mut prev_tpr = 0.0_f64;
    let mut prev_fpr = 0.0_f64;
    let mut auc = 0.0_f64;
    let mut ks = 0.0_f64;

    let mut cursor = 0usize;
    while cursor < indices.len() {
        let score_value = scores[indices[cursor]];
        let mut bucket_bad = 0.0_f64;
        let mut bucket_good = 0.0_f64;
        while cursor < indices.len() && scores[indices[cursor]] == score_value {
            if labels[indices[cursor]] == 1 {
                bucket_bad += 1.0;
            } else {
                bucket_good += 1.0;
            }
            cursor += 1;
        }

        cum_bad += bucket_bad;
        cum_good += bucket_good;
        let tpr = cum_bad / total_bad;
        let fpr = cum_good / total_good;
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5;
        ks = ks.max((tpr - fpr).abs());
        prev_tpr = tpr;
        prev_fpr = fpr;
    }

    (auc, ks)
}

fn _binary_metrics_lifts(
    labels: &[u8],
    scores: &[f64],
    levels: &[f64],
    descending: bool,
) -> Vec<f64> {
    let n = labels.len();
    if n == 0 {
        return levels.iter().map(|_| f64::NAN).collect();
    }
    let global_bad_rate = labels.iter().map(|value| *value as f64).sum::<f64>() / n as f64;
    if global_bad_rate == 0.0 {
        return levels.iter().map(|_| 0.0).collect();
    }

    let mut indices: Vec<usize> = (0..scores.len()).collect();
    if descending {
        indices.sort_by(|left, right| {
            scores[*right]
                .partial_cmp(&scores[*left])
                .unwrap_or(Ordering::Equal)
        });
    } else {
        indices.sort_by(|left, right| {
            scores[*left]
                .partial_cmp(&scores[*right])
                .unwrap_or(Ordering::Equal)
        });
    }

    let mut cumulative_bad = vec![0.0_f64; n];
    let mut running_bad = 0.0_f64;
    for (position, index) in indices.iter().enumerate() {
        running_bad += labels[*index] as f64;
        cumulative_bad[position] = running_bad;
    }

    levels
        .iter()
        .map(|level| {
            let n_top = ((n as f64 * *level).ceil() as usize).max(1).min(n);
            let top_bad_rate = cumulative_bad[n_top - 1] / n_top as f64;
            top_bad_rate / global_bad_rate
        })
        .collect()
}

fn _quantile_linear(sorted_values: &[f64], quantile: f64) -> f64 {
    if sorted_values.len() == 1 {
        return sorted_values[0];
    }
    let position = (sorted_values.len() - 1) as f64 * quantile;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        return sorted_values[lower];
    }
    let fraction = position - lower as f64;
    sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
}

fn _build_score_edges(scores: &[f64], bins: usize) -> Vec<f64> {
    let clean: Vec<f64> = scores
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect();
    if clean.is_empty() {
        return vec![f64::NEG_INFINITY, f64::INFINITY];
    }
    let mut unique = clean.clone();
    unique.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    unique.dedup();
    if unique.len() <= 1 {
        return vec![f64::NEG_INFINITY, f64::INFINITY];
    }

    let mut sorted_clean = clean.clone();
    sorted_clean.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let n_bins = bins.max(1).min(unique.len());
    let mut edges: Vec<f64> = (0..=n_bins)
        .map(|idx| _quantile_linear(&sorted_clean, idx as f64 / n_bins as f64))
        .collect();
    edges.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    edges.dedup();
    if edges.len() < 2 {
        return vec![f64::NEG_INFINITY, f64::INFINITY];
    }
    if let Some(first) = edges.first_mut() {
        *first = f64::NEG_INFINITY;
    }
    if let Some(last) = edges.last_mut() {
        *last = f64::INFINITY;
    }
    edges
}

fn _build_percentile_edges(values: &[f64], buckets: usize) -> Vec<f64> {
    if values.is_empty() {
        return vec![f64::NEG_INFINITY, f64::INFINITY];
    }
    let mut sorted_values = values.to_vec();
    sorted_values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let mut breakpoints: Vec<f64> = (0..=buckets.max(1))
        .map(|idx| _quantile_linear(&sorted_values, idx as f64 / buckets.max(1) as f64))
        .collect();
    breakpoints.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
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

fn _assign_bin_index(score: f64, edges: &[f64]) -> usize {
    let non_missing_bins = edges.len() - 1;
    let mut index = non_missing_bins - 1;
    for (offset, edge) in edges[1..non_missing_bins].iter().enumerate() {
        if score < *edge {
            index = offset;
            break;
        }
    }
    index
}

fn _binned_auc(good_counts: &[f64], bad_counts: &[f64]) -> f64 {
    let total_good: f64 = good_counts.iter().sum();
    let total_bad: f64 = bad_counts.iter().sum();
    if total_good == 0.0 || total_bad == 0.0 {
        return f64::NAN;
    }

    let mut prev_fpr = 0.0_f64;
    let mut prev_tpr = 0.0_f64;
    let mut cumulative_good = 0.0_f64;
    let mut cumulative_bad = 0.0_f64;
    let mut auc = 0.0_f64;
    for idx in 0..good_counts.len() {
        cumulative_good += good_counts[idx];
        cumulative_bad += bad_counts[idx];
        let fpr = cumulative_good / total_good;
        let tpr = cumulative_bad / total_bad;
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5;
        prev_fpr = fpr;
        prev_tpr = tpr;
    }
    auc
}

fn _binned_ks(good_counts: &[f64], bad_counts: &[f64]) -> f64 {
    let total_good: f64 = good_counts.iter().sum();
    let total_bad: f64 = bad_counts.iter().sum();
    if total_good == 0.0 || total_bad == 0.0 {
        return f64::NAN;
    }

    let mut cumulative_good = 0.0_f64;
    let mut cumulative_bad = 0.0_f64;
    let mut ks = 0.0_f64;
    for idx in 0..good_counts.len() {
        cumulative_good += good_counts[idx];
        cumulative_bad += bad_counts[idx];
        let diff = (cumulative_bad / total_bad - cumulative_good / total_good).abs();
        ks = ks.max(diff);
    }
    ks
}

fn _binned_lifts(
    good_counts: &[f64],
    bad_counts: &[f64],
    total_n: usize,
    levels: &[f64],
    descending: bool,
) -> Vec<f64> {
    let mut total_counts: Vec<f64> = good_counts
        .iter()
        .zip(bad_counts.iter())
        .map(|(good, bad)| good + bad)
        .collect();
    let mut ordered_bad: Vec<f64> = bad_counts.to_vec();
    if descending {
        total_counts.reverse();
        ordered_bad.reverse();
    }

    let global_bad_rate = bad_counts.iter().sum::<f64>() / total_n.max(1) as f64;
    if global_bad_rate == 0.0 {
        return levels.iter().map(|_| 0.0).collect();
    }

    let mut cum_total = vec![0.0_f64; total_counts.len()];
    let mut cum_bad = vec![0.0_f64; total_counts.len()];
    let mut running_total = 0.0_f64;
    let mut running_bad = 0.0_f64;
    for idx in 0..total_counts.len() {
        running_total += total_counts[idx];
        running_bad += ordered_bad[idx];
        cum_total[idx] = running_total;
        cum_bad[idx] = running_bad;
    }

    levels
        .iter()
        .map(|level| {
            let n_top = ((total_n as f64 * *level).ceil() as usize).max(1);
            let n_top_f = n_top as f64;
            let mut idx = 0usize;
            while idx < cum_total.len() && cum_total[idx] < n_top_f {
                idx += 1;
            }
            if idx >= cum_total.len() {
                idx = cum_total.len() - 1;
            }

            let top_bad_rate = if idx == 0 {
                if cum_total[0] > 0.0 {
                    cum_bad[0] / cum_total[0]
                } else {
                    0.0
                }
            } else {
                let prev_total = cum_total[idx - 1];
                let prev_bad = cum_bad[idx - 1];
                let needed = (n_top_f - prev_total).max(0.0);
                if total_counts[idx] > 0.0 {
                    let bin_bad_rate = ordered_bad[idx] / total_counts[idx];
                    (prev_bad + needed * bin_bad_rate) / n_top_f
                } else {
                    cum_bad[idx] / cum_total[idx].max(1.0)
                }
            };
            top_bad_rate / global_bad_rate
        })
        .collect()
}

fn _binary_metrics_binned(
    labels: &[u8],
    scores: &[f64],
    bins: usize,
    levels: &[f64],
    descending: bool,
) -> (f64, f64, Vec<f64>) {
    let edges = _build_score_edges(scores, bins);
    let n_bins = edges.len() - 1;
    let mut good_counts = vec![0.0_f64; n_bins];
    let mut bad_counts = vec![0.0_f64; n_bins];
    for (label, score) in labels.iter().zip(scores.iter()) {
        let idx = _assign_bin_index(*score, &edges);
        if *label == 1 {
            bad_counts[idx] += 1.0;
        } else {
            good_counts[idx] += 1.0;
        }
    }
    let mut good_desc = good_counts.clone();
    let mut bad_desc = bad_counts.clone();
    good_desc.reverse();
    bad_desc.reverse();
    let auc = _binned_auc(&good_desc, &bad_desc);
    let ks = _binned_ks(&good_desc, &bad_desc);
    let lifts = _binned_lifts(&good_counts, &bad_counts, labels.len(), levels, descending);
    (auc, ks, lifts)
}

fn _binary_metrics_for_group(
    labels: &[f64],
    scores: &[f64],
    levels: &[f64],
    lift_use_descending_score: bool,
    reverse_auc_label: bool,
    metrics_mode: &str,
    bins: usize,
) -> HashMap<String, f64> {
    let mut metrics: HashMap<String, f64> = HashMap::new();
    let total = labels.len() as f64;
    let good = labels.iter().filter(|value| **value == 0.0).count() as f64;
    let bad = labels.iter().filter(|value| **value == 1.0).count() as f64;
    let binary_total = good + bad;
    let bad_rate = if binary_total > 0.0 {
        bad / binary_total
    } else {
        f64::NAN
    };
    metrics.insert("总".to_string(), total);
    metrics.insert("好".to_string(), good);
    metrics.insert("坏".to_string(), bad);
    metrics.insert("坏占比".to_string(), bad_rate);

    let mut y_clean: Vec<u8> = Vec::new();
    let mut score_clean: Vec<f64> = Vec::new();
    for (label, score) in labels.iter().zip(scores.iter()) {
        if score.is_nan() {
            continue;
        }
        if *label == 0.0 {
            y_clean.push(0u8);
            score_clean.push(*score);
        } else if *label == 1.0 {
            y_clean.push(1u8);
            score_clean.push(*score);
        }
    }

    let class_count = y_clean
        .iter()
        .copied()
        .collect::<std::collections::HashSet<u8>>()
        .len();
    if y_clean.is_empty() || class_count < 2 {
        metrics.insert("KS".to_string(), f64::NAN);
        metrics.insert("AUC".to_string(), f64::NAN);
        for level in levels {
            let key = format!("{}%lift", (*level * 100.0).round() as i32);
            metrics.insert(key, f64::NAN);
        }
        return metrics;
    }

    let (mut auc, ks, lifts) = if metrics_mode == "binned" {
        _binary_metrics_binned(
            &y_clean,
            &score_clean,
            bins,
            levels,
            lift_use_descending_score,
        )
    } else {
        let (auc_exact, ks_exact) = _binary_metrics_exact(&y_clean, &score_clean);
        let lifts_exact =
            _binary_metrics_lifts(&y_clean, &score_clean, levels, lift_use_descending_score);
        (auc_exact, ks_exact, lifts_exact)
    };
    if reverse_auc_label {
        auc = 1.0 - auc;
    }
    metrics.insert("KS".to_string(), ks);
    metrics.insert("AUC".to_string(), auc);
    for (level, lift) in levels.iter().zip(lifts.iter()) {
        let key = format!("{}%lift", (*level * 100.0).round() as i32);
        metrics.insert(key, *lift);
    }
    metrics
}

#[pyfunction]
fn calculate_binary_metrics_batch_numpy(
    py: Python<'_>,
    label_groups: Vec<PyReadonlyArray1<f64>>,
    score_groups: Vec<PyReadonlyArray1<f64>>,
    lift_levels: Vec<f64>,
    lift_use_descending_score: bool,
    reverse_auc_label: bool,
    metrics_mode: String,
    bins: usize,
) -> PyResult<Vec<HashMap<String, f64>>> {
    if label_groups.len() != score_groups.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "label_groups and score_groups must have the same length.",
        ));
    }
    if metrics_mode != "exact" && metrics_mode != "binned" {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "metrics_mode must be 'exact' or 'binned'.",
        ));
    }

    let label_vecs: Vec<Vec<f64>> = label_groups
        .iter()
        .map(|group| group.as_slice().map(|slice| slice.to_vec()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| {
            pyo3::exceptions::PyValueError::new_err(format!("label group not contiguous: {}", err))
        })?;
    let score_vecs: Vec<Vec<f64>> = score_groups
        .iter()
        .map(|group| group.as_slice().map(|slice| slice.to_vec()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| {
            pyo3::exceptions::PyValueError::new_err(format!("score group not contiguous: {}", err))
        })?;

    for (labels, scores) in label_vecs.iter().zip(score_vecs.iter()) {
        if labels.len() != scores.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Each (labels, scores) group must have the same length.",
            ));
        }
    }

    py.allow_threads(|| {
        Ok((0..label_vecs.len())
            .into_par_iter()
            .map(|idx| {
                _binary_metrics_for_group(
                    &label_vecs[idx],
                    &score_vecs[idx],
                    &lift_levels,
                    lift_use_descending_score,
                    reverse_auc_label,
                    &metrics_mode,
                    bins,
                )
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
                let edges = _build_percentile_edges(&clean_expected, buckets.max(1));
                let expected_counts =
                    count_with_edges_f64(expected_values, &edges, include_missing_bucket);
                let actual_counts =
                    count_with_edges_f64(actual_values, &edges, include_missing_bucket);
                psi_from_counts(&expected_counts, &actual_counts, epsilon)
            })
            .collect())
    })
}

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

#[derive(Debug, Clone)]
struct BinNode {
    val: f64, // max value of the bin
    n: f64,   // count
    e: f64,   // event count
    prev: Option<usize>,
    next: Option<usize>,
    active: bool,
}

#[derive(PartialEq)]
struct ChiSquareEntry {
    chi2: f64,
    index: usize, // id of the left bin
}

impl Eq for ChiSquareEntry {}

impl PartialOrd for ChiSquareEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ChiSquareEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap
        other.chi2.partial_cmp(&self.chi2).unwrap_or(Ordering::Equal)
    }
}

fn _calculate_single_chi_merge(
    feature: &[f64],
    target: &[i64],
    n_bins: usize,
    threshold: f64,
) -> Vec<f64> {
    if feature.is_empty() {
        return vec![];
    }

    // 1. Initial binning using unique values
    let mut data: Vec<(f64, i64)> = feature
        .iter()
        .zip(target.iter())
        .map(|(&f, &t)| (f, t))
        .collect();
    data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let mut nodes: Vec<BinNode> = Vec::new();
    if !data.is_empty() {
        let mut curr_val = data[0].0;
        let mut curr_n = 0.0;
        let mut curr_e = 0.0;

        for (f, t) in data {
            if f == curr_val {
                curr_n += 1.0;
                if t == 1 {
                    curr_e += 1.0;
                }
            } else {
                let idx = nodes.len();
                nodes.push(BinNode {
                    val: curr_val,
                    n: curr_n,
                    e: curr_e,
                    prev: if idx > 0 { Some(idx - 1) } else { None },
                    next: None,
                    active: true,
                });
                if idx > 0 {
                    nodes[idx - 1].next = Some(idx);
                }
                curr_val = f;
                curr_n = 1.0;
                curr_e = if t == 1 { 1.0 } else { 0.0 };
            }
        }
        let idx = nodes.len();
        nodes.push(BinNode {
            val: curr_val,
            n: curr_n,
            e: curr_e,
            prev: if idx > 0 { Some(idx - 1) } else { None },
            next: None,
            active: true,
        });
        if idx > 0 {
            nodes[idx - 1].next = Some(idx);
        }
    }

    if nodes.len() <= n_bins {
        let splits: Vec<f64> = nodes.iter().map(|b| b.val).collect();
        return if splits.is_empty() { vec![] } else { splits[0..splits.len()-1].to_vec() };
    }

    // 2. Initial chi-squares in a min-heap
    let mut heap = BinaryHeap::new();
    for i in 0..nodes.len() - 1 {
        let chi2 = calculate_chi2_yates(nodes[i].n, nodes[i].e, nodes[i + 1].n, nodes[i + 1].e);
        heap.push(ChiSquareEntry { chi2, index: i });
    }

    let mut active_count = nodes.len();

    // 3. Merge Loop
    while active_count > n_bins {
        let min_entry = match heap.pop() {
            Some(e) => e,
            None => break,
        };

        if !nodes[min_entry.index].active {
            continue;
        }
        let next_idx = match nodes[min_entry.index].next {
            Some(idx) if nodes[idx].active => idx,
            _ => continue,
        };

        if min_entry.chi2 >= threshold {
            break;
        }

        // Merge next_idx into min_entry.index
        nodes[min_entry.index].n += nodes[next_idx].n;
        nodes[min_entry.index].e += nodes[next_idx].e;
        nodes[min_entry.index].val = nodes[next_idx].val; // update max val
        nodes[next_idx].active = false;
        active_count -= 1;

        // Update links
        let after_next = nodes[next_idx].next;
        nodes[min_entry.index].next = after_next;
        if let Some(an) = after_next {
            nodes[an].prev = Some(min_entry.index);
            // Re-calculate chi2 for the new adjacency
            let chi2 = calculate_chi2_yates(
                nodes[min_entry.index].n,
                nodes[min_entry.index].e,
                nodes[an].n,
                nodes[an].e,
            );
            heap.push(ChiSquareEntry {
                chi2,
                index: min_entry.index,
            });
        }

        if let Some(prev_idx) = nodes[min_entry.index].prev {
            let chi2 = calculate_chi2_yates(
                nodes[prev_idx].n,
                nodes[prev_idx].e,
                nodes[min_entry.index].n,
                nodes[min_entry.index].e,
            );
            heap.push(ChiSquareEntry {
                chi2,
                index: prev_idx,
            });
        }
    }

    // 4. Extract splits
    let mut final_vals = Vec::new();
    let mut curr = Some(0);
    while let Some(idx) = curr {
        if nodes[idx].active {
            final_vals.push(nodes[idx].val);
        }
        curr = nodes[idx].next;
    }

    if final_vals.len() < 2 {
        return vec![];
    }

    // Splits are points between bins. For ChiMerge, splits are max_val of bins except the last one.
    final_vals[0..final_vals.len() - 1].to_vec()
}

#[pyfunction]
fn calculate_chi_merge_numpy(
    py: Python<'_>,
    feature: PyReadonlyArray1<f64>,
    target: PyReadonlyArray1<i64>,
    n_bins: usize,
    threshold: f64,
) -> PyResult<Vec<f64>> {
    let f_slice = feature.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("feature not contiguous: {}", e))
    })?;
    let t_slice = target.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("target not contiguous: {}", e))
    })?;

    if f_slice.len() != t_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Feature and target must have same length.",
        ));
    }

    py.allow_threads(|| Ok(_calculate_single_chi_merge(f_slice, t_slice, n_bins, threshold)))
}

#[pyfunction]
fn calculate_batch_chi_merge_numpy(
    py: Python<'_>,
    features: Vec<PyReadonlyArray1<f64>>,
    target: PyReadonlyArray1<i64>,
    n_bins: usize,
    threshold: f64,
) -> PyResult<Vec<Vec<f64>>> {
    let t_slice = target.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("target not contiguous: {}", e))
    })?;
    let t_vec: Vec<i64> = t_slice.to_vec();

    let feature_vecs: Vec<Vec<f64>> = features
        .iter()
        .map(|f| f.as_slice().map(|s| s.to_vec()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("feature not contiguous: {}", e))
        })?;

    py.allow_threads(|| {
        Ok(feature_vecs
            .into_par_iter()
            .map(|f| _calculate_single_chi_merge(&f, &t_vec, n_bins, threshold))
            .collect())
    })
}

/// Internal structure to hold logistic regression results
struct LogitResult {
    coefficients: faer::Mat<f64>,
    p_values: Vec<f64>,
    aic: f64,
    bic: f64,
    log_likelihood: f64,
    converged: bool,
}

/// Internal IRLS solver for logistic regression
fn _fit_logit_internal(
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

    // Buffers to avoid allocations in loop
    for _ in 0..max_iter {
        let x_beta = x_faer * &beta;
        let p = faer::Mat::<f64>::from_fn(n, 1, |r, _| {
            let v = x_beta[(r, 0)];
            1.0_f64 / (1.0_f64 + (-v).exp())
        });

        let mut current_ll = 0.0_f64;
        for i in 0..n {
            let pi = p[(i, 0)].clamp(1e-15, 1.0 - 1e-15);
            current_ll += y_faer[(i, 0)] * pi.ln() + (1.0_f64 - y_faer[(i, 0)]) * (1.0_f64 - pi).ln();
        }

        let mut err = faer::Mat::<f64>::zeros(n, 1);
        for i in 0..n {
            err[(i, 0)] = y_faer[(i, 0)] - p[(i, 0)];
        }
        let gradient = x_faer.transpose() * &err;

        // Optimized Hessian calculation: X^T * W * X
        // Scaling rows of X by sqrt(W) and then X_weighted^T * X_weighted
        let mut x_weighted = faer::Mat::<f64>::zeros(n, k);
        for i in 0..n {
            let pi = p[(i, 0)];
            let w = pi * (1.0 - pi);
            for j in 0..k {
                x_weighted[(i, j)] = x_faer[(i, j)] * w;
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

    // Diagnostics
    let x_beta = x_faer * &beta;
    let p = faer::Mat::<f64>::from_fn(n, 1, |r, _| {
        let v = x_beta[(r, 0)];
        1.0_f64 / (1.0_f64 + (-v).exp())
    });

    let mut x_weighted = faer::Mat::<f64>::zeros(n, k);
    for i in 0..n {
        let pi = p[(i, 0)];
        let w = pi * (1.0 - pi);
        for j in 0..k {
            x_weighted[(i, j)] = x_faer[(i, j)] * w;
        }
    }
    let hessian = x_faer.transpose() * &x_weighted;
    let inv_hessian = hessian.full_piv_lu().inverse();

    let mut p_values = Vec::with_capacity(k);
    let normal = Normal::new(0.0, 1.0).map_err(|e| e.to_string())?;

    for i in 0..k {
        let var = inv_hessian[(i, i)];
        let p_val = if var > 0.0 {
            let se = var.sqrt();
            let b = beta[(i, 0)];
            let z = b / se;
            2.0 * (1.0 - normal.cdf(z.abs()))
        } else {
            f64::NAN
        };
        p_values.push(p_val);
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

    let x_faer = faer::Mat::<f64>::from_fn(n, k, |r, c| x_view[(r, c)]);
    let y_faer = faer::Mat::<f64>::from_fn(n, 1, |r, _| y_view[r]);

    match _fit_logit_internal(x_faer.as_ref(), y_faer.as_ref(), max_iter, tol) {
        Ok(res) => {
            let dict = PyDict::new(py);
            let mut beta_flat = Vec::with_capacity(k);
            for i in 0..k {
                beta_flat.push(res.coefficients[(i, 0)]);
            }

            dict.set_item("coefficients", beta_flat.into_pyobject(py)?)?;
            dict.set_item("p_values", res.p_values)?;
            dict.set_item("aic", res.aic)?;
            dict.set_item("bic", res.bic)?;
            dict.set_item("log_likelihood", res.log_likelihood)?;
            dict.set_item("converged", res.converged)?;

            Ok(dict.into_any().unbind())
        }
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
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

    // 1. Prepare shared data
    let y_faer = faer::Mat::<f64>::from_fn(n, 1, |r, _| y_view[r]);

    // Copy candidate data to Vec<Vec<f64>> to move across threads easily
    let candidate_vecs: Vec<Vec<f64>> = candidate_features
        .iter()
        .map(|f| {
            f.as_slice()
                .map(|s| s.to_vec())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Candidate not contiguous: {}", e)))
        })
        .collect::<PyResult<Vec<_>>>()?;

    // 2. Parallel processing
    let results: Vec<LogitResult> = py.allow_threads(|| {
        candidate_vecs.par_iter().map(|cand_vec| {
            // Build temporary X: [fixed_x | cand]
            let x_combined = faer::Mat::<f64>::from_fn(n, k_fixed + 1, |r, c| {
                if c < k_fixed {
                    fixed_x_view[(r, c)]
                } else {
                    cand_vec[r]
                }
            });

            // Fit model
            // We ignore errors here and return a "failed" result if needed
            match _fit_logit_internal(x_combined.as_ref(), y_faer.as_ref(), max_iter, tol) {
                Ok(res) => res,
                Err(_) => LogitResult {
                    coefficients: faer::Mat::zeros(k_fixed + 1, 1),
                    p_values: vec![1.0; k_fixed + 1],
                    aic: f64::INFINITY,
                    bic: f64::INFINITY,
                    log_likelihood: f64::NEG_INFINITY,
                    converged: false,
                }
            }
        }).collect()
    });

    // 3. Convert back to Python objects
    let mut py_results = Vec::with_capacity(results.len());
    for res in results {
        let dict = PyDict::new(py);
        dict.set_item("p_value", res.p_values.last().copied().unwrap_or(1.0))?;
        dict.set_item("aic", res.aic)?;
        dict.set_item("bic", res.bic)?;
        dict.set_item("converged", res.converged)?;
        py_results.push(dict.into_any().unbind());
    }

    Ok(py_results)
}

#[pymodule]
fn _newt_iv_rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(calculate_batch_iv, module)?)?;
    module.add_function(wrap_pyfunction!(calculate_batch_iv_numpy, module)?)?;
    module.add_function(wrap_pyfunction!(calculate_categorical_iv, module)?)?;
    module.add_function(wrap_pyfunction!(calculate_psi_batch_from_edges, module)?)?;
    module.add_function(wrap_pyfunction!(
        calculate_psi_batch_from_edges_numpy,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        calculate_binary_metrics_batch_numpy,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(calculate_feature_psi_pairs_numpy, module)?)?;
    module.add_function(wrap_pyfunction!(calculate_chi_merge_numpy, module)?)?;
    module.add_function(wrap_pyfunction!(
        calculate_batch_chi_merge_numpy,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(fit_logistic_regression_numpy, module)?)?;
    module.add_function(wrap_pyfunction!(batch_fit_logistic_regression_numpy, module)?)?;
    Ok(())
}
