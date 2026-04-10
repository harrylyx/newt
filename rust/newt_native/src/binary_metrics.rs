use std::cmp::Ordering;
use std::collections::HashMap;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::shared::quantile_linear;

fn binary_metrics_exact(labels: &[u8], scores: &[f64]) -> (f64, f64) {
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

fn binary_metrics_lifts(
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

fn build_score_edges(scores: &[f64], bins: usize) -> Vec<f64> {
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
        .map(|idx| quantile_linear(&sorted_clean, idx as f64 / n_bins as f64))
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

fn assign_bin_index(score: f64, edges: &[f64]) -> usize {
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

fn binned_auc(good_counts: &[f64], bad_counts: &[f64]) -> f64 {
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

fn binned_ks(good_counts: &[f64], bad_counts: &[f64]) -> f64 {
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

fn binned_lifts(
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
    let mut ordered_bad = bad_counts.to_vec();
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

fn binary_metrics_binned(
    labels: &[u8],
    scores: &[f64],
    bins: usize,
    levels: &[f64],
    descending: bool,
) -> (f64, f64, Vec<f64>) {
    let edges = build_score_edges(scores, bins);
    let n_bins = edges.len() - 1;
    let mut good_counts = vec![0.0_f64; n_bins];
    let mut bad_counts = vec![0.0_f64; n_bins];
    for (label, score) in labels.iter().zip(scores.iter()) {
        let idx = assign_bin_index(*score, &edges);
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
    let auc = binned_auc(&good_desc, &bad_desc);
    let ks = binned_ks(&good_desc, &bad_desc);
    let lifts = binned_lifts(&good_counts, &bad_counts, labels.len(), levels, descending);
    (auc, ks, lifts)
}

fn binary_metrics_for_group(
    labels: &[f64],
    scores: &[f64],
    levels: &[f64],
    lift_use_descending_score: bool,
    reverse_auc_label: bool,
    metrics_mode: &str,
    bins: usize,
) -> HashMap<String, f64> {
    let mut metrics = HashMap::new();
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

    let mut y_clean = Vec::new();
    let mut score_clean = Vec::new();
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
        binary_metrics_binned(
            &y_clean,
            &score_clean,
            bins,
            levels,
            lift_use_descending_score,
        )
    } else {
        let (auc_exact, ks_exact) = binary_metrics_exact(&y_clean, &score_clean);
        let lifts_exact =
            binary_metrics_lifts(&y_clean, &score_clean, levels, lift_use_descending_score);
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
                binary_metrics_for_group(
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

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(pyo3::wrap_pyfunction!(
        calculate_binary_metrics_batch_numpy,
        module
    )?)?;
    Ok(())
}
