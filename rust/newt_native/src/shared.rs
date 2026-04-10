use std::cmp::Ordering;

pub(crate) fn build_quantile_edges(values: &[f64], bins: usize) -> Vec<f64> {
    if values.is_empty() {
        return vec![f64::NEG_INFINITY, f64::INFINITY];
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    sorted.dedup();

    if sorted.len() <= 1 {
        return vec![f64::NEG_INFINITY, f64::INFINITY];
    }

    let unique_bins = bins.min(sorted.len());
    let mut edges = Vec::with_capacity(unique_bins + 1);
    let mut all_sorted = values.to_vec();
    all_sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));

    for index in 0..=unique_bins {
        let quantile = index as f64 / unique_bins as f64;
        let position = ((all_sorted.len() - 1) as f64 * quantile).round() as usize;
        edges.push(all_sorted[position]);
    }

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

pub(crate) fn bin_index(value: Option<f64>, edges: &[f64]) -> usize {
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

pub(crate) fn iv_from_counts(good_counts: &[f64], bad_counts: &[f64]) -> f64 {
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

pub(crate) fn count_with_edges(
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

pub(crate) fn count_with_edges_f64(
    values: &[f64],
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

pub(crate) fn psi_from_counts(expected_counts: &[f64], actual_counts: &[f64], epsilon: f64) -> f64 {
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

pub(crate) fn quantile_linear(sorted_values: &[f64], quantile: f64) -> f64 {
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
