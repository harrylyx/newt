use pyo3::prelude::*;
use rayon::prelude::*;

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
                if actual <= *edge {
                    return index;
                }
            }
            edges.len() - 2
        }
    }
}

fn calculate_single_iv(feature: &[Option<f64>], target: &[i64], bins: usize, epsilon: f64) -> f64 {
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

    good_counts
        .iter()
        .zip(bad_counts.iter())
        .map(|(good, bad)| {
            let dist_good = (good / total_good).max(epsilon);
            let dist_bad = (bad / total_bad).max(epsilon);
            (dist_good - dist_bad) * (dist_good / dist_bad).ln()
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

#[pymodule]
fn _newt_iv_rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(calculate_batch_iv, module)?)?;
    Ok(())
}
