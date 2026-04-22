use std::cmp::Ordering;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;

fn pearson_pairwise(x: &[f64], y: &[f64]) -> f64 {
    let mut n = 0.0_f64;
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_x2 = 0.0_f64;
    let mut sum_y2 = 0.0_f64;
    let mut sum_xy = 0.0_f64;

    for idx in 0..x.len() {
        let x_val = x[idx];
        let y_val = y[idx];
        if x_val.is_nan() || y_val.is_nan() {
            continue;
        }
        n += 1.0;
        sum_x += x_val;
        sum_y += y_val;
        sum_x2 += x_val * x_val;
        sum_y2 += y_val * y_val;
        sum_xy += x_val * y_val;
    }

    if n < 2.0 {
        return f64::NAN;
    }

    let cov = sum_xy - (sum_x * sum_y / n);
    let var_x = sum_x2 - (sum_x * sum_x / n);
    let var_y = sum_y2 - (sum_y * sum_y / n);
    if var_x <= 0.0 || var_y <= 0.0 {
        return f64::NAN;
    }
    cov / (var_x.sqrt() * var_y.sqrt())
}

fn rank_average(values: &[f64]) -> Vec<f64> {
    let mut ranked = vec![f64::NAN; values.len()];
    let mut non_missing: Vec<(f64, usize)> = values
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(idx, value)| {
            if value.is_nan() {
                None
            } else {
                Some((value, idx))
            }
        })
        .collect();

    non_missing.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));

    let mut start = 0usize;
    while start < non_missing.len() {
        let mut end = start + 1;
        while end < non_missing.len() && non_missing[end].0 == non_missing[start].0 {
            end += 1;
        }

        let avg_rank = ((start + 1) as f64 + end as f64) * 0.5;
        for (_, idx) in non_missing[start..end].iter() {
            ranked[*idx] = avg_rank;
        }
        start = end;
    }

    ranked
}

fn kendall_tau_b_pairwise(x: &[f64], y: &[f64]) -> f64 {
    let mut x_valid = Vec::new();
    let mut y_valid = Vec::new();
    for idx in 0..x.len() {
        let x_val = x[idx];
        let y_val = y[idx];
        if x_val.is_nan() || y_val.is_nan() {
            continue;
        }
        x_valid.push(x_val);
        y_valid.push(y_val);
    }

    let n = x_valid.len();
    if n < 2 {
        return f64::NAN;
    }

    let mut concordant = 0.0_f64;
    let mut discordant = 0.0_f64;
    let mut ties_x = 0.0_f64;
    let mut ties_y = 0.0_f64;

    for i in 0..(n - 1) {
        for j in (i + 1)..n {
            let x_cmp = x_valid[i]
                .partial_cmp(&x_valid[j])
                .unwrap_or(Ordering::Equal);
            let y_cmp = y_valid[i]
                .partial_cmp(&y_valid[j])
                .unwrap_or(Ordering::Equal);

            match (x_cmp, y_cmp) {
                (Ordering::Equal, Ordering::Equal) => {}
                (Ordering::Equal, _) => {
                    ties_x += 1.0;
                }
                (_, Ordering::Equal) => {
                    ties_y += 1.0;
                }
                (Ordering::Greater, Ordering::Greater) | (Ordering::Less, Ordering::Less) => {
                    concordant += 1.0;
                }
                _ => {
                    discordant += 1.0;
                }
            }
        }
    }

    let denom = ((concordant + discordant + ties_x) * (concordant + discordant + ties_y)).sqrt();
    if denom == 0.0 {
        return f64::NAN;
    }
    (concordant - discordant) / denom
}

fn pairwise_corr(x: &[f64], y: &[f64], method: &str) -> f64 {
    match method {
        "pearson" | "spearman" => pearson_pairwise(x, y),
        "kendall" => kendall_tau_b_pairwise(x, y),
        _ => f64::NAN,
    }
}

#[pyfunction]
fn calculate_correlation_matrix_numpy(
    py: Python<'_>,
    columns: Vec<PyReadonlyArray1<f64>>,
    method: String,
) -> PyResult<Vec<Vec<f64>>> {
    if method != "pearson" && method != "spearman" && method != "kendall" {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "method must be one of: pearson, spearman, kendall",
        ));
    }
    if columns.is_empty() {
        return Ok(Vec::new());
    }

    let mut col_vecs: Vec<Vec<f64>> = columns
        .iter()
        .map(|column| {
            column
                .as_slice()
                .map(|slice| slice.to_vec())
                .map_err(|err| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "column not contiguous: {}",
                        err
                    ))
                })
        })
        .collect::<PyResult<Vec<_>>>()?;

    let row_count = col_vecs[0].len();
    if col_vecs.iter().any(|column| column.len() != row_count) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "all columns must have the same length",
        ));
    }

    if method == "spearman" {
        col_vecs = col_vecs
            .par_iter()
            .map(|column| rank_average(column))
            .collect();
    }

    let col_count = col_vecs.len();
    let pair_indices: Vec<(usize, usize)> = (0..col_count)
        .flat_map(|left| (left..col_count).map(move |right| (left, right)))
        .collect();

    let pair_values = py.allow_threads(|| {
        pair_indices
            .into_par_iter()
            .map(|(left, right)| {
                let value = pairwise_corr(&col_vecs[left], &col_vecs[right], &method);
                (left, right, value)
            })
            .collect::<Vec<_>>()
    });

    let mut matrix = vec![vec![f64::NAN; col_count]; col_count];
    for (left, right, value) in pair_values {
        matrix[left][right] = value;
        matrix[right][left] = value;
    }

    Ok(matrix)
}

#[pyfunction]
fn extract_high_correlation_pairs_numpy(
    matrix_rows: Vec<PyReadonlyArray1<f64>>,
    threshold: f64,
) -> PyResult<Vec<(usize, usize, f64)>> {
    if matrix_rows.is_empty() {
        return Ok(Vec::new());
    }

    let rows: Vec<Vec<f64>> = matrix_rows
        .iter()
        .map(|row| {
            row.as_slice().map(|slice| slice.to_vec()).map_err(|err| {
                pyo3::exceptions::PyValueError::new_err(format!("row not contiguous: {}", err))
            })
        })
        .collect::<PyResult<Vec<_>>>()?;

    let n = rows.len();
    if rows.iter().any(|row| row.len() != n) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "corr matrix must be square",
        ));
    }

    let abs_threshold = threshold.abs();
    let mut pairs = Vec::new();
    for left in 0..n {
        for right in (left + 1)..n {
            let value = rows[left][right];
            if value.is_nan() {
                continue;
            }
            if value.abs() >= abs_threshold {
                pairs.push((left, right, value));
            }
        }
    }

    pairs.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap_or(Ordering::Equal));
    Ok(pairs)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(pyo3::wrap_pyfunction!(
        calculate_correlation_matrix_numpy,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        extract_high_correlation_pairs_numpy,
        module
    )?)?;
    Ok(())
}
