use std::cmp::Ordering;
use std::collections::BinaryHeap;

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

#[derive(Clone, Debug)]
struct BinNode {
    start_value: f64,
    count: f64,
    events: f64,
    prev: Option<usize>,
    next: Option<usize>,
    active: bool,
    version: u64,
}

#[derive(Clone, Copy, Debug)]
struct EdgeCandidate {
    chi2: f64,
    left_idx: usize,
    right_idx: usize,
    left_version: u64,
    right_version: u64,
}

impl PartialEq for EdgeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.chi2.to_bits() == other.chi2.to_bits()
            && self.left_idx == other.left_idx
            && self.right_idx == other.right_idx
            && self.left_version == other.left_version
            && self.right_version == other.right_version
    }
}

impl Eq for EdgeCandidate {}

impl Ord for EdgeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        match other
            .chi2
            .partial_cmp(&self.chi2)
            .unwrap_or(Ordering::Equal)
        {
            Ordering::Equal => match other.left_idx.cmp(&self.left_idx) {
                Ordering::Equal => match other.right_idx.cmp(&self.right_idx) {
                    Ordering::Equal => match other.left_version.cmp(&self.left_version) {
                        Ordering::Equal => other.right_version.cmp(&self.right_version),
                        ordering => ordering,
                    },
                    ordering => ordering,
                },
                ordering => ordering,
            },
            ordering => ordering,
        }
    }
}

impl PartialOrd for EdgeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn build_edge_candidate(bins: &[BinNode], left_idx: usize) -> Option<EdgeCandidate> {
    if left_idx >= bins.len() || !bins[left_idx].active {
        return None;
    }

    let right_idx = bins[left_idx].next?;
    if right_idx >= bins.len() || !bins[right_idx].active {
        return None;
    }

    if bins[right_idx].prev != Some(left_idx) {
        return None;
    }

    Some(EdgeCandidate {
        chi2: calculate_chi2_yates(
            bins[left_idx].count,
            bins[left_idx].events,
            bins[right_idx].count,
            bins[right_idx].events,
        ),
        left_idx,
        right_idx,
        left_version: bins[left_idx].version,
        right_version: bins[right_idx].version,
    })
}

fn is_edge_candidate_valid(bins: &[BinNode], candidate: EdgeCandidate) -> bool {
    if candidate.left_idx >= bins.len() || candidate.right_idx >= bins.len() {
        return false;
    }

    let left = &bins[candidate.left_idx];
    let right = &bins[candidate.right_idx];

    left.active
        && right.active
        && left.version == candidate.left_version
        && right.version == candidate.right_version
        && left.next == Some(candidate.right_idx)
        && right.prev == Some(candidate.left_idx)
}

fn extract_split_points(bins: &[BinNode]) -> Vec<f64> {
    let mut start_idx = bins.iter().position(|bin| bin.active && bin.prev.is_none());
    if start_idx.is_none() {
        start_idx = bins.iter().position(|bin| bin.active);
    }

    let Some(mut current_idx) = start_idx else {
        return vec![];
    };

    let mut starts: Vec<f64> = Vec::new();
    loop {
        let node = &bins[current_idx];
        if !node.active {
            break;
        }
        starts.push(node.start_value);
        match node.next {
            Some(next_idx) => current_idx = next_idx,
            None => break,
        }
    }

    if starts.len() < 2 {
        return vec![];
    }

    starts
        .windows(2)
        .map(|window| (window[0] + window[1]) / 2.0)
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

    let mut bins: Vec<BinNode> = Vec::new();
    if !data.is_empty() {
        let mut current_value = data[0].0;
        let mut current_count = 0.0;
        let mut current_events = 0.0;

        for (feature_value, target_value) in data {
            if feature_value == current_value {
                current_count += 1.0;
                if target_value == 1 {
                    current_events += 1.0;
                }
            } else {
                bins.push(BinNode {
                    start_value: current_value,
                    count: current_count,
                    events: current_events,
                    prev: None,
                    next: None,
                    active: true,
                    version: 0,
                });
                current_value = feature_value;
                current_count = 1.0;
                current_events = if target_value == 1 { 1.0 } else { 0.0 };
            }
        }

        bins.push(BinNode {
            start_value: current_value,
            count: current_count,
            events: current_events,
            prev: None,
            next: None,
            active: true,
            version: 0,
        });
    }

    if bins.len() < 2 {
        return vec![];
    }

    for idx in 0..bins.len() {
        bins[idx].prev = idx.checked_sub(1);
        bins[idx].next = if idx + 1 < bins.len() {
            Some(idx + 1)
        } else {
            None
        };
    }

    let target_bins = n_bins.max(1);
    let mut active_bins = bins.len();
    let mut heap = BinaryHeap::new();

    for left_idx in 0..bins.len().saturating_sub(1) {
        if let Some(candidate) = build_edge_candidate(&bins, left_idx) {
            heap.push(candidate);
        }
    }

    while active_bins > target_bins {
        let mut merged = false;

        while let Some(candidate) = heap.pop() {
            if !is_edge_candidate_valid(&bins, candidate) {
                continue;
            }

            if candidate.chi2 >= threshold {
                return extract_split_points(&bins);
            }

            let left_idx = candidate.left_idx;
            let right_idx = candidate.right_idx;
            let right_next_idx = bins[right_idx].next;

            bins[left_idx].count += bins[right_idx].count;
            bins[left_idx].events += bins[right_idx].events;
            bins[left_idx].next = right_next_idx;
            bins[left_idx].version += 1;

            bins[right_idx].active = false;
            bins[right_idx].prev = None;
            bins[right_idx].next = None;
            bins[right_idx].version += 1;

            if let Some(next_idx) = right_next_idx {
                bins[next_idx].prev = Some(left_idx);
            }

            active_bins -= 1;
            merged = true;

            if let Some(prev_idx) = bins[left_idx].prev {
                if let Some(updated_candidate) = build_edge_candidate(&bins, prev_idx) {
                    heap.push(updated_candidate);
                }
            }
            if let Some(updated_candidate) = build_edge_candidate(&bins, left_idx) {
                heap.push(updated_candidate);
            }

            break;
        }

        if !merged {
            break;
        }
    }

    extract_split_points(&bins)
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

    for feature_vec in &feature_vecs {
        if feature_vec.len() != target_vec.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Each feature must have the same length as target.",
            ));
        }
    }

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
