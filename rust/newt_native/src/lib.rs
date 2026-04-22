mod binary_metrics;
mod binning;
mod correlation;
mod iv;
mod logit;
mod psi;
mod shared;

use pyo3::prelude::*;

#[pymodule]
fn _newt_native(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    iv::register(module)?;
    psi::register(module)?;
    binary_metrics::register(module)?;
    correlation::register(module)?;
    binning::register(module)?;
    logit::register(module)?;
    Ok(())
}
