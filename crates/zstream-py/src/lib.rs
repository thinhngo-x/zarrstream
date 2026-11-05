//! The Python surface: a thin shell over the engine.
//!
//! Thin is the point. Nothing here decodes, encodes or waits on I/O — a call
//! either builds a plan or hands work to `zstream-core`, which runs it on its own
//! threads with the GIL released. The v0.1 API (`open`, `plan`, `create`, `stream`,
//! `writer`) is wired up as the milestones behind it land; see docs/PLAN.md.

use pyo3::prelude::*;

#[pymodule]
fn _zstream(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", zstream_core::VERSION)?;
    Ok(())
}
