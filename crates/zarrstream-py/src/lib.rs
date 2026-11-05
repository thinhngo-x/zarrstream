//! The Python surface: a thin shell over the engine.
//!
//! Thin is the point. Nothing here decodes, encodes or waits on I/O — a call
//! either builds a plan or hands work to `zarrstream-core`, which runs it on its own
//! threads with the GIL released. The v0.1 API (`open`, `plan`, `create`, `stream`,
//! `writer`) is wired up as the milestones behind it land; see docs/PLAN.md.

use pyo3::prelude::*;

#[pymodule]
fn _zarrstream(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", zarrstream_core::VERSION)?;
    Ok(())
}
