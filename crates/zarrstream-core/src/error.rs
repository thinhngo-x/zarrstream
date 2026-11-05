//! One error type for the whole engine.
//!
//! Rule from the plan: a background error always reaches the caller's loop — on the
//! next item, the next `write()`, or on close. Nothing is swallowed by a worker
//! thread, and nothing is committed after an error.

use std::path::PathBuf;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A stub that a later milestone fills in. Carries the milestone so the
    /// message says when it lands rather than just failing.
    #[error("{what} is not implemented yet (planned for {milestone})")]
    NotImplemented {
        what: &'static str,
        milestone: &'static str,
    },

    #[error("store {url}: {source}")]
    Store {
        url: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Zarr metadata or codec failure, naming the array or chunk involved.
    #[error("zarr {path}: {reason}")]
    Zarr { path: String, reason: String },

    #[error("plan: {0}")]
    Plan(String),

    /// A chunk is missing and the hole policy is `error`.
    #[error("missing chunk {key} (hole policy: error)")]
    Hole { key: String },

    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// The consumer stopped early (an exception, or `break` in the caller's loop),
    /// so the pipeline is shutting down.
    #[error("pipeline cancelled")]
    Cancelled,
}

impl Error {
    /// Used by the module stubs as they get filled in (see docs/PLAN.md).
    #[allow(dead_code)]
    pub(crate) fn todo(what: &'static str, milestone: &'static str) -> Self {
        Error::NotImplemented { what, milestone }
    }
}
