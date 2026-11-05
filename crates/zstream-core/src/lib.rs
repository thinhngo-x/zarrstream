//! zstream engine.
//!
//! Everything that touches I/O, decoding or encoding lives here, in Rust, so the
//! Python side never holds the GIL while waiting. Three stages run at once —
//! read, process (the caller's Python code) and write — linked by bounded queues,
//! so memory stays fixed and the slowest stage sets the pace.
//!
//! Module map, with the milestone that fills each one in (docs/PLAN.md):
//!
//! | Module   | Role                                        | Milestone |
//! |----------|---------------------------------------------|-----------|
//! | `store`  | Object storage behind one trait             | M1        |
//! | `plan`   | Lazy, immutable read plan                   | M2        |
//! | `read`   | Planner-driven read pipeline                | M2        |
//! | `write`  | Chunk/shard assembly and upload             | M3        |
//! | `queue`  | Bounded handoff with backpressure           | M2        |
//! | `holes`  | Missing-chunk policies                      | M5        |
//! | `stats`  | Per-stage counters, including queue fill    | M6        |
//! | `resume` | Reader position and committed-chunk state   | M7        |

pub mod error;
pub mod holes;
pub mod plan;
pub mod queue;
pub mod read;
pub mod resume;
pub mod stats;
pub mod store;
pub mod write;

pub use error::{Error, Result};
pub use holes::{Gap, HolePolicy};
pub use plan::{Plan, Selection};

/// Crate version, surfaced to Python as `zstream.__version__`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
