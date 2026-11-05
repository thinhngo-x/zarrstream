//! Bounded handoff between stages (M2).
//! `std::sync::mpsc::sync_channel` is already this; the blocking send is the
//! backpressure.
