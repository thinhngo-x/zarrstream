//! Hole policies (M5).
//!
//! A hole is a chunk that was never written. Reading finds them in a shard index,
//! or lazily when a key is missing — never by listing the whole store. A hole must
//! never cost a request in either direction, and on the write side a chunk that
//! holds only `fill_value` is not uploaded at all.

use std::str::FromStr;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HolePolicy {
    /// Return the array's `fill_value`. The default, and what zarr-python does.
    #[default]
    Fill,
    /// Leave the hole out of the stream entirely.
    Skip,
    /// Return the data plus a boolean mask of valid elements.
    Mask,
    /// Fail with `Error::Hole`.
    Error,
    /// Emit a `Gap` item instead of an array, so the consumer sees the boundary.
    Signal,
}

impl FromStr for HolePolicy {
    type Err = crate::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "fill" => Ok(HolePolicy::Fill),
            "skip" => Ok(HolePolicy::Skip),
            "mask" => Ok(HolePolicy::Mask),
            "error" => Ok(HolePolicy::Error),
            "signal" => Ok(HolePolicy::Signal),
            other => Err(crate::Error::Plan(format!(
                "holes({other:?}): expected fill, skip, mask, error or signal"
            ))),
        }
    }
}

impl HolePolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            HolePolicy::Fill => "fill",
            HolePolicy::Skip => "skip",
            HolePolicy::Mask => "mask",
            HolePolicy::Error => "error",
            HolePolicy::Signal => "signal",
        }
    }
}

/// A run of missing samples, emitted under `HolePolicy::Signal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gap {
    pub start: u64,
    pub end: u64,
}

impl Gap {
    pub fn length(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_policy_round_trips_through_its_name() {
        for p in [
            HolePolicy::Fill,
            HolePolicy::Skip,
            HolePolicy::Mask,
            HolePolicy::Error,
            HolePolicy::Signal,
        ] {
            assert_eq!(HolePolicy::from_str(p.as_str()).unwrap(), p);
        }
        assert!(HolePolicy::from_str("nearest").is_err());
    }

    #[test]
    fn gap_length_never_underflows() {
        assert_eq!(Gap { start: 10, end: 25 }.length(), 15);
        assert_eq!(Gap { start: 25, end: 10 }.length(), 0);
    }
}
