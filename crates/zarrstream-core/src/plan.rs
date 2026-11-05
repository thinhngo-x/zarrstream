//! The read plan (M2).
//!
//! A plan is lazy and immutable: every builder method returns a new plan, and no
//! I/O happens until `zs.stream(plan)` starts the engine. Because the engine sees
//! the whole plan up front, it can fetch chunks before they are needed, coalesce
//! neighbouring ranges into one request, and skip holes without any request at all.

use crate::holes::HolePolicy;

/// A half-open range on one axis, by index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AxisRange {
    pub start: u64,
    pub stop: u64,
}

/// Which part of the array to read: one range per named axis, in array order.
/// An axis left out means "all of it".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Selection {
    pub axes: Vec<(String, AxisRange)>,
}

/// Fixed-size windows along one axis, the unit the consumer iterates over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Samples {
    pub size: u64,
    pub axis: usize,
}

/// One shard of the plan for distributed reading: `rank` of `world`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shard {
    pub rank: usize,
    pub world: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub source: String,
    pub selection: Selection,
    pub samples: Option<Samples>,
    pub batch: Option<usize>,
    pub holes: HolePolicy,
    pub shard: Option<Shard>,
}

impl Plan {
    pub fn new(source: impl Into<String>) -> Self {
        Plan {
            source: source.into(),
            selection: Selection::default(),
            samples: None,
            batch: None,
            holes: HolePolicy::default(),
            shard: None,
        }
    }

    pub fn select(mut self, axis: impl Into<String>, start: u64, stop: u64) -> Self {
        self.selection
            .axes
            .push((axis.into(), AxisRange { start, stop }));
        self
    }

    pub fn samples(mut self, size: u64, axis: usize) -> Self {
        self.samples = Some(Samples { size, axis });
        self
    }

    pub fn batch(mut self, n: usize) -> Self {
        self.batch = Some(n);
        self
    }

    pub fn holes(mut self, policy: HolePolicy) -> Self {
        self.holes = policy;
        self
    }

    pub fn shard(mut self, rank: usize, world: usize) -> Self {
        self.shard = Some(Shard { rank, world });
        self
    }

    /// Reject a plan that cannot be satisfied, before any I/O starts (M2).
    pub fn validate(&self) -> crate::Result<()> {
        for (axis, r) in &self.selection.axes {
            if r.stop < r.start {
                return Err(crate::Error::Plan(format!(
                    "axis {axis}: stop {} is before start {}",
                    r.stop, r.start
                )));
            }
        }
        if let Some(s) = self.samples
            && s.size == 0
        {
            return Err(crate::Error::Plan("samples(size=0)".into()));
        }
        if let Some(b) = self.batch
            && b == 0
        {
            return Err(crate::Error::Plan("batch(0)".into()));
        }
        if let Some(sh) = self.shard
            && sh.rank >= sh.world
        {
            return Err(crate::Error::Plan(format!(
                "shard(rank={}, world={}): rank must be below world",
                sh.rank, sh.world
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_are_immutable_and_chain() {
        let base = Plan::new("s3://bucket/data.zarr");
        let derived = base.clone().samples(40, 0).batch(64);
        assert_eq!(base.samples, None, "the original plan was mutated");
        assert_eq!(derived.batch, Some(64));
    }

    #[test]
    fn validate_rejects_impossible_plans() {
        assert!(
            Plan::new("s3://b/a.zarr")
                .select("x", 10, 5)
                .validate()
                .is_err()
        );
        assert!(Plan::new("s3://b/a.zarr").batch(0).validate().is_err());
        assert!(Plan::new("s3://b/a.zarr").shard(3, 3).validate().is_err());
        assert!(
            Plan::new("s3://b/a.zarr")
                .select("x", 0, 10)
                .validate()
                .is_ok()
        );
    }
}
