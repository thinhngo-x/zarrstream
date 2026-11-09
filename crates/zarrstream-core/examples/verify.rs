//! M1 exit test: does zarrs read every test array the way zarr-python does?
//!
//! The oracle is `tests/checksums.json`, written by `scripts/checksums.py` from
//! what zarr-python actually returns. Run that first, then:
//!
//! ```text
//! AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
//!   cargo run --release --example verify -p zarrstream-core
//! ```
//!
//! Exits non-zero if any array disagrees, so it can go straight into CI.

use std::{error::Error, sync::Arc};

use zarrs::{array::Array, array_subset::ArraySubset, storage::AsyncReadableStorage};
use zarrs_object_store::AsyncObjectStore;

/// What one array should look like, as zarr-python saw it.
struct Expected {
    shape: Vec<u64>,
    count: usize,
    sum: f64,
    zeros: usize,
    first: f64,
    last: f64,
}

/// The f64 sums are accumulated in a different order here than in NumPy, so the
/// last bits differ. A relative tolerance keeps that from looking like a bug,
/// while still catching a single wrong element out of 131072.
const SUM_TOLERANCE: f64 = 1e-12;

fn stats(values: &[f32]) -> (f64, usize, f64, f64) {
    let sum = values.iter().map(|&v| f64::from(v)).sum();
    let zeros = values.iter().filter(|&&v| v == 0.0).count();
    // `expect` rather than `?`: an empty array means the oracle and the store
    // disagree about something more basic than values, and panicking here says so.
    let first = f64::from(*values.first().expect("empty array"));
    let last = f64::from(*values.last().expect("empty array"));
    (sum, zeros, first, last)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let oracle_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/checksums.json");
    let oracle: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(oracle_path)?)?;
    let oracle = oracle
        .as_object()
        .ok_or("checksums.json is not an object")?;

    let s3 = object_store::aws::AmazonS3Builder::new()
        .with_bucket_name("zarrstream")
        .with_endpoint("https://s3.fr-par.scw.cloud")
        .with_region("fr-par")
        .with_access_key_id(std::env::var("AWS_ACCESS_KEY_ID")?)
        .with_secret_access_key(std::env::var("AWS_SECRET_ACCESS_KEY")?)
        .with_virtual_hosted_style_request(false)
        .build()?;
    // One store, shared by every array below: that is what the Arc is for.
    let store: AsyncReadableStorage = Arc::new(AsyncObjectStore::new(s3));

    let mut failures = 0usize;
    for (name, want) in oracle {
        let want = Expected {
            shape: want["shape"]
                .as_array()
                .ok_or("shape")?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0))
                .collect(),
            count: want["count"].as_u64().ok_or("count")? as usize,
            sum: want["sum"].as_f64().ok_or("sum")?,
            zeros: want["zeros"].as_u64().ok_or("zeros")? as usize,
            first: want["first"].as_f64().ok_or("first")?,
            last: want["last"].as_f64().ok_or("last")?,
        };

        let array = Array::async_open(store.clone(), &format!("/testdata/{name}")).await?;
        let subset = ArraySubset::new_with_shape(array.shape().to_vec());
        let values: Vec<f32> = array
            .async_retrieve_array_subset_elements::<f32>(&subset)
            .await?;
        let (sum, zeros, first, last) = stats(&values);

        let mut problems: Vec<String> = Vec::new();
        if array.shape() != want.shape.as_slice() {
            problems.push(format!("shape {:?} != {:?}", array.shape(), want.shape));
        }
        if values.len() != want.count {
            problems.push(format!("count {} != {}", values.len(), want.count));
        }
        if (sum - want.sum).abs() > want.sum.abs() * SUM_TOLERANCE {
            problems.push(format!("sum {sum:.6} != {:.6}", want.sum));
        }
        if zeros != want.zeros {
            problems.push(format!("zeros {zeros} != {}", want.zeros));
        }
        if first != want.first || last != want.last {
            problems.push(format!(
                "ends ({first}, {last}) != ({}, {})",
                want.first, want.last
            ));
        }

        if problems.is_empty() {
            println!("ok    {name:26} {} elements, {zeros} holes", values.len());
        } else {
            failures += 1;
            println!("FAIL  {name:26} {}", problems.join("; "));
        }
    }

    println!("\n{} arrays, {failures} failed", oracle.len());
    if failures > 0 {
        std::process::exit(1);
    }
    Ok(())
}
