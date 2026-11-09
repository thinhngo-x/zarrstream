//! Time zarrs against the Python baselines, and find out whether concurrency
//! can be driven from above.
//!
//! Integration question this answers: zarrs owns the store, so does firing N
//! retrievals from our own task pool scale, or does its internal scheduling cap
//! us? That decides whether the planned engine can sit on top of zarrs at all.
//!
//! ```text
//! ZS_BUCKET=zarrstream ZS_ENDPOINT=https://s3.fr-par.scw.cloud \
//!   cargo run --release --example bench -p zarrstream-core -- /pf/bench_sharded chunks 32
//! ```
//!
//! Modes: `whole` (one retrieve of the full array, the equivalent of `arr[:]`)
//! and `chunks N` (every chunk retrieved with N requests in flight, the
//! equivalent of the Python prefetch loop).

use std::{error::Error, sync::Arc, time::Instant};

use futures::stream::{self, StreamExt};
use zarrs::{array::Array, array_subset::ArraySubset, storage::AsyncReadableStorage};
use zarrs_object_store::AsyncObjectStore;

fn store() -> Result<AsyncReadableStorage, Box<dyn Error>> {
    let s3 = object_store::aws::AmazonS3Builder::new()
        .with_bucket_name(std::env::var("ZS_BUCKET")?)
        .with_endpoint(std::env::var("ZS_ENDPOINT")?)
        .with_region(std::env::var("AWS_DEFAULT_REGION").unwrap_or_else(|_| "fr-par".into()))
        .with_access_key_id(std::env::var("AWS_ACCESS_KEY_ID")?)
        .with_secret_access_key(std::env::var("AWS_SECRET_ACCESS_KEY")?)
        .with_virtual_hosted_style_request(false)
        .build()?;
    Ok(Arc::new(AsyncObjectStore::new(s3)))
}

/// Every chunk index of the array, in row-major order.
fn chunk_indices(grid: &[u64]) -> Vec<Vec<u64>> {
    let mut out = vec![vec![]];
    for &n in grid {
        out = out
            .into_iter()
            .flat_map(|prefix| {
                (0..n).map(move |i| {
                    let mut v = prefix.clone();
                    v.push(i);
                    v
                })
            })
            .collect();
    }
    out
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("/pf/bench_sharded");
    let mode = args.get(2).map(String::as_str).unwrap_or("chunks");
    let concurrency: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);

    let array = Array::async_open(store()?, path).await?;
    let nbytes: u64 = array.shape().iter().product::<u64>() * 4; // f32

    let t0 = Instant::now();
    let checksum: f64 = match mode {
        "whole" => {
            let subset = ArraySubset::new_with_shape(array.shape().to_vec());
            let v: Vec<f32> = array
                .async_retrieve_array_subset_elements::<f32>(&subset)
                .await?;
            v.iter().map(|&x| f64::from(x)).sum()
        }
        "chunks" => {
            let array = Arc::new(array);
            let indices = chunk_indices(&array.chunk_grid_shape());
            // `buffer_unordered` is the whole experiment: N retrievals in flight,
            // results taken in whatever order they finish. The Python equivalent
            // was four threads each calling arr[i:i+step].
            stream::iter(indices)
                .map(|idx| {
                    let array = Arc::clone(&array);
                    async move {
                        let v = array
                            .async_retrieve_chunk_elements::<f32>(&idx)
                            .await
                            .expect("retrieve failed");
                        // Summing here keeps the decoded buffer from being
                        // optimised away, and mirrors what the Python loop did.
                        v.iter().map(|&x| f64::from(x)).sum::<f64>()
                    }
                })
                .buffer_unordered(concurrency)
                .fold(0.0f64, |acc, s| async move { acc + s })
                .await
        }
        other => return Err(format!("unknown mode {other:?}, want whole|chunks").into()),
    };
    let elapsed = t0.elapsed().as_secs_f64();

    println!(
        "rust_zarrs {mode:6} c={concurrency:<4} {:.2} GB {:7.2} s {:8.1} MB/s  (checksum {checksum:.3})",
        nbytes as f64 / 1e9,
        elapsed,
        nbytes as f64 / 1e6 / elapsed,
    );
    Ok(())
}
