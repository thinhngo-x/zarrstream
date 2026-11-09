use std::{error::Error, sync::Arc};
use zarrs::{array::Array, storage::AsyncReadableStorage};
use zarrs_object_store::AsyncObjectStore;

// `async fn main` is not a thing in Rust: something has to start a runtime
// before any future can run. This attribute rewrites the function into a plain
// `fn main` that builds a tokio runtime and blocks on your body.
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let s3 = object_store::aws::AmazonS3Builder::new()
        .with_bucket_name("zarrstream")
        .with_endpoint("https://s3.fr-par.scw.cloud")
        .with_region("fr-par")
        .with_access_key_id(std::env::var("AWS_ACCESS_KEY_ID")?)
        .with_secret_access_key(std::env::var("AWS_SECRET_ACCESS_KEY")?)
        .with_virtual_hosted_style_request(false) // required for S3-compatible endpoints
        .build()?;

    let store: AsyncReadableStorage = Arc::new(AsyncObjectStore::new(s3));
    // `.await` is a postfix keyword, not a method: no parentheses after it.
    let array = Array::async_open(store, "/testdata/v3_sharded_zstd").await?;

    println!("shape   {:?}", array.shape());
    println!("dtype   {}", array.data_type());
    // Shape of the chunk grid itself: how many chunks along each axis.
    println!("grid    {:?}", array.chunk_grid_shape());
    // Everything else (codecs, shards, fill value) lives in the metadata.
    println!("meta    {:?}", array.metadata());

    // `_elements::<f32>` decodes and reinterprets as a typed Vec, instead of the
    // raw `ArrayBytes` that `async_retrieve_chunk` hands back. It errors if the
    // array's dtype is not f32, so the turbofish is a checked claim, not a cast.
    let chunk: Vec<f32> = array.async_retrieve_chunk_elements::<f32>(&[0, 0]).await?;

    println!("\nchunk [0,0]  {} elements", chunk.len());
    println!("first 4      {:?}", &chunk[..4]);
    // f64 accumulator: summing 65k f32s in f32 loses precision, so the checksum
    // would not match Python's.
    let sum: f64 = chunk.iter().map(|&v| f64::from(v)).sum();
    println!("sum          {sum:.3}");
    Ok(())
}
