#!/usr/bin/env python3
"""M0 baselines: what we have to beat, and what we cannot beat.

Run this from a machine in the bucket's own region — across regions it measures
the WAN, not the store, and every number below becomes meaningless.

    python scripts/bench_baseline.py s3://zarrstream/bench_sharded \\
        --endpoint https://s3.fr-par.scw.cloud --concurrency 64

Reports MB/s and requests per GB for:

  raw_parallel_get  every object of the array fetched concurrently, nothing
                    decoded. The ceiling: zarrstream cannot beat this, and the
                    plan's target is to come within 20% of it.
  zarr_python       the reference implementation, and the 2x target.
  tensorstore       the other serious contender, if it is installed.

Deliberately not measured here: stall time and the read/process/write overlap.
Those are properties of zarrstream's own pipeline, so they belong in its
`stats()`, not in a baseline of other people's libraries.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

GB = 1e9


def fs_for(endpoint: str | None, concurrency: int):
    import s3fs

    client_kwargs = {}
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    if region := os.environ.get("AWS_DEFAULT_REGION"):
        client_kwargs["region_name"] = region
    return s3fs.S3FileSystem(
        client_kwargs=client_kwargs,
        # Without this the pool caps concurrency well below what we asked for and
        # the "ceiling" measurement silently under-reports.
        config_kwargs={"max_pool_connections": max(concurrency, 10)},
        skip_instance_cache=True,
    )


def storage_options(endpoint: str | None) -> dict:
    client_kwargs = {}
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    if region := os.environ.get("AWS_DEFAULT_REGION"):
        client_kwargs["region_name"] = region
    return {"client_kwargs": client_kwargs}


def raw_parallel_get(url: str, endpoint: str | None, concurrency: int):
    """The ceiling: fetch every object, decode nothing."""
    fs = fs_for(endpoint, concurrency)
    keys = [k for k in fs.find(url.removeprefix("s3://")) if not k.endswith(".json")]
    t0 = time.perf_counter()
    blobs = fs.cat(keys, batch_size=concurrency)
    elapsed = time.perf_counter() - t0
    return sum(len(b) for b in blobs.values()), elapsed, len(keys)


def zarr_python_read(url: str, endpoint: str | None, concurrency: int):
    import zarr
    from zarr.core.config import config

    config.set({"async.concurrency": concurrency})
    arr = zarr.open_array(url, mode="r", storage_options=storage_options(endpoint))
    nbytes = arr.nbytes
    t0 = time.perf_counter()
    # Read in shard-or-chunk-row blocks: the point is streaming throughput, and
    # arr[:] on a multi-GB array would measure the machine's RAM instead.
    step = (arr.shards or arr.chunks)[0]
    checksum = 0.0
    for i in range(0, arr.shape[0], step):
        checksum += float(arr[i : i + step, :].sum())
    return nbytes, time.perf_counter() - t0, checksum


def objectstore_store(url: str, endpoint: str | None):
    """zarr-python's ObjectStore backend, the only cloud store `zarrs` supports.

    Returns (store, path) because obstore is bucket-scoped: the array's path
    inside the bucket is passed to open_array separately.
    """
    import obstore
    from zarr.storage import ObjectStore

    bucket, _, path = url.removeprefix("s3://").partition("/")
    cfg = {
        "access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    }
    if endpoint:
        cfg["endpoint"] = endpoint
        cfg["virtual_hosted_style_request"] = False
    return ObjectStore(obstore.store.S3Store(bucket, **cfg)), path


def _read_blocks(arr, t0=None):
    """Stream the array a shard-row (or chunk-row) at a time and sum it.

    Same access pattern for every contender, so the numbers compare. Summing
    forces the data to actually materialise rather than staying lazy.
    """
    step = (arr.shards or arr.chunks)[0]
    t0 = t0 or time.perf_counter()
    for i in range(0, arr.shape[0], step):
        float(arr[i : i + step, :].sum())
    return arr.nbytes, time.perf_counter() - t0


def zarr_objectstore_read(url: str, endpoint: str | None, concurrency: int):
    """zarr-python over obstore, with its stock codec pipeline.

    The control for the next contender: it isolates the store change from the
    codec-pipeline change, so a zarrs speedup can't be credited to obstore.
    """
    import zarr
    from zarr.core.config import config

    config.set({"async.concurrency": concurrency, "codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"})
    store, path = objectstore_store(url, endpoint)
    return _read_blocks(zarr.open_array(store=store, path=path, mode="r"))


def zarrs_read(url: str, endpoint: str | None, concurrency: int):
    """zarr-python over obstore, decoding through the zarrs Rust crate.

    The prior art for this project's own thesis: a Rust engine under a Python
    API. If it already closes the gap to raw GETs, zarrstream's remaining claim
    is the read/process/write overlap, not read speed.
    """
    import zarr
    import zarrs  # noqa: F401  (registers the pipeline by import path)
    from zarr.core.config import config

    config.set(
        {
            "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
            "threading.max_workers": concurrency,
        }
    )
    store, path = objectstore_store(url, endpoint)
    return _read_blocks(zarr.open_array(store=store, path=path, mode="r"))


def tensorstore_read(url: str, endpoint: str | None, concurrency: int):
    import tensorstore as ts

    bucket, _, path = url.removeprefix("s3://").partition("/")
    kvstore = {"driver": "s3", "bucket": bucket, "path": path + "/"}
    if endpoint:
        kvstore["endpoint"] = endpoint
    arr = ts.open({"driver": "zarr3", "kvstore": kvstore}, read=True).result()
    nbytes = arr.domain.size * arr.dtype.numpy_dtype.itemsize
    step = arr.chunk_layout.write_chunk.shape[0]
    t0 = time.perf_counter()
    for i in range(0, arr.shape[0], step):
        arr[i : i + step, :].read().result()
    return nbytes, time.perf_counter() - t0


def report(name: str, nbytes: int, elapsed: float, requests: int | None = None):
    mbps = nbytes / 1e6 / elapsed
    line = f"{name:20} {nbytes / GB:7.2f} GB  {elapsed:7.2f} s  {mbps:9.1f} MB/s"
    if requests is not None:
        line += f"  {requests / (nbytes / GB):9.0f} req/GB"
    print(line, flush=True)
    return mbps


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url", help="s3://bucket/prefix of one array")
    p.add_argument("--endpoint", default=os.environ.get("AWS_ENDPOINT_URL"))
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--repeat", type=int, default=3, help="runs per contender; the median is reported")
    args = p.parse_args()

    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        print("set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY")
        return 2

    print(f"{args.url}  concurrency={args.concurrency}  repeat={args.repeat}\n")
    results: dict[str, float] = {}

    runs = [raw_parallel_get(args.url, args.endpoint, args.concurrency) for _ in range(args.repeat)]
    nbytes, _, nreq = runs[0]
    results["raw_parallel_get"] = report(
        "raw_parallel_get", nbytes, statistics.median(r[1] for r in runs), nreq
    )

    runs = [zarr_python_read(args.url, args.endpoint, args.concurrency) for _ in range(args.repeat)]
    results["zarr_python"] = report("zarr_python", runs[0][0], statistics.median(r[1] for r in runs))

    # Optional contenders: a missing one is reported and skipped, never fatal.
    for name, fn, hint in (
        ("zarr_obstore", zarr_objectstore_read, "pip install obstore"),
        ("zarrs_rust", zarrs_read, "pip install zarrs obstore"),
        ("tensorstore", tensorstore_read, "pip install tensorstore"),
    ):
        try:
            runs = [fn(args.url, args.endpoint, args.concurrency) for _ in range(args.repeat)]
            results[name] = report(name, runs[0][0], statistics.median(r[1] for r in runs))
        except ImportError:
            print(f"{name:20} not installed ({hint})")
        except Exception as e:  # unsupported store or layout, not a crash
            print(f"{name:20} failed: {type(e).__name__}: {e}")

    ceiling = results["raw_parallel_get"]
    print(f"\nzarrstream targets for this dataset:")
    print(f"  beat zarr-python 2x      -> {results['zarr_python'] * 2:9.1f} MB/s")
    print(f"  within 20% of the ceiling -> {ceiling * 0.8:9.1f} MB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
