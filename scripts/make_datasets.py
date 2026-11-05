#!/usr/bin/env python3
"""Write the synthetic test matrix (M0).

zarr-python writes these, because zarr-python is the correctness reference: if a
zarrstream read disagrees with what zarr-python wrote and reads back, zarrstream
is wrong. Deliberately small — a few MB total, so the whole matrix fits in CI.

    python scripts/make_datasets.py testdata                     # local dir
    python scripts/make_datasets.py s3://zarrstream/testdata     # Scaleway/MinIO

S3 targets read credentials from the environment (AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY) and the endpoint from --endpoint or AWS_ENDPOINT_URL.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import zarr

SHAPE = (512, 256)
CHUNKS = (64, 256)
SHARDS = (256, 256)  # 4 chunks per shard

# name -> (zarr_format, shards, compressors). The codec names are what the plan
# commits to supporting; `None` means stored raw.
CASES: dict[str, tuple[int, tuple[int, ...] | None, object]] = {
    "v2_none": (2, None, None),
    "v2_blosc": (2, None, "blosc"),
    "v2_zstd": (2, None, "zstd"),
    "v2_gzip": (2, None, "gzip"),
    "v3_none": (3, None, None),
    "v3_zstd": (3, None, "zstd"),
    "v3_blosc": (3, None, "blosc"),
    "v3_lz4": (3, None, "lz4"),
    "v3_sharded_zstd": (3, SHARDS, "zstd"),
    "v3_sharded_none": (3, SHARDS, None),
}

# Every case is also written with holes: the same data, but only the chunk rows
# listed here. Unwritten chunks are absent in the store, which is what a hole is.
HOLE_ROWS = [0, 1, 4, 7]

# The throughput datasets. The correctness matrix above is ~7 MB on purpose, which
# is far too small to measure MB/s — one request's latency dominates. These two are
# the same data sharded and unsharded, so the baseline shows what sharding is worth.
BENCH_CHUNKS = (256, 2048)
BENCH_SHARDS = (4096, 2048)  # 16 chunks per shard


def codec(name: object, zarr_format: int):
    """Resolve a codec name for this zarr format, or None for raw storage."""
    if name is None:
        return None
    if zarr_format == 2:
        import numcodecs

        return {
            "blosc": numcodecs.Blosc(cname="lz4", clevel=5),
            "zstd": numcodecs.Zstd(level=3),
            "gzip": numcodecs.GZip(level=5),
        }[name]
    from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec

    return {
        "blosc": BloscCodec(cname="lz4", clevel=5),
        "lz4": BloscCodec(cname="lz4", clevel=5, shuffle="noshuffle"),
        "zstd": ZstdCodec(level=3),
        "gzip": GzipCodec(level=5),
    }[name]


def reference_data() -> np.ndarray:
    """The expected contents of every dataset here.

    Seeded, so a test regenerates it instead of loading a blob from the repo.
    Not compressible to nothing — a constant array would make every codec look
    identical and hide shuffle bugs.
    """
    rng = np.random.default_rng(0)
    base = np.arange(SHAPE[0] * SHAPE[1], dtype="float32").reshape(SHAPE)
    return base + rng.standard_normal(SHAPE, dtype="float32")


def write_bench(root: str, gb: float, opts) -> list[str]:
    """One array of roughly `gb` gigabytes, sharded and unsharded.

    Generate this from a machine in the same region as the bucket — uploading it
    over a home uplink takes far longer than the benchmark it feeds.
    """
    rows = int(gb * 1e9 / (4 * BENCH_CHUNKS[1]))
    rows -= rows % BENCH_SHARDS[0]  # whole shards, so the index is uniform
    shape = (max(rows, BENCH_SHARDS[0]), BENCH_CHUNKS[1])
    rng = np.random.default_rng(1)
    written = []

    for name, shards in (("bench_sharded", BENCH_SHARDS), ("bench_flat", None)):
        path = f"{root}/{name}"
        kwargs = dict(
            shape=shape,
            chunks=BENCH_CHUNKS,
            dtype="float32",
            zarr_format=3,
            fill_value=0.0,
            overwrite=True,
            compressors=[codec("zstd", 3)],
            storage_options=opts or None,
        )
        if shards is not None:
            kwargs["shards"] = shards
        arr = zarr.create_array(path, **kwargs)
        # A row block at a time: the whole array does not need to be in memory.
        step = BENCH_SHARDS[0]
        for i in range(0, shape[0], step):
            block = rng.standard_normal((min(step, shape[0] - i), shape[1]), dtype="float32")
            arr[i : i + block.shape[0], :] = block
        written.append(f"{path}  shape={shape} shards={shards}")
    return written


def write_case(root: str, name: str, spec, data: np.ndarray, holes: bool, opts):
    zarr_format, shards, codec_name = spec
    path = f"{root}/{name}" + ("_holes" if holes else "")
    kwargs = dict(
        shape=SHAPE,
        chunks=CHUNKS,
        dtype="float32",
        zarr_format=zarr_format,
        fill_value=0.0,
        overwrite=True,
        storage_options=opts or None,
    )
    if shards is not None:
        kwargs["shards"] = shards
    c = codec(codec_name, zarr_format)
    # v2 takes a single compressor (or None); v3 takes a codec chain.
    kwargs["compressors"] = c if zarr_format == 2 else ([c] if c is not None else [])

    arr = zarr.create_array(path, **kwargs)
    if holes:
        rows = CHUNKS[0]
        for i in HOLE_ROWS:
            arr[i * rows : (i + 1) * rows, :] = data[i * rows : (i + 1) * rows, :]
    else:
        arr[:] = data
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", help="local directory, or s3://bucket/prefix")
    p.add_argument("--endpoint", default=os.environ.get("AWS_ENDPOINT_URL"))
    p.add_argument(
        "--bench",
        type=float,
        metavar="GB",
        help="instead of the correctness matrix, write ~GB gigabytes for throughput "
        "baselines (run this in the bucket's own region)",
    )
    args = p.parse_args()

    opts: dict[str, object] = {}
    if args.target.startswith("s3://"):
        if not os.environ.get("AWS_ACCESS_KEY_ID"):
            print("set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY", file=sys.stderr)
            return 2
        if args.endpoint:
            opts["client_kwargs"] = {"endpoint_url": args.endpoint}

    if args.bench:
        for line in write_bench(args.target, args.bench, opts):
            print(line)
        return 0

    data = reference_data()
    for name, spec in CASES.items():
        for holes in (False, True):
            path = write_case(args.target, name, spec, data, holes, opts)
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
