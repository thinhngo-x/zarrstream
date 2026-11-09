#!/usr/bin/env python3
"""Emit what zarr-python reads from every test array, as the oracle for Rust.

zarr-python is the correctness reference, so the Rust reader is checked against
what zarr-python actually returns rather than against a reimplementation of the
generator. Writes JSON to stdout or --out.

    python scripts/checksums.py s3://zarrstream/testdata --out tests/checksums.json

Per array: shape, dtype, element count, the f64 sum, the count of exact zeros
(which is how hole handling shows up: an unwritten chunk reads back as
fill_value), and the first and last elements for a bit-exact spot check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import zarr


def stats(a: np.ndarray) -> dict:
    flat = a.reshape(-1)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "count": int(flat.size),
        # f64 accumulation: an f32 sum of 131k values drifts enough to fail a
        # comparison for reasons that have nothing to do with correctness.
        "sum": float(flat.astype(np.float64).sum()),
        "zeros": int(np.count_nonzero(flat == 0)),
        "first": float(flat[0]),
        "last": float(flat[-1]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", help="local dir or s3://bucket/prefix holding the matrix")
    p.add_argument("--endpoint", default=os.environ.get("AWS_ENDPOINT_URL"))
    p.add_argument("--out")
    args = p.parse_args()

    opts = None
    if args.root.startswith("s3://"):
        client = {"endpoint_url": args.endpoint} if args.endpoint else {}
        if region := os.environ.get("AWS_DEFAULT_REGION"):
            client["region_name"] = region
        opts = {"client_kwargs": client}
        import s3fs

        fs = s3fs.S3FileSystem(**opts)
        names = sorted({k.split("/")[-1] for k in fs.ls(args.root.removeprefix("s3://"))})
    else:
        names = sorted(os.listdir(args.root))

    out = {}
    for name in names:
        try:
            arr = zarr.open_array(f"{args.root}/{name}", mode="r", storage_options=opts)
        except Exception as e:
            print(f"skip {name}: {type(e).__name__}", file=sys.stderr)
            continue
        out[name] = stats(arr[:])
        print(f"{name:26} sum={out[name]['sum']:.4f} zeros={out[name]['zeros']}", file=sys.stderr)

    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\nwrote {args.out} ({len(out)} arrays)", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
