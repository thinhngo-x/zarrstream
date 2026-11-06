#!/usr/bin/env python3
"""Does batching plus a prefetch thread capture the whole win?

The question this answers: zarr-python streams at ~100-190 MB/s when asked for
one shard-row at a time, and several times that when handed the whole array.
If a few lines of prefetching recover most of that, the Rust engine in the plan
is not justified by read speed and has to earn its keep elsewhere.

    python scripts/bench_prefetch.py s3://bucket/array --rounds 3

Methodology, after being burned by the first attempt (see docs/PLAN.md):
object storage here appears to run burst credits, so absolute throughput drifts
by up to 9x between runs minutes apart. Every variant therefore runs once per
round, round-robin, and the report is a median plus the full spread. A single
pass of each variant proves nothing, however clean the number looks.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def stream(arr, block: int, lookahead: int):
    """Yield row-blocks of `arr`, keeping `lookahead` blocks in flight.

    The whole prototype. While the consumer works on one block, the thread pool
    is already fetching the next ones, so read time and process time overlap
    instead of adding. `lookahead` bounds memory: at most lookahead+1 blocks are
    resident, whatever the array's size.
    """
    step = block * (arr.shards or arr.chunks)[0]
    rows = arr.shape[0]
    with ThreadPoolExecutor(max(lookahead, 1)) as ex:
        futures = deque(
            ex.submit(lambda i=i: arr[i : i + step])
            for i in range(0, min(lookahead * step, rows), step)
        )
        for i in range(0, rows, step):
            nxt = i + lookahead * step
            if nxt < rows:
                futures.append(ex.submit(lambda j=nxt: arr[j : j + step]))
            yield futures.popleft().result()


def sequential(arr, block: int):
    """The baseline: one block at a time, nothing in flight. What a naive
    streaming loop over zarr-python does today."""
    step = block * (arr.shards or arr.chunks)[0]
    for i in range(0, arr.shape[0], step):
        yield arr[i : i + step]


def consume(it, process_ms: float) -> None:
    """Touch every block, optionally burning CPU to stand in for real work.

    `.sum()` forces materialisation; without it a lazy read would be timed as
    instant. The optional spin is what exposes overlap: with no processing cost
    there is nothing for prefetching to hide behind.
    """
    for blk in it:
        blk.sum()
        if process_ms:
            end = time.perf_counter() + process_ms / 1000
            while time.perf_counter() < end:
                pass


VARIANTS = [
    ("sequential b1", lambda a: sequential(a, 1)),
    ("prefetch b1 la2", lambda a: stream(a, 1, 2)),
    ("prefetch b2 la4", lambda a: stream(a, 2, 4)),
    ("prefetch b4 la2", lambda a: stream(a, 4, 2)),
    ("prefetch b8 la4", lambda a: stream(a, 8, 4)),
    ("whole array", lambda a: iter([a[:]])),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url")
    p.add_argument("--endpoint", default=os.environ.get("AWS_ENDPOINT_URL"))
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--process-ms", type=float, default=0.0,
                   help="CPU burned per block, standing in for your pipeline")
    p.add_argument("--store", choices=["s3fs", "obstore"], default="obstore")
    args = p.parse_args()

    import zarr
    from zarr.core.config import config

    config.set({"async.concurrency": args.concurrency, "threading.max_workers": args.concurrency})

    def open_array():
        if args.store == "obstore":
            from bench_baseline import objectstore_store

            store, path = objectstore_store(args.url, args.endpoint)
            return zarr.open_array(store=store, path=path, mode="r")
        from bench_baseline import storage_options

        return zarr.open_array(args.url, mode="r", storage_options=storage_options(args.endpoint))

    nbytes = open_array().nbytes
    print(f"{args.url}  {nbytes / 1e9:.2f} GB  store={args.store}  "
          f"concurrency={args.concurrency}  process_ms={args.process_ms}  rounds={args.rounds}\n")

    results: dict[str, list[float]] = {name: [] for name, _ in VARIANTS}
    for rnd in range(args.rounds):
        for name, make in VARIANTS:
            arr = open_array()  # fresh array object, so nothing carries over
            t0 = time.perf_counter()
            consume(make(arr), args.process_ms)
            mbps = nbytes / 1e6 / (time.perf_counter() - t0)
            results[name].append(mbps)
            print(f"  round {rnd}  {name:18} {mbps:8.1f} MB/s", flush=True)
        print(flush=True)

    base = statistics.median(results["sequential b1"])
    print(f"{'variant':20}{'median':>10}{'min':>8}{'max':>8}{'vs seq':>9}")
    for name, _ in VARIANTS:
        v = results[name]
        print(f"{name:20}{statistics.median(v):10.1f}{min(v):8.0f}{max(v):8.0f}"
              f"{statistics.median(v) / base:8.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
