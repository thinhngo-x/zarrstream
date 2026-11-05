# zstream v0.1 — plan

The living plan, with the full milestone detail, exit tests, benchmarks and risk
table, is the shared doc:

https://claude.ai/artifact/9xpgi6kHo3fHEqPEapYm9k

That doc is the source of truth. This file exists so the milestone markers in the
code (`Error::todo("…", "M2")`, the module table in `zstream-core/src/lib.rs`)
point somewhere, and so the shape stays visible without opening a browser:

| Milestone | Week  | Lands                                                      |
|-----------|-------|------------------------------------------------------------|
| M0        | 1     | Baselines on S3, MinIO + test datasets, Rust/PyO3 skeleton |
| M1        | 2–3   | Native reader: `zarrs` metadata and codecs, `object_store`, sharded v3, zero-copy to NumPy |
| M2        | 4–5   | Planner (`select`/`samples`/`batch`), `tokio` read engine, coalescing, bounded queues |
| M3        | 6–7   | Streaming writer: `zs.create`, chunk/shard assembly, multipart upload |
| M4        | 8     | Retries with backoff, aborted multiparts, fault injection   |
| M5        | 9     | Hole policies both directions, `Gap` events                 |
| M6        | 10    | One shared adaptive request budget, `stats()` per stage      |
| M7        | 11    | `state_dict()` / resume for reader and writer, `shard()`, PyTorch |
| M8        | 12    | Docs, benchmarks, wheels, `v0.1.0` on PyPI                  |

Two standing rules:

- **zarr-python is the correctness reference**, for reads and for arrays zstream
  writes. Raw parallel GETs/PUTs are the speed ceiling.
- **Out of scope stays out.** Local/SAN/NFS, random writes, local cache, block
  shuffle, Icechunk, GPU paths — all v0.2. New ideas go to the v0.2 backlog, not
  into v0.1.
