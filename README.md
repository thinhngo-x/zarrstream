# zarrstream

Streaming Zarr I/O: read from cloud object storage, process in Python, and write
back out — all three at the same time, with a native Rust engine that never holds
the GIL.

> **Skeleton.** Nothing reads or writes yet. Milestone map: [docs/PLAN.md](docs/PLAN.md).
> The name is a placeholder.

| Path                   | What                                                    |
|------------------------|---------------------------------------------------------|
| `crates/zarrstream-core`  | The engine: store, planner, read/write pipelines, stats |
| `crates/zarrstream-py`    | PyO3 bindings — thin, no I/O of its own                  |
| `python/zarrstream`       | The importable package; the extension lands inside it    |
| `tests`                | Python tests, zarr-python as the reference               |

```sh
pip install maturin
maturin develop --extras test
pytest && cargo test --workspace
```
