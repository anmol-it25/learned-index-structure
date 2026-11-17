# Group 3: B+ Tree vs Learned Index

Educational Python benchmark of a B+ Tree and a linear least-squares rank predictor with exact local correction.

## Run

```bash
python3 src/benchmark.py --sizes 1000 10000 50000 --lookups 2000 --updates 10
```

Outputs `results/measurements.csv`. Requires Python 3.10+; no external packages.

## Methodology

- Unique sorted integer keys, uniformly sampled or exponentially skewed.
- Both indexes return the exact zero-based sorted rank or -1.
- Half of lookup queries are present keys; half are absent.
- Five repetitions of lookup timing; report median nanoseconds per query.
- `tracemalloc` peak allocations during construction are reported as a **proxy**, not precise retained index memory.
- Insertions are appended above the largest key so B+ Tree ranks stay valid. The learned model is refitted after every insertion; this is a simple baseline, not an optimized dynamic learned index.
- Dataset generation, Python interpreter overhead, cache effects, and index representation affect outcomes. This is not a comparison of production disk-backed indexes.

## Important caveats

A single global linear model may have large correction windows on skewed data. Its correction search uses Python's optimized `bisect_left`, whereas the B+ Tree traversal is pure Python. Report this implementation asymmetry in the paper. Model prediction plus correction is exact for present keys. For absent keys, a restricted correction window is safe for nonmembership because sorted present keys with that value do not exist.

For a more accurate memory comparison, measure retained object graphs separately with a dedicated profiler.
