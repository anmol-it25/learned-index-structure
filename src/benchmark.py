import argparse
import csv
import gc
import random
import statistics
import time
import tracemalloc
from bisect import bisect_left
from pathlib import Path
from indexes import BPlusTree, LearnedIndex


def dataset(n, distribution, seed=42):
    rng = random.Random(seed)
    if distribution == 'uniform':
        return sorted(rng.sample(range(n * 20), n))
    return skewed(n, rng)


def skewed(n, rng):
    keys = set()
    while len(keys) < n:
        keys.add(int(rng.expovariate(1 / (n * 2))))
    return sorted(keys)


def peak_build(factory):
    gc.collect()
    tracemalloc.start()
    obj = factory()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return obj, peak


def timing(index, queries, rounds=5):
    samples = []
    for _ in range(rounds):
        t0 = time.perf_counter_ns()
        for key in queries:
            index.search(key)
        samples.append((time.perf_counter_ns() - t0) / len(queries))
    return statistics.median(samples)


def run(n, distribution, lookups, updates):
    if n < 1 or lookups < 1 or updates < 0:
        raise ValueError('sizes and lookups must be positive; updates cannot be negative')
    keys = dataset(n, distribution)
    if len(keys) != n or len(set(keys)) != n:
        raise AssertionError('invalid dataset')
    tree, tree_peak = peak_build(lambda: build_tree(keys))
    model, model_peak = peak_build(lambda: LearnedIndex(keys))

    # FIX: capture prediction error BEFORE updates trigger model refitting.
    initial_model_error = model.max_error

    rng = random.Random(100 + n)
    hits = rng.choices(keys, k=lookups // 2)
    misses = []
    keyset = set(keys)
    while len(misses) < lookups - len(hits):
        candidate = rng.randrange(keys[-1] + n * 20 + 1)
        if candidate not in keyset:
            misses.append(candidate)
    queries = hits + misses
    rng.shuffle(queries)

    for q in queries:
        pos = bisect_left(keys, q)
        expected = pos if pos < len(keys) and keys[pos] == q else -1
        assert tree.search(q) == expected, ('tree', q)
        assert model.search(q) == expected, ('model', q)

    tree_ns = timing(tree, queries)
    model_ns = timing(model, queries)

    # Append-only insertions: existing B+ tree rank values remain valid.
    newkeys = list(range(keys[-1] + 10, keys[-1] + 10 + updates))
    if updates:
        t0 = time.perf_counter_ns()
        for i, key in enumerate(newkeys):
            tree.insert(key, n + i)
        tree_update = (time.perf_counter_ns() - t0) / updates
        t0 = time.perf_counter_ns()
        for key in newkeys:
            model.insert(key)
        model_update = (time.perf_counter_ns() - t0) / updates
    else:
        tree_update = model_update = 0.0

    for i, key in enumerate(newkeys):
        assert tree.search(key) == n + i
        assert model.search(key) == n + i

    return dict(
        n=n,
        distribution=distribution,
        tree_lookup_ns=round(tree_ns, 1),
        model_lookup_ns=round(model_ns, 1),
        tree_build_peak_bytes=tree_peak,
        model_build_peak_bytes=model_peak,
        tree_insert_ns=round(tree_update, 1),
        model_insert_ns=round(model_update, 1),
        max_model_error=round(initial_model_error, 2),
        queries=lookups,
        updates=updates,
    )


def build_tree(keys):
    tree = BPlusTree(order=32)
    for i, k in enumerate(keys):
        tree.insert(k, i)
    return tree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000, 10000, 50000])
    parser.add_argument('--lookups', type=int, default=2000)
    parser.add_argument('--updates', type=int, default=10)
    parser.add_argument('--output', default='results/measurements.csv')
    args = parser.parse_args()
    rows = []
    for distribution in ('uniform', 'skewed'):
        for n in args.sizes:
            row = run(n, distribution, args.lookups, args.updates)
            print(row, flush=True)
            rows.append(row)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print('Saved', path)


if __name__ == '__main__':
    main()
