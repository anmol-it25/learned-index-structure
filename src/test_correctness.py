
"""
Comprehensive lookup correctness verification.

Tests B+ Tree and Linear Regression Learned Index
on uniform and skewed datasets.
"""

import sys
from bisect import bisect_left
from pathlib import Path

# Import existing project implementations
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from benchmark import dataset, build_tree
from indexes import LearnedIndex


def verify_lookup(index, keys, key, expected, name):
    actual = index.search(key)

    if actual != expected:
        raise AssertionError(
            f"{name} FAILED | key={key} | "
            f"expected={expected} | actual={actual}"
        )


def test_distribution(distribution, n=100000):
    print("\n" + "=" * 50)
    print(f"TESTING: {distribution.upper()}")
    print(f"Dataset size: {n:,}")
    print("=" * 50)

    keys = dataset(n, distribution)

    # Ensure the dataset is sorted and unique
    assert len(keys) == n
    assert all(a < b for a, b in zip(keys, keys[1:]))

    tree = build_tree(keys)
    model = LearnedIndex(keys)

    # Test 1: Every existing key
    hits_checked = 0

    for i, key in enumerate(keys):
        verify_lookup(tree, keys, key, i, "B+ Tree")
        verify_lookup(model, keys, key, i, "Learned Index")
        hits_checked += 1

    print(f"Existing keys passed: {hits_checked:,}")

    # Test 2: Missing keys
    missing = [
        keys[0] - 1,
        keys[-1] + 1,
        keys[-1] + n * 20 + 1,
    ]

    for left, right in zip(keys, keys[1:]):
        if right - left > 1:
            missing.append(left + 1)

    misses_checked = 0

    for key in missing:
        pos = bisect_left(keys, key)
        expected = (
            pos if pos < n and keys[pos] == key
            else -1
        )

        assert expected == -1

        verify_lookup(tree, keys, key, expected, "B+ Tree")
        verify_lookup(model, keys, key, expected, "Learned Index")

        misses_checked += 1

    print(f"Missing keys passed: {misses_checked:,}")
    print(f"Maximum model error: {model.max_error:.2f}")
    print(f"PASS: {distribution.upper()}")

    return hits_checked, misses_checked


def main():
    total_hits = 0
    total_misses = 0

    for distribution in ("uniform", "skewed"):
        hits, misses = test_distribution(distribution)
        total_hits += hits
        total_misses += misses

    print("\n" + "=" * 50)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"Total existing keys tested: {total_hits:,}")
    print(f"Total missing keys tested: {total_misses:,}")
    print(f"Total test keys: {total_hits + total_misses:,}")
    print("Both indexes passed all lookup checks.")
    print("FINAL RESULT: ALL TESTS PASSED")


if __name__ == "__main__":
    main()
