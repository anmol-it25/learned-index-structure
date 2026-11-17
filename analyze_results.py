"""Analyze repeated B+ Tree vs Learned Index benchmark runs.

Run from the repository root:
    python3 analyze_results.py

Inputs: results/final_run_1.csv, final_run_2.csv, final_run_3.csv
Outputs: results/benchmark_summary.csv and graphs/*.png
Requires: matplotlib (python3 -m pip install matplotlib)
"""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")  # Write images without opening plot windows.
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
GRAPHS = ROOT / "graphs"
INPUTS = [RESULTS / f"final_run_{number}.csv" for number in (1, 2, 3)]
METRICS = (
    "tree_lookup_ns", "model_lookup_ns",
    "tree_insert_ns", "model_insert_ns",
    "tree_build_peak_bytes", "model_build_peak_bytes",
    "max_model_error",
)


def read_runs():
    grouped = defaultdict(list)
    for run_number, path in enumerate(INPUTS, start=1):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run the benchmarks and save all three CSV files first.")
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            required = {"n", "distribution", *METRICS}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{path.name} is missing columns: {sorted(required - set(reader.fieldnames or []))}")
            seen = set()
            for row in reader:
                key = (row["distribution"].strip().lower(), int(row["n"]))
                if key in seen:
                    raise ValueError(f"Duplicate configuration {key} in {path.name}")
                seen.add(key)
                data = {metric: float(row[metric]) for metric in METRICS}
                grouped[key].append((run_number, data))
    for key, observations in grouped.items():
        runs = {run for run, _ in observations}
        if runs != {1, 2, 3}:
            raise ValueError(f"Configuration {key} is missing from one or more runs: present in {sorted(runs)}")
    return grouped


def summarize(grouped):
    summary = []
    for (distribution, n), observations in sorted(grouped.items()):
        records = [metrics for _, metrics in observations]
        row = {"distribution": distribution, "n": n, "runs": len(records)}
        for metric in METRICS:
            values = [record[metric] for record in records]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
        row["tree_lookup_speedup"] = (
            row["model_lookup_ns_mean"] / row["tree_lookup_ns_mean"]
            if row["tree_lookup_ns_mean"] else float("nan")
        )
        summary.append(row)
    return summary


def save_summary(summary):
    destination = RESULTS / "benchmark_summary.csv"
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        for row in summary:
            writer.writerow({key: f"{val:.4f}" if isinstance(val, float) else val
                             for key, val in row.items()})
    print(f"Saved {destination.relative_to(ROOT)}")


def graph(summary, metric, title, ylabel, filename, scale=1.0):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = {
        ("uniform", "B+ Tree"): ("o", "-"),
        ("uniform", "Learned Index"): ("s", "-"),
        ("skewed", "B+ Tree"): ("o", "--"),
        ("skewed", "Learned Index"): ("s", "--"),
    }
    if metric == "max_model_error":
        series = [(distribution, "Learned Index") for distribution in ("uniform", "skewed")]
    else:
        series = [(distribution, model) for distribution in ("uniform", "skewed")
                  for model in ("B+ Tree", "Learned Index")]
    for distribution, model in series:
        rows = sorted((row for row in summary if row["distribution"] == distribution),
                      key=lambda row: row["n"])
        if not rows:
            continue
        name = metric if metric == "max_model_error" else (
            ("tree_" if model == "B+ Tree" else "model_") + metric
        )
        marker, linestyle = styles[(distribution, model)]
        x = [row["n"] for row in rows]
        y = [row[f"{name}_mean"] / scale for row in rows]
        sd = [row[f"{name}_sd"] / scale for row in rows]
        ax.errorbar(x, y, yerr=sd, marker=marker, linestyle=linestyle,
                    capsize=3, label=f"{distribution.title()} — {model}")
    ax.set(xlabel="Dataset size (number of keys)", ylabel=ylabel, title=title)
    ax.set_xticks(sorted({row["n"] for row in summary}))
    ax.ticklabel_format(axis="x", style="plain")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    destination = GRAPHS / filename
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    print(f"Saved {destination.relative_to(ROOT)}")


def main():
    grouped = read_runs()
    summary = summarize(grouped)
    if not summary:
        raise ValueError("No benchmark rows found in the CSV files.")
    GRAPHS.mkdir(exist_ok=True)
    save_summary(summary)
    graph(summary, "lookup_ns", "Lookup Time: B+ Tree vs Learned Index",
          "Time per lookup (ns)", "lookup_time.png")
    graph(summary, "insert_ns", "Insertion Time: B+ Tree vs Learned Index",
          "Time per insertion (ms)", "insertion_time.png", scale=1_000_000)
    graph(summary, "build_peak_bytes", "Peak Memory Allocation During Index Construction",
          "Peak allocated memory (MiB)", "build_peak_memory.png", scale=1024 * 1024)
    graph(summary, "max_model_error", "Maximum Raw Prediction Error (Learned Index)",
          "Absolute prediction error (positions)", "prediction_error.png")
    print("\nMean lookup times (ns), averaged over 3 runs:")
    print(f"{'Distribution':<14} {'Keys':>8} {'B+ Tree':>12} {'Learned':>12} {'Tree speedup':>14}")
    for row in summary:
        print(f"{row['distribution']:<14} {row['n']:>8,} "
              f"{row['tree_lookup_ns_mean']:>12.1f} "
              f"{row['model_lookup_ns_mean']:>12.1f} "
              f"{row['tree_lookup_speedup']:>13.2f}x")
    print("\nNote: Error bars represent SD across three complete runs; peak memory is construction allocation.")


if __name__ == "__main__":
    main()
