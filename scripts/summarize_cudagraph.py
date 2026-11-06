"""Recompute the held-out sigmoid-gate speedups from raw GPU timings. CPU only."""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

base = Path(__file__).resolve().parents[1] / "results/cudagraph-20261102"
times = defaultdict(list)
with (base / "holdout.csv").open(newline="") as f:
    for row in csv.DictReader(f):
        times[row["N"], row["D"], row["seed"], row["candidate"]].append(float(row["gpu_us"]))
summary = json.loads((base / "summary.json").read_text())
medians = []
for item in summary["selection"]:
    n, d = map(str, item["shape"])
    seeds = sorted({k[2] for k in times if k[:2] == (n, d)})
    ratios = [statistics.median(times[n, d, s, "baseline"]) / statistics.median(times[n, d, s, item["selected"]]) for s in seeds]
    median = statistics.median(ratios)
    assert abs(median - item["holdout_speedup_median"]) < 1e-9, (n, d)
    medians.append(median)
    print(f"{n}x{d} {item['selected']} median {median:.3f} lowest {min(ratios):.3f}")
print(f"geometric mean {math.exp(sum(map(math.log, medians)) / len(medians)):.3f}")
