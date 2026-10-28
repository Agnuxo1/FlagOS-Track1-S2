# SPDX-License-Identifier: Apache-2.0
"""Print Markdown result tables and update experiments.csv from a benchmark summary JSON."""

import csv
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = {"softcap_row_argmax": "E001", "rmsnorm_fp32": "E002", "gemma_fused_add_rmsnorm": "E003"}


def main(path, commit="workdir"):
    data = json.load(open(path))
    meta, summary = data["meta"], data["summary"]
    print(f"_GPU: {meta['gpu']} · torch {meta['torch']} · triton {meta['triton']} · CUDA {meta['cuda']} · "
          f"{meta['runs']} runs × {meta['rounds']} rounds · {meta['utc']}_\n")
    csv_path = os.path.join(ROOT, "experiments", "experiments.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["id", "date_utc", "hypothesis", "configuration", "commit", "metric", "decision"])
        for op in EXP:
            rows = [r for r in summary if r["op"] == op]
            if not rows:
                continue
            print(f"#### `{op}`\n")
            print("| case | reference (µs) | optimized (µs) | speedup | min–max over runs |")
            print("|---|---:|---:|---:|---:|")
            for r in rows:
                print(f"| {r['case']} | {r['ref_us']:.2f} | {r['opt_us']:.2f} | **{r['speedup']:.2f}×** | "
                      f"{r['speedup_min']:.2f}–{r['speedup_max']:.2f} |")
            sp = [r["speedup"] for r in rows]
            wins = sum(r["win_all_runs"] for r in rows)
            print(f"\nMean {st.mean(sp):.2f}× · geomean {st.geometric_mean(sp):.2f}× · "
                  f"wins every run on {wins}/{len(rows)} cases\n")
            decision = "accepted" if wins == len(rows) else f"partial ({wins}/{len(rows)} cases win every run)"
            w.writerow([EXP[op], meta["utc"], f"see LOG.md {EXP[op]}", os.path.basename(path), commit,
                        f"mean={st.mean(sp):.3f};geomean={st.geometric_mean(sp):.3f};min_case={min(sp):.3f}",
                        decision])


if __name__ == "__main__":
    main(*sys.argv[1:])
