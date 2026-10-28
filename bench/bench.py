# SPDX-License-Identifier: Apache-2.0
"""Interleaved latency benchmark: SGLang reference kernel vs. optimized kernel.

Protocol (fixed before any measurement):
  * every (op, case) is correctness-checked first; a failing case aborts;
  * ``--runs`` independent runs, each with its own seed (fresh inputs);
  * inside a run, reference and candidate are timed alternately ``--rounds``
    times with ``triton.testing.do_bench`` (median of its samples, L2 flushed);
  * per case we report the median over all samples and the per-run speedup
    ``T_ref / T_opt``; a candidate is accepted only if it wins in every run on
    every case.
"""

import argparse
import csv
import datetime as dt
import json
import os
import platform
import statistics as st
import sys

import torch
import triton
from triton.testing import do_bench

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kernels import gemma_fused_add_rmsnorm as opt_gemma  # noqa: E402
from kernels import rmsnorm_fp32 as opt_rms  # noqa: E402
from kernels import softcap_row_argmax as opt_soft  # noqa: E402
from reference import gemma_fused_add_rmsnorm as ref_gemma  # noqa: E402
from reference import rmsnorm_fp32 as ref_rms  # noqa: E402
from reference import softcap_row_argmax as ref_soft  # noqa: E402

DEV = "cuda"


def softcap_cases():
    for rows in (1, 4, 16, 64, 256):
        for width in (32064, 128256, 262144):
            yield f"rows={rows},V={width}", (rows, width)


def softcap_make(case, seed):
    rows, width = case
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(rows, width, device=DEV, generator=g) * 20
    return (x, 30.0, width, 0)


def rms_cases():
    for dtype in ("bf16", "fp32"):
        for rows, d in ((1, 4096), (16, 4096), (256, 4096), (4096, 4096), (64, 8192),
                        (8192, 128), (32768, 128), (4096, 1024)):
            yield f"{dtype},M={rows},D={d}", (rows, d, dtype)


def rms_make(case, seed):
    rows, d, dtype = case
    dt_ = torch.bfloat16 if dtype == "bf16" else torch.float32
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(rows, d, device=DEV, generator=g).to(dt_)
    w = torch.randn(d, device=DEV, generator=g).to(dt_)
    return (x, w, 1e-6)


def gemma_cases():
    for rows, n in ((1, 3072), (32, 3072), (2048, 3072), (512, 6144),
                    (4096, 128), (16384, 128), (16384, 256), (65536, 64)):
        yield f"bf16,M={rows},N={n}", (rows, n)


def gemma_make(case, seed):
    rows, n = case
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(rows, n, device=DEV, generator=g).to(torch.bfloat16)
    r = torch.randn(rows, n, device=DEV, generator=g).to(torch.bfloat16)
    w = torch.randn(n, device=DEV, generator=g) * 0.1
    return (x, r, w, 1e-6)


def check_softcap(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def check_close(a, b):
    a = a if isinstance(a, tuple) else (a,)
    b = b if isinstance(b, tuple) else (b,)
    for u, v in zip(a, b):
        tol = dict(rtol=1e-5, atol=1e-5) if u.dtype == torch.float32 else dict(rtol=1.6e-2, atol=2e-2)
        torch.testing.assert_close(u, v, **tol)


OPS = {
    "softcap_row_argmax": (softcap_cases, softcap_make, ref_soft.softcap_row_argmax,
                           opt_soft.softcap_row_argmax, check_softcap),
    "rmsnorm_fp32": (rms_cases, rms_make, ref_rms.rmsnorm_fp32, opt_rms.rmsnorm_fp32, check_close),
    "gemma_fused_add_rmsnorm": (gemma_cases, gemma_make, ref_gemma.gemma_fused_add_rmsnorm,
                                opt_gemma.gemma_fused_add_rmsnorm, check_close),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default=",".join(OPS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    meta = {
        "utc": stamp, "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "triton": triton.__version__,
        "cuda": torch.version.cuda, "python": platform.python_version(),
        "runs": args.runs, "rounds": args.rounds,
        "timer": "triton.testing.do_bench(warmup=25, rep=100, return_mode='median')",
    }
    print(json.dumps(meta))
    raw, summary = [], []
    for op in args.ops.split(","):
        cases_fn, make, ref, opt, check = OPS[op]
        for name, case in cases_fn():
            per_run = []
            ref_all, opt_all = [], []
            for run in range(args.runs):
                seed = 1000 * (run + 1)
                inputs = make(case, seed)
                check(opt(*inputs), ref(*inputs))
                t_ref, t_opt = [], []
                for rnd in range(args.rounds):
                    order = [(ref, t_ref), (opt, t_opt)] if rnd % 2 == 0 else [(opt, t_opt), (ref, t_ref)]
                    for fn, sink in order:
                        sink.append(do_bench(lambda: fn(*inputs), warmup=25, rep=100, return_mode="median"))
                for i in range(args.rounds):
                    raw.append([op, name, run, i, t_ref[i] * 1e3, t_opt[i] * 1e3])
                per_run.append(st.median(t_ref) / st.median(t_opt))
                ref_all += t_ref
                opt_all += t_opt
            row = {
                "op": op, "case": name,
                "ref_us": st.median(ref_all) * 1e3, "opt_us": st.median(opt_all) * 1e3,
                "speedup": st.median(ref_all) / st.median(opt_all),
                "speedup_runs": per_run,
                "speedup_min": min(per_run), "speedup_max": max(per_run),
                "speedup_std": st.pstdev(per_run),
                "win_all_runs": all(s > 1.0 for s in per_run),
            }
            summary.append(row)
            print(f"{op:26s} {name:28s} ref {row['ref_us']:9.2f}us opt {row['opt_us']:9.2f}us "
                  f"x{row['speedup']:.3f} [{row['speedup_min']:.3f},{row['speedup_max']:.3f}]", flush=True)
    base = os.path.join(args.out, f"bench_{stamp}")
    with open(base + "_raw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "case", "run", "round", "ref_us", "opt_us"])
        w.writerows(raw)
    with open(base + "_summary.json", "w") as f:
        json.dump({"meta": meta, "summary": summary}, f, indent=1)
    for op in args.ops.split(","):
        rows = [r for r in summary if r["op"] == op]
        print(f"[{op}] mean speedup {st.mean(r['speedup'] for r in rows):.3f}  "
              f"geomean {st.geometric_mean([r['speedup'] for r in rows]):.3f}  "
              f"wins all runs on {sum(r['win_all_runs'] for r in rows)}/{len(rows)} cases")


if __name__ == "__main__":
    main()
