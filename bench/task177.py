# SPDX-License-Identifier: Apache-2.0
"""Three fixed packing interventions, fresh inputs, both layouts, all outcomes.

Control: our same-formula one-row Triton kernel. The task's actual Torch
reference supplies correctness. Ratios against this control are NOT official
SGLang speedups or multi-chip results. Seeds0/1/2;7 alternating timing rounds.
"""
import csv
import functools
import json
import math
import os
import platform
import random
import statistics
import sys
from pathlib import Path

import torch
import triton
from triton.testing import do_bench

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernels.zimage_qk_rmsnorm import packed_solution

SHAPES = [(1, 1, 1, 128), (1, 17, 3, 128), (2, 31, 8, 128),
          (1, 128, 24, 128), (2, 512, 16, 128), (1, 4096, 24, 128)]
ARMS = {'control_row1': (1, 4), 'Z1_rows4': (4, 4), 'Z2_rows8': (8, 4), 'Z3_rows16': (16, 8)}
SEEDS = [int(s) for s in os.environ.get('TASK177_SEEDS', '0,1,2').split(',')]


def reference(x, weight, eps):
    xf = x.to(torch.float32)
    rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * rms).to(x.dtype) * weight


def main():
    torch.set_num_threads(1)
    out = Path('output'); out.mkdir(exist_ok=True)
    result = dict(protocol='fixed before GPU measurements', shapes=SHAPES, arms=ARMS,
                  seeds=SEEDS, rounds=int(os.environ.get('TASK177_ROUNDS', '7')), gpu=torch.cuda.get_device_name(0),
                  torch=torch.__version__, triton=triton.__version__, python=platform.python_version(),
                  timing='triton do_bench, median, alternating randomised arm order; warmup5ms,rep20ms',
                  control='own one-row Triton geometry, not official SGLang baseline', cases=[])
    raw = []
    failures = []
    for seed in result['seeds']:
        torch.manual_seed(seed)
        order_rng = random.Random(20261029 + seed)
        for shape in SHAPES:
            for layout in ['contiguous', 'fused_qkv_slice']:
                b, s, h, d = shape
                storage = torch.randn((b, s, h * 3 if layout != 'contiguous' else h, d),
                                      device='cuda', dtype=torch.bfloat16)
                x = storage[:, :, h:2*h, :] if layout != 'contiguous' else storage
                w = torch.randn(d, device='cuda', dtype=torch.bfloat16)
                before = x.clone()
                expected = reference(x, w, 1e-6)
                callables = {name: functools.partial(packed_solution, x, w, 1e-6, rows=r, warps=n)
                             for name, (r, n) in ARMS.items()}
                checks = {}
                for name, fn in callables.items():
                    actual = fn()
                    try:
                        torch.testing.assert_close(actual, expected, rtol=.016, atol=1e-5)
                        assert actual.is_contiguous() and actual.data_ptr() != x.data_ptr()
                        assert torch.equal(x, before), 'input mutated'
                        checks[name] = dict(passed=True, max_abs=float((actual.float()-expected.float()).abs().max()))
                    except (AssertionError, RuntimeError) as exc:
                        checks[name] = dict(passed=False, error=str(exc)[:600])
                        failures.append(dict(seed=seed, shape=shape, layout=layout, arm=name, check=checks[name]))
                if not all(v['passed'] for v in checks.values()):
                    result['cases'].append(dict(seed=seed, shape=shape, layout=layout, checks=checks, timings=None))
                    continue
                timings = {name: [] for name in ARMS}
                for round_index in range(result['rounds']):
                    names = list(ARMS); order_rng.shuffle(names)
                    for name in names:
                        ms = do_bench(callables[name], warmup=5, rep=20, return_mode='median')
                        timings[name].append(ms)
                        raw.append(dict(seed=seed, shape='x'.join(map(str,shape)), layout=layout,
                                        round=round_index, arm=name, median_ms=ms))
                medians = {name: statistics.median(v) for name, v in timings.items()}
                ratios = {name: medians['control_row1'] / medians[name] for name in ARMS if name != 'control_row1'}
                result['cases'].append(dict(seed=seed, shape=shape, layout=layout, checks=checks,
                                            median_ms=medians, speedup_vs_own_row1=ratios))
                print(seed, shape, layout, ratios, flush=True)
    result['failures'] = failures
    result['correctness_checks'] = sum(len(c['checks']) for c in result['cases'])
    result['correctness_passed'] = not failures and len(result['cases']) == len(SEEDS) * len(SHAPES) * 2
    result['summary'] = {}
    for name in list(ARMS)[1:]:
        ratios = [c['speedup_vs_own_row1'][name] for c in result['cases'] if 'speedup_vs_own_row1' in c]
        result['summary'][name] = dict(geomean=math.exp(statistics.fmean(math.log(x) for x in ratios)) if ratios else None,
                                       wins=sum(x > 1 for x in ratios), losses=sum(x < 1 for x in ratios),
                                       minimum=min(ratios) if ratios else None, maximum=max(ratios) if ratios else None)
    (out/'task177.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'task177_raw.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','shape','layout','round','arm','median_ms'])
        writer.writeheader(); writer.writerows(raw)
    print(json.dumps(result['summary']),flush=True)
    if not result['correctness_passed']:
        raise SystemExit('Correctness failed; preserve results and do not submit')


if __name__ == '__main__':
    main()
