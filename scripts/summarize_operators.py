"""Reconstruct all recorded operator medians and run ratios without a GPU."""
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = ROOT / 'results/operators-20261028'
raw_path, = data.glob('*_raw.csv')
summary_path, = data.glob('*_summary.json')
record = json.loads(summary_path.read_text())
groups = defaultdict(list)
for row in csv.DictReader(raw_path.open()):
    groups[row['op'], row['case']].append(row)
lines = ['# Recorded operator benchmark', '',
         'RTX 3090; 3 fresh-input runs, 5 interleaved rounds per run.',
         'Ratios are reference time / candidate time. Each row passed correctness.',
         'This is a CUDA proxy, not an official eight-chip evaluation.', '',
         '| Operator | Case | Speedup | Range across runs | Wins in all runs |',
         '|---|---|---:|---:|---|']
for expected in record['summary']:
    rows = groups[expected['op'], expected['case']]
    assert len(rows) == record['meta']['runs'] * record['meta']['rounds']
    ref = st.median(float(x['ref_us']) for x in rows)
    opt = st.median(float(x['opt_us']) for x in rows)
    ratios = []
    for run in range(record['meta']['runs']):
        r = [x for x in rows if int(x['run']) == run]
        assert len(r) == record['meta']['rounds']
        ratios.append(st.median(float(x['ref_us']) for x in r) /
                      st.median(float(x['opt_us']) for x in r))
    for a, b in [(ref, expected['ref_us']), (opt, expected['opt_us']),
                 (ref / opt, expected['speedup']), *zip(ratios, expected['speedup_runs'])]:
        assert math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7)
    win = all(x > 1 for x in ratios)
    assert win == expected['win_all_runs']
    lines.append(f"| {expected['op']} | {expected['case']} | {ref / opt:.4f}× | "
                 f"{min(ratios):.4f}–{max(ratios):.4f} | {'yes' if win else 'no'} |")
lines += ['', 'None of the three candidates satisfies the preregistered all-case,',
          'all-run speed gate. The shape-specific gains remain measured results;',
          'the regressions prevent unconditional promotion, not their reporting.', '']
(data / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({'cases_reconstructed': len(record['summary']),
                  'timing_pairs': sum(map(len, groups.values())),
                  'all_case_gate_passed': False}))
