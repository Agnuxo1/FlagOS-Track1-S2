"""Reconstruct every case and aggregate directly from raw recorded timings."""
import collections
import csv
import json
import math
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'results/task177-20261029'
for phase in ('exploratory', 'confirmation'):
    folder = root / phase
    recorded = json.loads((folder / 'task177.json').read_text())
    groups = collections.defaultdict(list)
    with (folder / 'task177_raw.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            value = float(row['median_ms'])
            assert math.isfinite(value) and value > 0
            groups[int(row['seed']), row['shape'], row['layout'], row['arm']].append(value)
    assert len(groups) == 144 and all(len(v) == 7 for v in groups.values())
    assert recorded['correctness_passed'] and recorded['correctness_checks'] == 144
    all_ratios = collections.defaultdict(list)
    for case in recorded['cases']:
        assert all(c['passed'] for c in case['checks'].values())
        prefix = case['seed'], 'x'.join(map(str, case['shape'])), case['layout']
        medians = {arm: statistics.median(groups[*prefix, arm]) for arm in recorded['arms']}
        for arm in medians:
            assert math.isclose(medians[arm], case['median_ms'][arm], rel_tol=1e-12)
        for arm in recorded['summary']:
            ratio = medians['control_row1'] / medians[arm]
            assert math.isclose(ratio, case['speedup_vs_own_row1'][arm], rel_tol=1e-12)
            all_ratios[arm].append(ratio)
    for arm, ratios in all_ratios.items():
        summary = recorded['summary'][arm]
        assert math.isclose(math.exp(statistics.fmean(math.log(x) for x in ratios)), summary['geomean'], rel_tol=1e-12)
        assert sum(x > 1 for x in ratios) == summary['wins']
        assert sum(x < 1 for x in ratios) == summary['losses']
    print(phase, '144 correctness records; 1008 raw timing rows independently reconstructed')
