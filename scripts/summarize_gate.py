"""Recompute every stored latency ratio from raw timings; no winner filtering."""
import csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=ROOT/'results/gate-20261028'
    d=json.loads((p/'flagos_gate_results.json').read_text(encoding='utf-8'))
    raw=list(csv.DictReader((p/'flagos_gate_raw.csv').open(encoding='utf-8')))
    buckets={}
    for r in raw:
        k=(int(r['session']),int(r['n']),int(r['d']),r['method'])
        buckets.setdefault(k,[]).append(float(r['microseconds']))
    for r in d['summary']:
        k=(r['session'],r['n'],r['d'])
        candidate=statistics.median(buckets[(*k,r['method'])])
        baseline=statistics.median(buckets[(*k,'baseline')])
        assert abs(candidate-r['median_us'])<1e-12
        assert abs(baseline/candidate-r['speedup'])<1e-12
    assert len(d['correctness'])==72 and all(x['passed'] for x in d['correctness'])
    lines=['# Three tiling interventions — all measured shapes',
           '',f"GPU: {d['gpu']}; three sessions,24 interleaved rounds/session,100 calls/sample.",
           '','| N | D | tile512 /4warps | tile1024 /4warps | tile2048 /8warps |',
           '|---:|---:|---:|---:|---:|']
    methods=['F1_tile512','F2_tile1024','F3_tile2048']
    for n,width in d['shapes']:
        values=[statistics.mean(r['speedup'] for r in d['summary']
                  if r['n']==n and r['d']==width and r['method']==m) for m in methods]
        lines.append(f'| {n} | {width} | '+' | '.join(f'{v:.6f}×' for v in values)+' |')
    lines+=['','Ratios are baseline median / candidate median, averaged over sessions.',
            'All72 correctness checks passed. Raw timings independently reproduce all stored ratios.',
            'One RTX3090; these are not official multi-chip results. Small aggregate gains do not establish a general replacement.']
    (p/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('PASS: all stored medians and ratios reconstructed from raw timings;72 correctness checks')


if __name__=='__main__':main()
