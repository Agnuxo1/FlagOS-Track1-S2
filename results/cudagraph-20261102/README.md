# CUDA Graph measurements of the sigmoid gate, 2–3 November 2026

- `summary.json`, `holdout.csv`, `tuning.csv`: job 7cf84144b3ce49bc. Settings chosen on training seeds, validated on held-out seeds.
- `audits/<job>/`: five separate audit runs with raw timings and a report each.

RTX 3090, bfloat16, torch 2.14.1, Triton 3.8.0. Baseline file SHA-256 `991be249cc8942c7fdbe1a6b834efa89a05e0a8a0ed02824cc476840041975f0`.
`python scripts/summarize_cudagraph.py` recomputes the held-out speedups from `holdout.csv`.
