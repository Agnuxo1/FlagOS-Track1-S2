# Experiment log

Every entry follows: hypothesis → design → control → execution → metric → decision.
Control = the SGLang reference kernel (the same baseline the FlagOS evaluator divides by).
Metric = latency speedup `T_ref / T_opt` per case, timed with `triton.testing.do_bench`
(median, L2 flushed), reference and candidate interleaved, 3 independent runs with
different input seeds × 5 rounds each. Acceptance rule (fixed before measuring): the
candidate must pass the correctness tests and beat the control in **every run on every
case** of the suite.

Machine-readable rows: [`experiments.csv`](experiments.csv).

## E001 — `softcap_row_argmax`: split the vocab row across programs
- **Hypothesis.** The reference runs one program per row and scans the vocab shard
  serially. Decode batches have 1–64 rows, so ≤ 64 of the GPU's SMs work. Splitting each
  row into `n_split` column chunks (≈ 2 programs per SM in total) plus a tiny merge kernel
  should cut latency several-fold for small row counts, and be neutral for large ones
  (single-pass path kept when `rows ≥ 2·SMs`).
- **Correctness design.** Element-wise math unchanged (`x/cap → tanh → ·cap`); merge
  keeps the leftmost maximal chunk, so values and first-occurrence indices are bit-exact.
  Tests: 40 parametrized shapes + saturated-tanh tie test + strided rows, `atol = rtol = 0`.
- **Status.** CPU interpreter smoke test passed (split path, single path, ties).
  GPU measurement: see results.

## E002 — `rmsnorm_fp32`: multi-row tiles + warp count from tile size
- **Hypothesis.** The reference uses 4 warps and one row per program for every width.
  Narrow rows (D = 128, per-head norms) waste launch/scheduling work; wide rows
  (D ≥ 4096) are under-threaded. Packing `ROWS` rows per program (~16 KB input/program)
  and choosing 4/8/16 warps from the tile size should help both ends.
- **Correctness design.** Same formula, `enable_fp_fusion=False` kept; reduction order
  may differ → tolerance test vs. reference and vs. PyTorch math (fp32: 1e-5).

## E003 — `gemma_fused_add_rmsnorm`: multi-row tiles for narrow rows
- **Hypothesis.** Same reasoning as E002 for the residual-add + Gemma `(1 + w)` norm,
  whose main use is per-head q/k norms with n = 64–256 and thousands of rows.
  Wide rows keep the reference's 1-row/16-warp plan, so no regression is expected there.
- **Correctness design.** The pre-norm sum must match bit-exactly (`atol = rtol = 0`);
  normalized output within dtype tolerance; strided `qkv.split` views tested.

## F1–F3 — executed gate-tiling interventions,28October2026
Job56a155bfb0184f10 succeeded. SixBF16shapes,3sessions×24interleavedrounds×100calls.72correctnesspassed. Arithmetic speedups512/1024/2048:1.0043979/.9997988/1.0011155. F1positive8×40961.0111,32×81921.0140,small1×128regression.9893. Everyshapeandnegativekept. scripts/summarize_gate.py independently reconstructs all medians/ratios fromrawCSV. No officialscore. Keepdefaultgeometry;testshape-selectionprospectively beforeadoption. JEVsecondreviewprovenance=jev,model1.14.0:retain_default_expose_experimental andpositive_with_scope.

E001–E003candidatekerneltests/benchmarkedjobd1006ec4412e4f09queueprojectflagos-track1-s2;resultsnotready. No registration/rulesacceptance/officialsubmissionmadebythispublication.
