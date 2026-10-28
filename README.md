# FlagOS Track1 S2 — controlled Triton kernels

Own Triton optimization candidates for [FlagOS Open Computing Global Challenge, Track1](https://www.kaggle.com/competitions/flag-os-open-computing-global-challenge-track-1), with reproducible correctness gates and raw latency evidence.

## Measured improvement: sigmoid-gate tiling

Three fixed tiling interventions were executed on an RTX3090 against the SGLang reference geometry: one program per row, next-power-of-two block width and its width-dependent warp heuristic. All **72 correctness checks passed**. Each shape has three sessions,24 interleaved rounds/session and100 calls/sample. An independent CPU check reproduces every stored median and ratio from raw timings.

| Intervention | Arithmetic mean speedup over all shapes/sessions |
|---|---:|
| F1: tile512,4warps | **1.004398×** |
| F2: tile1024,4warps | 0.999799× |
| F3: tile2048,8warps | 1.001115× |

The positive shape-specific gains remain visible:

| Shape N×D | F1 speedup, mean over3sessions |
|---|---:|
| 1×128 | 0.9893× |
| 1×4096 | 1.0036× |
| 4×4096 | 1.0033× |
| 8×4096 | **1.0111×** |
| 16×16384 | 1.0051× |
| 32×8192 | **1.0140×** |

The reference remains the default: the small aggregate effect and small-shape regression do not support replacing it generally. A shape-specific dispatcher selected from these results would need prospective confirmation. This does not discard the measured positive effects. These are **one-GPU benchmark results, not an official multi-chip score**.

[All variants and shapes](results/gate-20261028/SUMMARY.md) · [Raw timings](results/gate-20261028/flagos_gate_raw.csv) · [Correctness/results](results/gate-20261028/flagos_gate_results.json)

```bash
python bench/controlled_gate.py       # rerun on a compatible CUDA GPU
python scripts/summarize_gate.py      # CPU-only independent reconstruction of saved results
```

## Three further kernel implementations

These code changes implement distinct hypotheses against the included SGLang references:

| Candidate | Change | Validation status |
|---|---|---|
| `softcap_row_argmax` | Split a vocabulary row across programs and merge with first-index tie handling | Full GPU tests/benchmark queued |
| `rmsnorm_fp32` | Pack narrow rows and select warps from tile size | Full GPU tests/benchmark queued |
| `gemma_fused_add_rmsnorm` | Pack narrow residual-add/norm rows; retain the wide-row plan | Full GPU tests/benchmark queued |

The candidate files, reference implementations, tie/stride/dtype tests and interleaved benchmark are included. GPU job `d1006ec4412e4f09` is queued under the canonical server project `flagos-track1-s2`; no pending result is presented as measured. The acceptance rule written before execution requires correctness and a win on **every tested case in every run**. All regressions will be retained.

```bash
python -m pytest -q tests/test_kernels.py
python bench/bench.py --runs 3 --rounds 5 --out results
```

## Official entry status and dates

No official submission receipt, eight-chip score or completed FlagOS registration has been independently verified for this project. The ZIPs under `submissions/` are prepared candidates, not proof of acceptance. Verify the exact released task statement and evaluator before uploading a candidate.

The28October rules snapshot in the project evidence identifies batch12 development closing **5November19:59:59UTC+8**, batch13 closing **13November19:59:59UTC+8** (12:59:59Madrid), and registration ending13November23:59UTC+8. The later Kaggle container date must not be substituted for these platform batch deadlines. Official registration: [FlagOS portal](https://flagos.io/Register?raceId=8q4m2x7p&lang=en). Team/track exclusivity must be checked before entering both tracks.

## Reproduction and provenance

Requires a compatible PyTorch/Triton CUDA environment. The executed gate result records exact software versions and GPU. Candidate benchmarks use the same input seeds and shapes for reference and optimized code, with correctness checked before timing. Run heavy jobs through the server queue; no GPU on the Windows workstation is needed.

This project extends the author's existing local kernel line and publishes the actual optimization code and evidence. The reference kernels are from [SGLang](https://github.com/sgl-project/sglang), retained under Apache-2.0 with their licence in `reference/LICENSE.sglang`. The gate reference geometry is documented in [SGLang sigmoid-gate implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/moe/triton_sigmoid_gate_mul.py).

Code: Apache-2.0. Author: Francisco Angulo de Lafuente.
