# FlagOS Track1 S2 — controlled Triton kernels

Own Triton optimization candidates for [FlagOS Open Computing Global Challenge, Track1](https://www.kaggle.com/competitions/flag-os-open-computing-global-challenge-track-1), with reproducible correctness gates and raw latency evidence.

## Live Batch 11 candidate: strided ZImage QK RMSNorm

The newly registered team now targets **task177 `zimage_qk_rmsnorm`**. This
adapts our packed-row RMSNorm to the official bf16/128-column interface,
reads fused-QKV slices without a contiguous input copy, and preserves the
required bf16 rounding before weight multiplication. Output is fresh and
contiguous. All core arithmetic runs in Triton, with no Torch math fallback.
RMSNorm's mathematical background is [Zhang & Sennrich, NeurIPS2019](https://arxiv.org/abs/1910.07467);
that paper is not evidence for our measured packing speedups.

Three interventions were fixed before measuring. Each used six shapes,
two layouts, three fresh seeds and seven interleaved timing rounds. The
control is **our same-formula one-row Triton kernel**, not SGLang's production
baseline. Inputs are generated benchmark tensors; timing and correctness
measurements come from the actual RTX3090.

| Candidate | Exploratory geometric ratio | Fresh-seed confirmation ratio | Confirmation wins / ties / losses |
|---|---:|---:|---:|
| Z1:4rows,4warps | 1.130623× | **1.139398×** | 23 /13 /0 |
| Z2:8rows,4warps | **1.141692×** | 1.149730× | 24 /11 /1 |
| Z3:16rows,8warps | 1.140406× | 1.147811× | 24 /11 /1 |

Z1 was selected **before** confirmation (seeds10/11/12), following the JEV
review, and remains the exported default. Faster Z2/Z3 aggregate results and
their regressions remain visible. All **288 numerical checks passed** across
the two phases, with maximum absolute difference0.03125 against the official
Torch formula, within bf16 `rtol=.016,atol=1e-5`. These are tolerance checks,
not a bitwise equality claim. Microsecond event timings are quantised; ties
and this fixed shape panel limit interpretation. These ratios are not an
official multi-chip score.

[Full JSON and all2016 timing rows](results/task177-20261029) ·
[Frozen exploratory protocol](evidence/task177-protocol-20261029.json)

```sh
python scripts/verify_task177.py  # independently reconstruct every recorded ratio on CPU
python bench/task177.py         # fresh GPU benchmark through the managed queue
```

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
| `softcap_row_argmax` | Split a vocabulary row across programs and merge with first-index tie handling | Measured: 2.3577× geometric mean; wins every run on 12/15 cases |
| `rmsnorm_fp32` | Pack narrow rows and select warps from tile size | Measured: 1.0393× geometric mean; wins every run on 3/16 cases |
| `gemma_fused_add_rmsnorm` | Pack narrow residual-add/norm rows; retain the wide-row plan | Measured: 1.0548× geometric mean; wins every run on 3/8 cases |

GPU job `d1006ec4412e4f09` completed successfully on an RTX 3090. All 39 cases
passed correctness in each of three fresh-input runs. The raw file contains
585 paired timing rounds. The strongest measured per-case ratios are **9.9333×**
for softcap, **1.3929×** for RMSNorm and **1.5745×** for fused add/RMSNorm.

The preregistered gate requires a speed win on **every tested case in every run**.
None meets that unconditional gate: worst-case ratios are respectively 0.9674×,
0.8889× and 0.9091×. The gains remain measured positive results; these regressions
prevent unconditional promotion. [All cases and run ranges](results/operators-20261028/SUMMARY.md)
are retained alongside raw timings. This is one CUDA GPU, not the official
multi-chip score. Microsecond timings and this fixed shape grid limit generalisation.

```bash
python -m pytest -q tests/test_kernels.py
python bench/bench.py --runs 3 --rounds 5 --out results
python scripts/summarize_operators.py  # reconstruct all 39 cases on CPU
```

## Official entry status and dates

**Registration completed, 29 October 2026:** the official portal confirmed
“Registration Successful!” for **Agnuxo1Lab, Track 1**, with the author as captain.
The consent and contact details remain in the private operational record.
The first official code entry, `task177-z1.zip`, was submitted on29October
at00:20UTC. The task history shows **Evaluating**, one entry and14/15
remaining daily attempts. No evaluated speedup or chip pass count is claimed
while pending. [Recorded receipt](evidence/task177-official-receipt-20261029.json).
The other ZIPs under `submissions/` remain prepared candidates.

The current submission selector exposes **Batch 11, tasks161–177**. Our older
softcap/RMSNorm candidates cannot be uploaded under a different task name.
The exact released task contract must match before an official code upload.
Batch11 closes **29October19:59:59UTC+8 (12:59:59Madrid)**. The portal's
countdown conflicts with the stated UTC+8 schedule; use the earlier explicit
converted deadline until the organiser resolves it.

The28October rules snapshot in the project evidence identifies batch12 development closing **5November19:59:59UTC+8**, batch13 closing **13November19:59:59UTC+8** (12:59:59Madrid), and registration ending13November23:59UTC+8. The later Kaggle container date must not be substituted for these platform batch deadlines. Official registration: [FlagOS portal](https://flagos.io/Register?raceId=8q4m2x7p&lang=en). Team/track exclusivity must be checked before entering both tracks.

## Reproduction and provenance

Requires a compatible PyTorch/Triton CUDA environment. The executed gate result records exact software versions and GPU. Candidate benchmarks use the same input seeds and shapes for reference and optimized code, with correctness checked before timing. Run heavy jobs through the server queue; no GPU on the Windows workstation is needed.

This project extends the author's existing local kernel line and publishes the actual optimization code and evidence. The reference kernels are from [SGLang](https://github.com/sgl-project/sglang), retained under Apache-2.0 with their licence in `reference/LICENSE.sglang`. The gate reference geometry is documented in [SGLang sigmoid-gate implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/moe/triton_sigmoid_gate_mul.py).

Code: Apache-2.0. Author: Francisco Angulo de Lafuente.
