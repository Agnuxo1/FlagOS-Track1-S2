# Recorded operator benchmark

RTX 3090; 3 fresh-input runs, 5 interleaved rounds per run.
Ratios are reference time / candidate time. Each row passed correctness.
This is a CUDA proxy, not an official eight-chip evaluation.

| Operator | Case | Speedup | Range across runs | Wins in all runs |
|---|---|---:|---:|---|
| softcap_row_argmax | rows=1,V=32064 | 2.0000× | 1.9231–2.0000 | yes |
| softcap_row_argmax | rows=1,V=128256 | 5.6429× | 5.5714–5.6429 | yes |
| softcap_row_argmax | rows=1,V=262144 | 9.9333× | 9.8667–10.0000 | yes |
| softcap_row_argmax | rows=4,V=32064 | 1.8571× | 1.7857–1.8571 | yes |
| softcap_row_argmax | rows=4,V=128256 | 4.8750× | 4.5882–4.8750 | yes |
| softcap_row_argmax | rows=4,V=262144 | 7.4500× | 7.4000–7.4500 | yes |
| softcap_row_argmax | rows=16,V=32064 | 1.6875× | 1.6250–1.6875 | yes |
| softcap_row_argmax | rows=16,V=128256 | 3.0769× | 3.0769–3.2000 | yes |
| softcap_row_argmax | rows=16,V=262144 | 4.0541× | 4.0541–4.0541 | yes |
| softcap_row_argmax | rows=64,V=32064 | 1.1346× | 1.1154–1.1538 | yes |
| softcap_row_argmax | rows=64,V=128256 | 1.4483× | 1.4483–1.4483 | yes |
| softcap_row_argmax | rows=64,V=262144 | 1.5859× | 1.5859–1.5859 | yes |
| softcap_row_argmax | rows=256,V=32064 | 0.9811× | 0.9811–1.0000 | no |
| softcap_row_argmax | rows=256,V=128256 | 0.9825× | 0.9767–0.9882 | no |
| softcap_row_argmax | rows=256,V=262144 | 0.9674× | 0.9688–0.9732 | no |
| rmsnorm_fp32 | bf16,M=1,D=4096 | 0.8889× | 1.0000–1.0000 | no |
| rmsnorm_fp32 | bf16,M=16,D=4096 | 1.0000× | 0.8889–1.0000 | no |
| rmsnorm_fp32 | bf16,M=256,D=4096 | 0.9286× | 0.9286–1.0000 | no |
| rmsnorm_fp32 | bf16,M=4096,D=4096 | 1.0000× | 0.9884–1.0118 | no |
| rmsnorm_fp32 | bf16,M=64,D=8192 | 1.0000× | 1.0000–1.0000 | no |
| rmsnorm_fp32 | bf16,M=8192,D=128 | 1.3333× | 1.1852–1.3333 | yes |
| rmsnorm_fp32 | bf16,M=32768,D=128 | 1.3929× | 1.3929–1.3929 | yes |
| rmsnorm_fp32 | bf16,M=4096,D=1024 | 1.0370× | 1.0000–1.0370 | no |
| rmsnorm_fp32 | fp32,M=1,D=4096 | 1.0000× | 1.0000–1.1250 | no |
| rmsnorm_fp32 | fp32,M=16,D=4096 | 0.9000× | 0.9000–1.0000 | no |
| rmsnorm_fp32 | fp32,M=256,D=4096 | 1.0588× | 1.0000–1.0588 | no |
| rmsnorm_fp32 | fp32,M=4096,D=4096 | 0.9939× | 0.9939–1.0000 | no |
| rmsnorm_fp32 | fp32,M=64,D=8192 | 1.0769× | 1.0000–1.0769 | no |
| rmsnorm_fp32 | fp32,M=8192,D=128 | 1.1176× | 1.0556–1.1176 | yes |
| rmsnorm_fp32 | fp32,M=32768,D=128 | 1.0000× | 1.0000–1.0217 | no |
| rmsnorm_fp32 | fp32,M=4096,D=1024 | 1.0217× | 0.9787–1.0217 | no |
| gemma_fused_add_rmsnorm | bf16,M=1,N=3072 | 1.0000× | 0.9000–1.0000 | no |
| gemma_fused_add_rmsnorm | bf16,M=32,N=3072 | 0.9091× | 0.9091–1.0000 | no |
| gemma_fused_add_rmsnorm | bf16,M=2048,N=3072 | 0.9706× | 0.9851–0.9853 | no |
| gemma_fused_add_rmsnorm | bf16,M=512,N=6144 | 0.9487× | 0.9487–1.0000 | no |
| gemma_fused_add_rmsnorm | bf16,M=4096,N=128 | 1.0769× | 1.0769–1.0769 | yes |
| gemma_fused_add_rmsnorm | bf16,M=16384,N=128 | 1.0345× | 1.0000–1.0345 | no |
| gemma_fused_add_rmsnorm | bf16,M=16384,N=256 | 1.0435× | 1.0217–1.0435 | yes |
| gemma_fused_add_rmsnorm | bf16,M=65536,N=64 | 1.5745× | 1.5417–1.5745 | yes |

None of the three candidates satisfies the preregistered all-case,
all-run speed gate. The shape-specific gains remain measured results;
the regressions prevent unconditional promotion, not their reporting.
