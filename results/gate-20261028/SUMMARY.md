# Three tiling interventions — all measured shapes

GPU: NVIDIA GeForce RTX 3090; three sessions,24 interleaved rounds/session,100 calls/sample.

| N | D | tile512 /4warps | tile1024 /4warps | tile2048 /8warps |
|---:|---:|---:|---:|---:|
| 1 | 128 | 0.989294× | 0.986952× | 0.991504× |
| 1 | 4096 | 1.003615× | 1.006492× | 1.003052× |
| 4 | 4096 | 1.003254× | 1.002657× | 1.002840× |
| 8 | 4096 | 1.011108× | 0.998384× | 1.003535× |
| 32 | 8192 | 1.014035× | 0.999323× | 0.999772× |
| 16 | 16384 | 1.005082× | 1.004986× | 1.005990× |

Ratios are baseline median / candidate median, averaged over sessions.
All72 correctness checks passed. Raw timings independently reproduce all stored ratios.
One RTX3090; these are not official multi-chip results. Small aggregate gains do not establish a general replacement.
