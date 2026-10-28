# SPDX-License-Identifier: Apache-2.0
"""Out-of-place RMSNorm with FP32 statistics and weight multiplication.

Drop-in replacement for SGLang's ``rmsnorm_fp32`` (same signature, same math,
``enable_fp_fusion=False`` kept so rounding follows the reference).

The reference always launches one row per program with 4 warps. That is too
few threads for wide rows (D >= 4096) and far too many programs, each doing
very little work, for narrow rows (per-head norms, D = 64..256). This version
packs ``ROWS`` rows into one 2-D tile and sizes ``num_warps`` from the tile, so
each program moves a few KB to a few tens of KB of data.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fp32_rows_kernel(
    X, W, Y, n_rows,
    D: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr, ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, B)
    row_mask = rows < n_rows
    col_mask = col < D
    mask = row_mask[:, None] & col_mask[None, :]
    offs = rows.to(tl.int64)[:, None] * D + col[None, :]
    x = tl.load(X + offs, mask, 0).to(tl.float32)
    weight = tl.load(W + col, col_mask, 0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / D + EPS)
    normalized = x * inv_rms[:, None]
    tl.store(Y + offs, normalized * weight[None, :], mask)


def _plan(dim: int, rows: int, elem_bytes: int):
    block = triton.next_power_of_2(dim)
    # Aim for ~16 KB of input per program on narrow rows.
    rows_per_prog = max(1, min(16384 // (block * elem_bytes), 64))
    rows_per_prog = min(rows_per_prog, triton.next_power_of_2(max(rows, 1)))
    tile = block * rows_per_prog
    if tile >= 8192:
        num_warps = 16
    elif tile >= 2048:
        num_warps = 8
    else:
        num_warps = 4
    return block, rows_per_prog, num_warps


def rmsnorm_fp32(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Match ``weight * (x.float() * rsqrt(mean(x.float()**2) + eps))``."""
    assert x.is_cuda and x.is_contiguous() and weight.is_contiguous()
    assert x.dtype in (torch.bfloat16, torch.float32)
    assert weight.dtype in (torch.bfloat16, torch.float32)
    assert x.device == weight.device and weight.shape == (x.shape[-1],)
    output = torch.empty_like(x)
    dim = x.shape[-1]
    rows = x.numel() // dim
    if rows:
        block, rows_per_prog, num_warps = _plan(dim, rows, x.element_size())
        _rmsnorm_fp32_rows_kernel[(triton.cdiv(rows, rows_per_prog),)](
            x,
            weight,
            output,
            rows,
            dim,
            eps,
            block,
            rows_per_prog,
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
    return output
