# SPDX-License-Identifier: Apache-2.0
"""Fused residual add + Gemma RMSNorm, ``normalize(x + r) * (1 + w)``.

Drop-in replacement for SGLang's ``gemma_fused_add_rmsnorm`` (same signature,
returns ``(normed, pre-norm sum)``, accepts strided ``qkv.split`` views).

The reference launches one row per program. For the per-head q/k norms this
kernel serves (n = 64..256, thousands of rows) each program touches only a
few hundred bytes, so launch and scheduling overhead dominates. This version
packs ``ROWS`` rows into one 2-D tile and picks ``num_warps`` from the tile
size; wide rows keep one row per program as in the reference.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma_fused_add_rmsnorm_rows_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    res_out_ptr,
    n_rows,
    n_cols,
    stride_xrow,
    stride_xcol,
    stride_rrow,
    stride_rcol,
    eps,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = (pid * ROWS + tl.arange(0, ROWS)).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < n_cols
    mask = (rows < n_rows)[:, None] & col_mask[None, :]
    x = tl.load(
        x_ptr + rows[:, None] * stride_xrow + cols[None, :] * stride_xcol,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    r = tl.load(
        res_ptr + rows[:, None] * stride_rrow + cols[None, :] * stride_rcol,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    s = x + r
    out_offs = rows[:, None] * n_cols + cols[None, :]
    # residual_out is the pre-norm sum (consumed by the next layer's add).
    tl.store(res_out_ptr + out_offs, s.to(res_out_ptr.dtype.element_ty), mask=mask)
    var = tl.sum(s * s, axis=1) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    out = s * rstd[:, None] * (1.0 + w)[None, :]
    tl.store(out_ptr + out_offs, out.to(out_ptr.dtype.element_ty), mask=mask)


def _plan(n: int, m: int, elem_bytes: int):
    block_n = triton.next_power_of_2(n)
    rows = max(1, min(8192 // (block_n * elem_bytes), 32))
    rows = min(rows, triton.next_power_of_2(max(m, 1)))
    tile = block_n * rows
    if tile >= 4096:
        num_warps = 16 if block_n >= 4096 else 8
    elif tile >= 1024:
        num_warps = 4
    else:
        num_warps = 2
    return block_n, rows, num_warps


def gemma_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
):
    """Fused (x + residual) then Gemma RMSNorm; returns (normed, pre-norm sum)."""
    orig_shape = x.shape
    n = orig_shape[-1]
    x2 = x.reshape(-1, n)
    r2 = residual.reshape(-1, n)
    m = x2.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    res_out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m:
        block_n, rows, num_warps = _plan(n, m, x.element_size())
        _gemma_fused_add_rmsnorm_rows_kernel[(triton.cdiv(m, rows),)](
            x2,
            r2,
            weight,
            out,
            res_out,
            m,
            n,
            x2.stride(0),
            x2.stride(1),
            r2.stride(0),
            r2.stride(1),
            eps,
            BLOCK_N=block_n,
            ROWS=rows,
            num_warps=num_warps,
        )
    return out.reshape(orig_shape), res_out.reshape(orig_shape)
