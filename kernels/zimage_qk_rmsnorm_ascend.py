# SPDX-License-Identifier: Apache-2.0
"""Ascend launch repair: bounded grid plus complete grid-stride row coverage.

The official runtime rejected coreDim131072 (>65535). Arithmetic and bf16
rounding match the generic kernel; no rows are dropped when the grid is capped.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _qk_rows_bounded(X, W, Y, M, H: tl.constexpr, S: tl.constexpr,
                     XB: tl.constexpr, XS: tl.constexpr, EPS: tl.constexpr,
                     ROWS: tl.constexpr):
    col = tl.arange(0, 128)
    weight = tl.load(W + col).to(tl.float32)
    for start in range(tl.program_id(0) * ROWS, M, tl.num_programs(0) * ROWS):
        row = start + tl.arange(0, ROWS)
        head = row % H
        token = (row // H) % S
        batch = row // (H * S)
        offsets = batch.to(tl.int64) * XB + token.to(tl.int64) * XS + head.to(tl.int64) * 128
        x = tl.load(X + offsets[:, None] + col[None, :], row[:, None] < M, 0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 1) * (1.0 / 128.0) + EPS)
        rounded = (x * inv[:, None]).to(Y.dtype.element_ty).to(tl.float32)
        tl.store(Y + row.to(tl.int64)[:, None] * 128 + col[None, :],
                 rounded * weight[None, :], row[:, None] < M)


def zimage_qk_rmsnorm(x, weight, eps):
    if (x.ndim != 4 or x.shape[-1] != 128 or x.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16 or weight.shape != (128,)
            or not weight.is_contiguous() or x.stride(-1) != 1 or x.stride(-2) != 128):
        return None
    if x.device != weight.device:
        raise ValueError('Input and weight must share a device')
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    m = x.numel() // 128
    if m:
        grid = min(triton.cdiv(m, 4), 4096)
        _qk_rows_bounded[(grid,)](x, weight, y, m, x.shape[2], x.shape[1],
                                x.stride(0), x.stride(1), float(eps), 4,
                                num_warps=4, enable_fp_fusion=False)
    return y
