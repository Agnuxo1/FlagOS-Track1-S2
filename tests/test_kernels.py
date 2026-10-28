# SPDX-License-Identifier: Apache-2.0
"""Correctness: optimized kernels vs. the SGLang reference kernels and vs. PyTorch math."""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kernels import gemma_fused_add_rmsnorm as opt_gemma  # noqa: E402
from kernels import rmsnorm_fp32 as opt_rms  # noqa: E402
from kernels import softcap_row_argmax as opt_soft  # noqa: E402
from reference import gemma_fused_add_rmsnorm as ref_gemma  # noqa: E402
from reference import rmsnorm_fp32 as ref_rms  # noqa: E402
from reference import softcap_row_argmax as ref_soft  # noqa: E402

DEV = "cuda"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------- softcap_row_argmax
@pytest.mark.parametrize("rows", [1, 3, 16, 64, 257])
@pytest.mark.parametrize("width,valid", [(1000, 1000), (32064, 32000), (128256, 128256), (262144, 262000)])
@pytest.mark.parametrize("col_offset", [0, 131072])
def test_softcap_row_argmax_matches_reference(rows, width, valid, col_offset):
    torch.manual_seed(rows * 7 + width)
    logits = torch.randn(rows, width, device=DEV, dtype=torch.float32) * 20
    ref = ref_soft.softcap_row_argmax(logits, 30.0, valid, col_offset)
    got = opt_soft.softcap_row_argmax(logits, 30.0, valid, col_offset)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_softcap_row_argmax_ties_pick_first():
    # Saturated tanh produces many equal softcapped values: first index must win.
    rows, width = 4, 65536
    logits = torch.full((rows, width), -5.0, device=DEV)
    for r, pos in enumerate([5, 40000, 65535, 20000]):
        logits[r, pos:] = 1e4  # all saturate to +cap
    ref = ref_soft.softcap_row_argmax(logits, 15.0, width, 0)
    got = opt_soft.softcap_row_argmax(logits, 15.0, width, 0)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    assert got[:, 1].tolist() == [5.0, 40000.0, 65535.0, 20000.0]


def test_softcap_row_argmax_strided_rows():
    big = torch.randn(8, 70000, device=DEV) * 10
    view = big[:, :50000]
    ref = ref_soft.softcap_row_argmax(view, 30.0, 50000, 7)
    got = opt_soft.softcap_row_argmax(view, 30.0, 50000, 7)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


# ---------------------------------------------------------------- rmsnorm_fp32
def _rms_torch(x, w, eps):
    xf = x.float()
    return (w.float() * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps))).to(x.dtype)


@pytest.mark.parametrize("shape", [(1, 4096), (7, 4096), (8192, 128), (333, 96), (64, 8192), (2, 3, 1024), (5, 16384)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("wdtype", [torch.bfloat16, torch.float32])
def test_rmsnorm_fp32(shape, dtype, wdtype):
    torch.manual_seed(0)
    x = torch.randn(shape, device=DEV, dtype=dtype)
    w = torch.randn(shape[-1], device=DEV, dtype=wdtype)
    ref = ref_rms.rmsnorm_fp32(x, w, 1e-6)
    got = opt_rms.rmsnorm_fp32(x, w, 1e-6)
    tol = dict(rtol=1.6e-2, atol=1e-2) if dtype == torch.bfloat16 else dict(rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got, ref, **tol)
    torch.testing.assert_close(got, _rms_torch(x, w, 1e-6), **tol)


# ---------------------------------------------------------------- gemma_fused_add_rmsnorm
def _gemma_torch(x, r, w, eps):
    s = x.float() + r.float()
    out = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())
    return out.to(x.dtype), s.to(x.dtype)


@pytest.mark.parametrize("shape", [(1, 3072), (37, 3072), (4096, 128), (1000, 256), (3, 8, 64), (16, 6144), (2, 12288)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gemma_fused_add_rmsnorm(shape, dtype):
    torch.manual_seed(1)
    x = torch.randn(shape, device=DEV, dtype=dtype)
    r = torch.randn(shape, device=DEV, dtype=dtype)
    w = torch.randn(shape[-1], device=DEV, dtype=torch.float32) * 0.1
    ro, rs = ref_gemma.gemma_fused_add_rmsnorm(x, r, w, 1e-6)
    go, gs = opt_gemma.gemma_fused_add_rmsnorm(x, r, w, 1e-6)
    tol = dict(rtol=1e-5, atol=1e-5) if dtype == torch.float32 else dict(rtol=1.6e-2, atol=2e-2)
    torch.testing.assert_close(gs, rs, rtol=0, atol=0)  # the sum is computed identically
    torch.testing.assert_close(go, ro, **tol)
    to, ts = _gemma_torch(x, r, w, 1e-6)
    torch.testing.assert_close(go, to, **tol)
    torch.testing.assert_close(gs, ts, **tol)


def test_gemma_fused_add_rmsnorm_strided_views():
    # Per-head q/k norm on non-contiguous qkv.split views.
    torch.manual_seed(2)
    qkv = torch.randn(512, 3 * 1024, device=DEV, dtype=torch.bfloat16)
    q = qkv[:, :1024].reshape(512, 8, 128)
    res = torch.randn(512, 8, 128, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(128, device=DEV) * 0.1
    ro, rs = ref_gemma.gemma_fused_add_rmsnorm(q, res, w, 1e-6)
    go, gs = opt_gemma.gemma_fused_add_rmsnorm(q, res, w, 1e-6)
    torch.testing.assert_close(gs, rs, rtol=0, atol=0)
    torch.testing.assert_close(go, ro, rtol=1.6e-2, atol=2e-2)
