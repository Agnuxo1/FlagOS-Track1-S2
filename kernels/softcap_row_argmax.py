# SPDX-License-Identifier: Apache-2.0
"""Split-vocab greedy argmax over softcapped logits.

Drop-in replacement for SGLang's ``softcap_row_argmax`` (same signature, same
output layout, same first-occurrence tie-break).

The reference launches one program per row and walks the whole vocab shard
serially. Decode batches are small (1-64 rows), so most SMs sit idle. This
version splits each row into ``n_split`` contiguous column chunks, reduces
each chunk in its own program, then merges the partials with a tiny second
kernel. Element-wise math is identical to the reference (``x / cap``,
``tanh``, ``* cap``), so values and argmax indices match bit for bit.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _softcap_argmax_partial_kernel(
    logits_ptr,
    part_val_ptr,
    part_idx_ptr,
    out_ptr,
    row_stride,
    valid_width,
    col_offset,
    softcapping_value,
    chunk,
    n_split,
    BLOCK_SIZE: tl.constexpr,
    WRITE_FINAL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    begin = split * chunk
    end = tl.minimum(begin + chunk, valid_width)
    best_val = -float("inf")
    best_idx = begin
    for start in range(begin, end, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < end
        x = tl.load(logits_ptr + row * row_stride + cols, mask=mask, other=0.0)
        x = x / softcapping_value
        x = libdevice.tanh(x)
        x = x * softcapping_value
        x = tl.where(mask, x, -float("inf"))
        block_val, block_idx = tl.max(
            x, axis=0, return_indices=True, return_indices_tie_break_left=True
        )
        take = block_val > best_val
        best_idx = tl.where(take, start + block_idx, best_idx)
        best_val = tl.where(take, block_val, best_val)
    if WRITE_FINAL:
        tl.store(out_ptr + row * 2, best_val)
        tl.store(out_ptr + row * 2 + 1, (best_idx + col_offset).to(tl.float32))
    else:
        tl.store(part_val_ptr + row * n_split + split, best_val)
        tl.store(part_idx_ptr + row * n_split + split, best_idx)


@triton.jit
def _softcap_argmax_merge_kernel(
    part_val_ptr,
    part_idx_ptr,
    out_ptr,
    col_offset,
    n_split,
    BLOCK_S: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_S)
    mask = offs < n_split
    vals = tl.load(part_val_ptr + row * n_split + offs, mask=mask, other=-float("inf"))
    # Chunks are ordered by column, so the leftmost maximal chunk holds the
    # first global occurrence of the maximum.
    best_val, best_split = tl.max(
        vals, axis=0, return_indices=True, return_indices_tie_break_left=True
    )
    best_idx = tl.load(part_idx_ptr + row * n_split + best_split)
    tl.store(out_ptr + row * 2, best_val)
    tl.store(out_ptr + row * 2 + 1, (best_idx + col_offset).to(tl.float32))


_SM_CACHE = {}


def _sm_count(device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SM_CACHE:
        _SM_CACHE[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SM_CACHE[idx]


def _plan(rows: int, valid_width: int, device):
    """Pick (block, chunk, n_split, num_warps) so that rows * n_split ~ 2x SMs."""
    block = 2048 if valid_width >= 2048 else max(triton.next_power_of_2(valid_width), 16)
    target_programs = 2 * _sm_count(device)
    max_split = triton.cdiv(valid_width, block)
    n_split = max(1, min(max_split, triton.cdiv(target_programs, max(rows, 1)), 128))
    chunk = triton.cdiv(triton.cdiv(valid_width, n_split), block) * block
    n_split = triton.cdiv(valid_width, chunk)
    num_warps = 8 if block >= 2048 else 4
    return block, chunk, n_split, num_warps


def softcap_row_argmax(logits, final_logit_softcapping, valid_width, col_offset):
    """Per row of a vocab shard: ``(max, first argmax + col_offset)`` of the
    softcapped logits over the first ``valid_width`` columns, as fp32 ``[rows, 2]``."""
    assert logits.dtype == torch.float32 and logits.stride(1) == 1
    assert col_offset + logits.shape[1] < 2**24
    rows = logits.shape[0]
    out = torch.empty(rows, 2, dtype=torch.float32, device=logits.device)
    if rows == 0:
        return out
    if valid_width <= 0:
        out[:, 0] = -float("inf")
        out[:, 1] = float(col_offset)
        return out
    block, chunk, n_split, num_warps = _plan(rows, valid_width, logits.device)
    if n_split == 1:
        _softcap_argmax_partial_kernel[(rows, 1)](
            logits, out, out, out,
            logits.stride(0), valid_width, col_offset, final_logit_softcapping,
            chunk, 1,
            BLOCK_SIZE=block, WRITE_FINAL=True, num_warps=num_warps,
        )
        return out
    part_val = torch.empty(rows, n_split, dtype=torch.float32, device=logits.device)
    part_idx = torch.empty(rows, n_split, dtype=torch.int32, device=logits.device)
    _softcap_argmax_partial_kernel[(rows, n_split)](
        logits, part_val, part_idx, out,
        logits.stride(0), valid_width, col_offset, final_logit_softcapping,
        chunk, n_split,
        BLOCK_SIZE=block, WRITE_FINAL=False, num_warps=num_warps,
    )
    _softcap_argmax_merge_kernel[(rows,)](
        part_val, part_idx, out, col_offset, n_split,
        BLOCK_S=max(triton.next_power_of_2(n_split), 16), num_warps=1,
    )
    return out
