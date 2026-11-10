# Copyright 2026 Francisco Angulo de Lafuente
# SPDX-License-Identifier: Apache-2.0
"""Task196: fused bias plus tanh GELU with specified low precision rounding.

The sigmoid identity follows the public FlagGems-sglang GELU implementation;
this independent bias/rounding kernel uses no backend-specific math imports.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _flat(X,B,Y,N,H:tl.constexpr,BLOCK:tl.constexpr):
    for start in range(tl.program_id(0)*BLOCK,N,tl.num_programs(0)*BLOCK):
        i=start+tl.arange(0,BLOCK)
        x=tl.load(X+i,i<N,0).to(tl.float32)
        b=tl.load(B+i%H,i<N,0).to(tl.float32)
        z=(x+b).to(X.dtype.element_ty).to(tl.float32)
        q=1.5957691216057308*(z+0.044715*(z*z*z))
        y=z*tl.fdiv(1.0,1.0+tl.exp(-q))
        tl.store(Y+i,y.to(Y.dtype.element_ty),i<N)


@triton.jit
def _row(X,B,Y,N,H:tl.constexpr,BLOCK:tl.constexpr,TILES:tl.constexpr):
    rows=N//H
    for p in range(tl.program_id(0),rows*TILES,tl.num_programs(0)):
        row=p//TILES
        col=(p%TILES)*BLOCK+tl.arange(0,BLOCK)
        i=row*H+col
        x=tl.load(X+i,col<H,0).to(tl.float32)
        b=tl.load(B+col,col<H,0).to(tl.float32)
        z=(x+b).to(X.dtype.element_ty).to(tl.float32)
        q=1.5957691216057308*(z+0.044715*(z*z*z))
        y=z*tl.fdiv(1.0,1.0+tl.exp(-q))
        tl.store(Y+i,y.to(Y.dtype.element_ty),col<H)


def _validate(x,bias):
    if x.ndim<1 or x.numel()==0 or x.shape[-1]%8:
        raise ValueError('Expected nonempty [...,H] with H divisible by8')
    if (x.dtype not in (torch.float16,torch.bfloat16) or bias.dtype!=x.dtype
            or bias.ndim!=1 or bias.shape[0]!=x.shape[-1]
            or not x.is_contiguous() or not bias.is_contiguous() or bias.device!=x.device):
        raise ValueError('Inputs must satisfy the published shape/dtype/device/contiguity contract')


def run_variant(x,bias,variant):
    _validate(x,bias)
    y=torch.empty_like(x)
    n=x.numel();h=x.shape[-1]
    if variant=='B0_row':
        block=min(triton.next_power_of_2(h),1024)
        tiles=triton.cdiv(h,block)
        grid=min((n//h)*tiles,4096)
        _row[(grid,)](x,bias,y,n,h,block,tiles,num_warps=4,enable_fp_fusion=False)
    else:
        block={'F256':256,'F512':512,'F1024':1024}[variant]
        _flat[(min(triton.cdiv(n,block),4096),)](x,bias,y,n,h,block,num_warps=4,enable_fp_fusion=False)
    return y


def bias_gelu_tanh(x,bias):
    return run_variant(x,bias,'F1024')


reference=bias_gelu_tanh

# Round2: direct traversal and packed-row bias reuse, same arithmetic.
@triton.jit
def _direct(X,B,Y,N,H:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+i,i<N,0).to(tl.float32)
    b=tl.load(B+i%H,i<N,0).to(tl.float32)
    z=(x+b).to(X.dtype.element_ty).to(tl.float32)
    q=1.5957691216057308*(z+0.044715*(z*z*z))
    y=z*tl.fdiv(1.0,1.0+tl.exp(-q))
    tl.store(Y+i,y.to(Y.dtype.element_ty),i<N)

@triton.jit
def _packed(X,B,Y,R:tl.constexpr,H:tl.constexpr,BR:tl.constexpr,BH:tl.constexpr):
    rows=tl.program_id(0)*BR+tl.arange(0,BR)
    cols=tl.program_id(1)*BH+tl.arange(0,BH)
    offsets=rows[:,None]*H+cols[None,:]
    mask=(rows[:,None]<R)&(cols[None,:]<H)
    x=tl.load(X+offsets,mask,0).to(tl.float32)
    b=tl.load(B+cols,cols<H,0).to(tl.float32)
    z=(x+b[None,:]).to(X.dtype.element_ty).to(tl.float32)
    q=1.5957691216057308*(z+0.044715*(z*z*z))
    y=z*tl.fdiv(1.0,1.0+tl.exp(-q))
    tl.store(Y+offsets,y.to(Y.dtype.element_ty),mask)

def run_round2_variant(x,bias,variant):
    _validate(x,bias)
    y=torch.empty_like(x)
    n=x.numel();h=x.shape[-1]
    if variant in ('D1024','D4096'):
        block={'D1024':1024,'D4096':4096}[variant]
        _direct[(triton.cdiv(n,block),)](x,bias,y,n,h,block,num_warps=4,enable_fp_fusion=False)
    else:
        br,bh={'P4x256':(4,256),'P4x1024':(4,1024),'P8x256':(8,256)}[variant]
        _packed[(triton.cdiv(n//h,br),triton.cdiv(h,bh))](x,bias,y,n//h,h,br,bh,num_warps=4,enable_fp_fusion=False)
    return y

def round2_entry(x,bias,variant):
    return run_round2_variant(x,bias,variant)
