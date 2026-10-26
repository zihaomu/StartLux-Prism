"""Experimental gfx950 native MXFP4 projections, separate from GGUF k-quants.

E2M1 payloads pack adjacent K values low nibble first. E8M0 scales cover 32
values along K. The initial recipe chooses the smallest power-of-two scale
that contains the block maximum in [-6, 6], then rounds ties to even.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class Packed:
    q: torch.Tensor                 # uint8 [rows, K / 2]
    s: torch.Tensor                 # uint8 [rows, K / 32], biased exponent

    @property
    def shape(self):
        return self.q.shape[0], self.q.shape[1] * 2

    def rows(self, start, end):
        return Packed(self.q[start:end], self.s[start:end])

    def __getitem__(self, rows):
        if not isinstance(rows, slice) or rows.step not in (None, 1):
            raise ValueError('MXFP4 weights only support contiguous row slices')
        return self.rows(rows.start, rows.stop)

    def nbytes(self):
        return self.q.numel() + self.s.numel()


def empty(rows, k, device):
    if k % 32:
        raise ValueError('MXFP4 K must be a multiple of 32')
    return Packed(torch.empty((rows, k // 2), dtype=torch.uint8, device=device),
                  torch.empty((rows, k // 32), dtype=torch.uint8, device=device))


@triton.jit
def _encode(x):
    a = tl.abs(x)
    c = ((a > .25).to(tl.int32) + (a >= .75).to(tl.int32)
         + (a > 1.25).to(tl.int32) + (a >= 1.75).to(tl.int32)
         + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32)
         + (a > 5.).to(tl.int32))
    return (c | ((x.to(tl.int32, bitcast=True) >> 28) & 8)).to(tl.uint8)


@triton.jit
def _quant(X, Q, S, LIVE, ROWS: tl.constexpr, K: tl.constexpr,
           SX: tl.constexpr, GROUPS: tl.constexpr):
    g = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    r, b = g // (K // 32), g % (K // 32)
    j = tl.arange(0, 16)
    active = (r < ROWS) & (r < tl.load(LIVE))
    off = r[:, None] * SX + b[:, None] * 32 + 2 * j[None, :]
    x0 = tl.load(X + off, active[:, None], 0).to(tl.float32)
    x1 = tl.load(X + off + 1, active[:, None], 0).to(tl.float32)
    nonfinite = tl.sum(((tl.abs(x0) == float('inf')) | (x0 != x0)
                       | (tl.abs(x1) == float('inf')) | (x1 != x1)).to(tl.int32), 1) > 0
    amax = tl.max(tl.maximum(tl.abs(x0), tl.abs(x1)), 1)
    e = tl.ceil(tl.log2(tl.maximum(amax / 6., 2.**-126)))
    e = tl.where(amax == 0, 0., tl.minimum(tl.maximum(e, -126.), 127.))
    inv = tl.exp2(-e)
    q = _encode(x0 * inv[:, None]) | (_encode(x1 * inv[:, None]) << 4)
    q = tl.where(nonfinite[:, None], 0, q)
    s = tl.where(nonfinite, 255, e.to(tl.int32) + 127).to(tl.uint8)
    tl.store(Q + g[:, None] * 16 + j[None, :], q, r[:, None] < ROWS)
    tl.store(S + g, s, r < ROWS)


def quantize_into(x, packed, live):
    if x.ndim != 2 or x.stride(1) != 1 or tuple(x.shape) != packed.shape:
        raise ValueError('MXFP4 input must have matching [rows, K] shape and contiguous K')
    if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError('MXFP4 quantization requires floating-point input')
    _quant[(triton.cdiv(x.numel() // 32, 16),)](
        x, packed.q, packed.s, live, x.shape[0], x.shape[1], x.stride(0), 16)
    return packed


def quantize(x):
    out = empty(*x.shape, x.device)
    live = torch.tensor(x.shape[0], dtype=torch.int32, device=x.device)
    return quantize_into(x, out, live)


@triton.jit
def _gemm(A, AS, W, WS, Y, LIVE, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          LD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    live = tl.load(LIVE)
    if tl.program_id(0) * BM >= live:
        return
    k = tl.arange(0, BK // 2)
    sk = tl.arange(0, BK // 32)
    acc = tl.zeros((BM, BN), tl.float32)
    for b in range(tl.cdiv(K, BK)):
        kk, ss = b * (BK // 2) + k, b * (BK // 32) + sk
        a = tl.load(A + r[:, None] * (K // 2) + kk[None, :],
                    (r[:, None] < live) & (kk[None, :] < K // 2), 0)
        w = tl.load(W + n[:, None] * (K // 2) + kk[None, :],
                    (n[:, None] < N) & (kk[None, :] < K // 2), 0)
        sa = tl.load(AS + r[:, None] * (K // 32) + ss[None, :],
                     (r[:, None] < live) & (ss[None, :] < K // 32), 127)
        sw = tl.load(WS + n[:, None] * (K // 32) + ss[None, :],
                     (n[:, None] < N) & (ss[None, :] < K // 32), 127)
        acc = tl.dot_scaled(a, sa, 'e2m1', tl.trans(w), sw, 'e2m1', acc)
    tl.store(Y + r[:, None] * LD + n[None, :], acc,
             (r[:, None] < live) & (n[None, :] < N))


def gemm(a, w, out, live, cfg=(32, 128, 256, 4, 2)):
    m, k = a.shape
    n, wk = w.shape
    if wk != k or out.shape != (m, n) or out.stride(1) != 1:
        raise ValueError('MXFP4 product/output dimensions or output strides do not match')
    if not torch.version.hip or not torch.cuda.get_device_properties(out.device).gcnArchName.startswith('gfx950'):
        raise ValueError('Native MXFP4 requires ROCm gfx950')
    bm, bn, bk, warps, stages = cfg
    return _gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        a.q, a.s, w.q, w.s, out, live, m, n, k, out.stride(0), bm, bn, bk,
        num_warps=warps, num_stages=stages, matrix_instr_nonkdim=16)
