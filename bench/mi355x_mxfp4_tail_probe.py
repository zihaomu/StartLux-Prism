"""Rejected mixed-main/tail FP4 experiment; compare using the standard micro harness."""
import hashlib
from pathlib import Path
import sys
import triton
import triton.language as tl
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from startlux_prism.kernels import mxfp4 as F
import mi355x_mxfp4 as micro

@triton.jit
def _two_part(A, AS, W, WS, Y, LIVE, N: tl.constexpr, K: tl.constexpr, LD: tl.constexpr,
              BN: tl.constexpr, BK: tl.constexpr, TAIL: tl.constexpr):
    main = tl.arange(0, 128)
    tail = 128 + tl.arange(0, TAIL)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k, s = tl.arange(0, BK // 2), tl.arange(0, BK // 32)
    live = tl.load(LIVE)
    acc = tl.zeros((128, BN), tl.float32)
    acc_tail = tl.zeros((TAIL, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk, ss = block * (BK // 2) + k, block * (BK // 32) + s
        w = tl.load(W + n[:, None] * (K // 2) + kk[None, :],
                    (n[:, None] < N) & (kk[None, :] < K // 2), 0)
        sw = tl.load(WS + n[:, None] * (K // 32) + ss[None, :],
                     (n[:, None] < N) & (ss[None, :] < K // 32), 127)
        x = tl.load(A + main[:, None] * (K // 2) + kk[None, :],
                    (main[:, None] < live) & (kk[None, :] < K // 2), 0)
        sx = tl.load(AS + main[:, None] * (K // 32) + ss[None, :],
                     (main[:, None] < live) & (ss[None, :] < K // 32), 127)
        xt = tl.load(A + tail[:, None] * (K // 2) + kk[None, :],
                     (tail[:, None] < live) & (kk[None, :] < K // 2), 0)
        st = tl.load(AS + tail[:, None] * (K // 32) + ss[None, :],
                     (tail[:, None] < live) & (ss[None, :] < K // 32), 127)
        acc = tl.dot_scaled(x, sx, 'e2m1', tl.trans(w), sw, 'e2m1', acc)
        acc_tail = tl.dot_scaled(xt, st, 'e2m1', tl.trans(w), sw, 'e2m1', acc_tail)
    tl.store(Y + main[:, None] * LD + n[None, :], acc, (main[:, None] < live) & (n[None, :] < N))
    tl.store(Y + tail[:, None] * LD + n[None, :], acc_tail, (tail[:, None] < live) & (n[None, :] < N))


@triton.jit
def _gemm_tail(A, AS, W, WS, Y, LIVE, N: tl.constexpr, K: tl.constexpr, LD: tl.constexpr,
               BN: tl.constexpr, BK: tl.constexpr):
    if tl.load(LIVE) > 144:
        _two_part(A, AS, W, WS, Y, LIVE, N, K, LD, BN, BK, 64)
    else:
        _two_part(A, AS, W, WS, Y, LIVE, N, K, LD, BN, BK, 16)


def candidate(a, w, out, live, cfg=None):
    if a.shape[0] != 192:
        raise ValueError('This experiment covers only the 192-row bucket')
    _, bn, bk, warps, stages = cfg[:5]
    n, k = w.shape
    return _gemm_tail[(triton.cdiv(n, bn),)](a.q, a.s, w.q, w.s, out, live, n, k, out.stride(0), bn, bk,
                                            num_warps=warps, num_stages=stages, matrix_instr_nonkdim=16)

if __name__ == '__main__':
    F.gemm = candidate
    F.kernel_identity = lambda: hashlib.sha256((_gemm_tail.src + _two_part.src).encode()).hexdigest()
    micro.main()
