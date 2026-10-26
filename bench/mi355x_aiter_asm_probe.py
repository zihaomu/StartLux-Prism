"""Optional AITER ASM probe with offline B packing and runtime A-scale packing."""
import argparse
import json
import math
from pathlib import Path
import sys

bench_dir = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != bench_dir]
sys.path.insert(0, str(bench_dir.parent))
import torch
import aiter
from aiter.ops.shuffle import shuffle_weight, shuffle_scale
from startlux_prism.kernels import mxfp4 as F
sys.path.insert(0, str(bench_dir))
from gemm_micro import timed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    rows = []
    path = Path(a.out)
    torch.manual_seed(712)
    for n, k in [(34816, 5120), (5120, 17408)]:
        x = torch.randn((192, k), device='cuda', dtype=torch.bfloat16)
        aq = F.empty(192, k, 'cuda')
        live = torch.tensor(144, device='cuda', dtype=torch.int32)
        F.quantize_into(x, aq, live)
        copies = max(4, math.ceil(512 * 1024**2 / (n * k * 17 / 32)))
        weights = [F.quantize(torch.randn((n, k), device='cuda', dtype=torch.bfloat16)) for _ in range(copies)]
        ws = [shuffle_scale(w.s).view(torch.float8_e8m0fnu) for w in weights]
        wq = [shuffle_weight(w.q, layout=(16, 16)).view(torch.float4_e2m1fn_x2) for w in weights]
        y = torch.empty((192, n), device='cuda', dtype=torch.bfloat16)
        ref = torch.empty_like(y)
        F.gemm(aq, weights[0], ref, live)
        asc = shuffle_scale(aq.s).view(torch.float8_e8m0fnu)
        row = dict(m=192, live=144, n=n, k=k, copies=copies, split_k=0)
        try:
            def product():
                for q, s in zip(wq, ws):
                    aiter.gemm_a4w4_asm(aq.q.view(torch.float4_e2m1fn_x2), q, asc, s, y, log2_k_split=0)
            def complete():
                for q, s in zip(wq, ws):
                    F.quantize_into(x, aq, live)
                    scales = shuffle_scale(aq.s).view(torch.float8_e8m0fnu)
                    aiter.gemm_a4w4_asm(aq.q.view(torch.float4_e2m1fn_x2), q, scales, s, y, log2_k_split=0)
            product()
            # The last rotated weight differs; compare the first weight explicitly.
            aiter.gemm_a4w4_asm(aq.q.view(torch.float4_e2m1fn_x2), wq[0], asc, ws[0], y, log2_k_split=0)
            row['max_difference'] = float((y[:144].float() - ref[:144].float()).abs().max())
            assert torch.equal(y[:144], ref[:144]), row['max_difference']
            row.update(gemm_ms=timed(product, 8) / copies, complete_ms=timed(complete, 8) / copies)
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
        rows.append(row)
        path.write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
