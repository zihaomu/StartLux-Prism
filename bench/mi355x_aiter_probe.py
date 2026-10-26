"""Optional AITER Triton/Gluon FP4 comparison, using Prism's quantized values."""
import argparse
import json
import math
from pathlib import Path
import sys

bench_dir = Path(__file__).resolve().parent
# AITER imports cProfile; keep bench/profile.py from shadowing the stdlib module.
sys.path = [p for p in sys.path if Path(p).resolve() != bench_dir]
sys.path.insert(0, str(bench_dir.parent))
import torch
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from startlux_prism.kernels import mxfp4 as F
sys.path.insert(0, str(bench_dir))
from gemm_micro import timed, model_shapes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = []
    path = Path(a.out)
    for name, (n, k) in model_shapes(hidden=5120, ffn=17408, v_heads=48, n_head=24).items():
        if name.split()[0] not in ('mlp_gate_up', 'mlp_down', 'gdn_in', 'gdn_out'):
            continue
        x = torch.randn((192, k), device='cuda', dtype=torch.bfloat16)
        live = torch.tensor(144, dtype=torch.int32, device='cuda')
        aq = F.empty(192, k, 'cuda')
        F.quantize_into(x, aq, live)
        copies = max(4, math.ceil(512 * 1024**2 / (n * k * 17 / 32)))
        weights = [F.quantize(torch.randn((n, k), device='cuda', dtype=torch.bfloat16)) for _ in range(copies)]
        y = torch.empty((192, n), device='cuda', dtype=torch.bfloat16)
        for backend in ('triton', 'gluon'):
            row = dict(shape=name, backend=backend, live=144, bucket=192, copies=copies)
            try:
                def product():
                    for w in weights:
                        gemm_afp4wfp4(aq.q, w.q, aq.s, w.s, torch.bfloat16, y, backend=backend)
                def complete():
                    for w in weights:
                        F.quantize_into(x, aq, live)
                        gemm_afp4wfp4(aq.q, w.q, aq.s, w.s, torch.bfloat16, y, backend=backend)
                row.update(gemm_ms=timed(product, 4) / copies, complete_ms=timed(complete, 4) / copies)
            except Exception as e:
                row['error'] = f'{type(e).__name__}: {e}'
            out.append(row)
            print(json.dumps(row), flush=True)
            path.write_text(json.dumps(out, indent=2) + '\n')
        del weights, aq, x, y
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
