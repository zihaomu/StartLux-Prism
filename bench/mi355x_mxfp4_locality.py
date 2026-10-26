"""Compare XCD workgroup ordering on the existing FP4 projection candidates."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from startlux_prism.kernels import mxfp4 as F
from gemm_micro import timed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--table', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    data = json.loads(Path(a.table).read_text())
    report = dict(bucket=None, live=None, source_sha256=F.kernel_identity(), cases=[])
    torch.manual_seed(722)
    for entry in data['entries']:
        m, n, k = entry['shape']
        live_count = entry['live']
        x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
        aq, live = F.empty(m, k, 'cuda'), torch.tensor(live_count, device='cuda', dtype=torch.int32)
        F.quantize_into(x, aq, live, rotate=True)
        copies = max(4, math.ceil(512 * 1024**2 / (n * k * 17 / 32)))
        weights = [F.quantize(torch.randn((n, k), device='cuda', dtype=torch.bfloat16), rotate=True) for _ in range(copies)]
        y = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        base = entry['config'][:6]
        if len(base) == 5:
            base.append(16)
        for group_n in (0, 8):
            cfg = tuple(base + [group_n])
            def complete():
                for w in weights:
                    F.quantize_into(x, aq, live, rotate=True)
                    F.gemm(aq, w, y, live, cfg)
            ms = timed(complete, 4) / copies
            row = dict(shape=entry['shape'], live=live_count, config=cfg, complete_ms=ms)
            report['cases'].append(row)
            print(json.dumps(row), flush=True)
            Path(a.out).write_text(json.dumps(report, indent=2) + '\n')
        del x, aq, live, weights, y
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
