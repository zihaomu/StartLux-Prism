"""Paired exact INT8 model comparison: vary only MI355X GDN launch parameters."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from startlux_prism import jevfmt as J, progress as P
from startlux_prism.engine.model import Engine, Options
from startlux_prism.kernels import gdn as G
from startlux_prism.prompt import rows_for
import suites


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf', required=True)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--calib', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--reps', type=int, default=100)
    a = p.parse_args()
    P.on()
    torch.set_num_threads(8)
    e = Engine(a.gguf, a.model_dir, Options(precision='int8', max_tokens=4096, prefix_cache=0, calib=a.calib))
    cases = {}
    for name in ('yesno', 'review1', 'readme3', 'startlux3', 'longpage2'):
        state, qs = suites.fixed()[name]
        cases[name] = rows_for(e.tok, [J.from_systemone(state, q, allow_none=q['type'] == 'choice')
                                     for q in qs.values()], list(qs))
    modes = dict(stock=dict(INTRA_WARPS=8, HREC_BV=16, OUT_BV=32, HREC_NS=1, OUT_NS=1),
                 candidate=dict(INTRA_WARPS=4, HREC_BV=64, OUT_BV=64, HREC_NS=1, OUT_NS=1))
    refs, report = {}, dict(baseline='4534c11 tuned INT8', modes=modes, measurements=[])
    out = Path(a.out)
    for round_no in range(a.rounds):
        for mode in (('stock', 'candidate') if round_no % 2 == 0 else ('candidate', 'stock')):
            for key, value in modes[mode].items():
                os.environ['STARTLUX_PRISM_' + key] = str(value)
            G.launch.cache_clear()
            e.body_graphs.clear()
            e.head_graphs.clear()
            e._pool = None
            e._init_prefix()
            for name, rows in cases.items():
                for _ in range(3):
                    got = e.letter_logits(rows)
                if name not in refs:
                    refs[name] = [x.clone() for x in got]
                assert all(torch.equal(x, y) for x, y in zip(got, refs[name])), (mode, name, 'logits changed')
                samples = []
                for _ in range(a.reps):
                    t0 = time.perf_counter()
                    again = e.letter_logits(rows)
                    samples.append((time.perf_counter() - t0) * 1000)
                    assert all(torch.equal(x, y) for x, y in zip(again, refs[name]))
                row = dict(round=round_no, mode=mode, suite=name, samples_ms=samples,
                           median_ms=statistics.median(samples), exact_logits=True)
                report['measurements'].append(row)
                out.write_text(json.dumps(report, indent=2) + '\n')
                print(round_no, mode, name, round(row['median_ms'], 3), 'ms exact', flush=True)


if __name__ == '__main__':
    main()
