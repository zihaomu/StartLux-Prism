"""Alternate baseline/tuned graphs on one loaded INT8 engine; require identical logits."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from startlux_prism import jevfmt as J
from startlux_prism.engine.model import Engine, Options
from startlux_prism.prompt import rows_for
import suites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--calib', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--reps', type=int, default=30)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--suites', default='yesno,review1,readme3,startlux3,longpage2')
    args = parser.parse_args()
    engine = Engine(args.gguf, args.model_dir, Options(precision='int8', max_tokens=4096,
                                                     prefix_cache=0, calib=args.calib))
    cases = {}
    for name in args.suites.split(','):
        state, questions = suites.fixed()[name]
        rows = rows_for(engine.tok, [J.from_systemone(state, q) for q in questions.values()], list(questions))
        cases[name] = [r.ids for r in rows]
    report = dict(gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  model=Path(args.gguf).name, prefix_cache=False, rounds=[])
    reference = {}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    for round_no in range(args.rounds):
        modes = ('baseline','tuned') if round_no % 2 == 0 else ('tuned','baseline')
        for mode in modes:
            os.environ['STARTLUX_PRISM_MI355X_GEMM'] = mode
            engine.body_graphs.clear()
            engine.head_graphs.clear()
            engine._pool = None
            for name, group in cases.items():
                for _ in range(3):
                    logits = engine.run([group])
                if name not in reference:
                    reference[name] = logits
                assert all(torch.equal(x,y) for x,y in zip(logits,reference[name])), (name,mode,'logits changed')
                samples = []
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    engine.run([group])
                    samples.append((time.perf_counter()-t0)*1000)
                row = dict(round=round_no, mode=mode, suite=name, median_ms=statistics.median(samples),
                           min_ms=min(samples), samples_ms=samples, bit_identical=True,
                           tokens=engine.last['tokens'], bucket=engine.last['bucket'])
                report['rounds'].append(row)
                path.write_text(json.dumps(report,indent=2)+'\n')
                print(mode,round_no,name,round(row['median_ms'],3),'ms',flush=True)


if __name__ == '__main__':
    main()
