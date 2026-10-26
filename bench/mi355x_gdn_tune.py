"""Bounded MI355X GDN coordinate search, requiring exact model logits."""
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
    p.add_argument('--model-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--scaling', default='ceil')
    p.add_argument('--reps', type=int, default=30)
    a = p.parse_args()
    torch.set_num_threads(8)
    P.on()
    e = Engine(a.model_dir, a.model_dir, Options(precision='mxfp4', max_tokens=4096,
        prefix_cache=0, mxfp4_components='all', mxfp4_scaling=a.scaling))
    cases = {}
    for name in ('readme3', 'yesno', 'longpage2'):
        state, qs = suites.fixed()[name]
        cases[name] = rows_for(e.tok, [J.from_systemone(state, q, allow_none=q['type'] == 'choice')
                                     for q in qs.values()], list(qs))
    knobs = ('INTRA_WARPS', 'HREC_BV', 'OUT_BV', 'HREC_NS', 'OUT_NS', 'HREC_WARPS', 'OUT_WARPS')
    base = {k: G.launch(k) for k in knobs}
    report = dict(baseline=base, trials=[], selected=dict(base), scaling=a.scaling)
    path = Path(a.out)
    references, base_times = {}, {}

    def measure(config):
        report['pending'] = dict(config)
        path.write_text(json.dumps(report, indent=2) + '\n')
        for key, value in config.items():
            os.environ['STARTLUX_PRISM_' + key] = str(value)
        G.launch.cache_clear()
        e.body_graphs.clear()
        e.head_graphs.clear()
        e._pool = None
        e._init_prefix()
        row = dict(config=dict(config), suites={}, exact=True)
        for name, rows in cases.items():
            try:
                for _ in range(2):
                    got = e.letter_logits(rows)
                if name not in references:
                    references[name] = [x.clone() for x in got]
                if not all(torch.equal(x, y) for x, y in zip(got, references[name])):
                    row.update(exact=False, failed_suite=name)
                    break
                samples = []
                for _ in range(a.reps):
                    t0 = time.perf_counter()
                    e.letter_logits(rows)
                    samples.append((time.perf_counter() - t0) * 1000)
                row['suites'][name] = dict(median_ms=statistics.median(samples), samples_ms=samples)
            except Exception as error:
                row.update(exact=False, error=str(error), failed_suite=name)
                break
        if row['exact']:
            if not base_times:
                base_times.update({k: v['median_ms'] for k, v in row['suites'].items()})
            row['score'] = sum(v['median_ms'] / base_times[k] * (2 if k == 'readme3' else 1)
                               for k, v in row['suites'].items())
            row['no_regressions'] = all(v['median_ms'] <= base_times[k] * 1.03 for k, v in row['suites'].items())
        report['trials'].append(row)
        report.pop('pending', None)
        path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'suites'}),
              {k: round(v['median_ms'], 3) for k, v in row['suites'].items()}, flush=True)
        return row

    best = measure(base)
    config = dict(base)
    for knob, values in [('INTRA_WARPS', (4, 8)), ('HREC_BV', (16, 32, 64)),
                         ('OUT_BV', (32, 64, 128)), ('OUT_NS', (1, 2)),
                         ('HREC_WARPS', (4, 8)), ('OUT_WARPS', (4, 8))]:
        trials = [best]
        for value in values:
            if value != config[knob]:
                trials.append(measure(config | {knob: value}))
        eligible = [r for r in trials if r.get('exact') and r.get('no_regressions')]
        best = min(eligible, key=lambda r: r['score'])
        config = dict(best['config'])
        report['selected'] = config
        path.write_text(json.dumps(report, indent=2) + '\n')
    print('SELECTED', config, flush=True)


if __name__ == '__main__':
    main()
