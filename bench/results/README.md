# AMD benchmark evidence

The source tree keeps three small, reproducible records:

- [MI355X INT8 summary](mi355x_int8_20261026.json): measured bandwidth/compute ceilings,
  matched latency results and profiler counters.
- [MI355X next-version summary](mi355x_next_summary_20261026.json): matched INT8/FP4
  HTTP p50/p95, quality screen, validation counts and source identities.
- [Frozen workload manifest](mi355x_fp4_frozen_20261026.json): exact HTTP requests,
  checkpoint/calibration hashes and pinned original BF16 shard identities.

The [execution report](../../docs/mi355x_next_version_results.md) includes the
protocol, rejected candidates and remaining quality/latency gates.

Raw captures, profiler traces, calibration files, logs and rejected research
scripts belong outside the checkout. This workspace archives them at:

```text
starlux_workspace/
  StartLux-Prism-mi355x/                 # source worktree
  models/                               # original checkpoints and FP4 artifacts
  artifacts/startlux-prism/2026-10-26/
    benchmark-results/                  # original tracked captures
    results/                            # local runs, traces, logs and calibration
    research/bench/                      # rejected experiments
    caches/                             # pytest, Python and rocprofiler scratch
    manifest.json                       # original paths, sizes and SHA256 hashes
```

All 23 formerly tracked captures/research scripts were preserved byte for byte.
They remain available remotely at the immutable
[published source snapshot](https://github.com/zihaomu/StartLux-Prism/tree/32c77d63aa3d485f9d145251407fb6a0796a74d8/bench).
Report links to archived evidence use that commit so they also work in a fresh clone.
The local archive is not required to build, serve or run unit tests.

For new measurements, select an output directory outside the checkout and pass
model, calibration and converted-artifact paths explicitly. For example:

```bash
ARTIFACTS=../artifacts/startlux-prism/2026-10-26
M=../models/StartLux-Decision-v1.1-27B-AutoThink
HIP_VISIBLE_DEVICES=0 python3 bench/mi355x_next_http.py \
  --model-dir "$M" --gguf StartLux-Decision-v1.1-27B-AutoThink-Q8_0.gguf \
  --calib "$ARTIFACTS/results/mi355x-27b-calib.pt" \
  --mxfp4-artifact ../models/StartLux-Decision-v1.1-27B-AutoThink-MXFP4-R32 \
  --manifest bench/results/mi355x_fp4_frozen_20261026.json \
  --rounds 3 --reps 100 --out "$ARTIFACTS/results/mi355x-next-http.json"
```
