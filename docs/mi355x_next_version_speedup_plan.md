# MI355X next-version speedup plan

Status: proposed. Date: 2026-10-26.

The next version should deliver a substantial reduction in **single-GPU 27B
INT8 decision latency**. The primary release target is **12 ms HTTP p50 for
`readme3`**, compared with the current 20.54 ms. A 9 ms result is a stretch goal.
These are engineering targets, not measured results or promised gains.

The comparison baseline is commit **`4534c11`**, including its 67 qualified
MI355X tiles. The earlier untuned port is useful historical context; release
claims must compare against the current version.

## 1. Workload and numerical contract

Keep the following fixed for the primary comparison:

- One AMD Instinct MI355X (`gfx950`), one model replica, one HTTP client.
- `StartLux-Decision-v1.1-27B-AutoThink-Q8_0.gguf`, the matching tokenizer and
  configuration, and the existing SmoothQuant calibration.
- The INT8/W8A8 path, thinking disabled, text input, GPU graphs enabled.
- Cross-request prefix caching disabled; retain Prism's normal constant system
  prefix. Keep within-request evidence sharing and API decision settings fixed.
- Python 3.12, PyTorch `2.14.0+rocm10.1.0`, HIP `7.16.26385`, Triton `3.8.0`.
  Qualify a different stack as a separate experiment.
- Exact INT32 projection sums, the existing ordered FP32 scale/residual
  operations, and unchanged activation quantization and BF16 conversion points.

The candidate must reproduce the current version's logits, answers, usage,
cache behavior and seeded generation where those paths are affected.
Precision changes belong to separately evaluated profiles. They do not satisfy
this version's exact INT8 performance target.

## 2. Measured baseline and release targets

The baseline measurements are in
[`mi355x_performance.md`](mi355x_performance.md) and
[`bench/results/mi355x_int8_20261026.json`](../bench/results/mi355x_int8_20261026.json).

| HTTP workload | Current p50 | Next-version target | Required speedup |
|---|---:|---:|---:|
| `yesno` | 15.96 ms | ≤10 ms | 1.60× |
| `review1` | 18.33 ms | ≤11 ms | 1.67× |
| `readme3`, primary | 20.54 ms | ≤12 ms | 1.71× |
| `startlux3` | 34.62 ms | ≤23 ms | 1.51× |
| `longpage2` | 96.03 ms | ≤65 ms | 1.48× |

Every row is a release target. Record an unmet target as open; keep any partial
improvement explicitly scoped. Also require no >3% p50 or p95 regression in the
expanded request suite after accounting for run-to-run variation.

The current `readme3` eager profile supplies the priority order:

| Component | Current GPU kernel time | Approximate share |
|---|---:|---:|
| INT8 GEMMs | 14.72 ms | 72.8% |
| GDN intra, recurrence, output, convolution prep and gated norm | 3.74 ms | 18.5% |
| Other kernels | 1.78 ms | 8.8% |

A working allocation for the 12 ms target is approximately 7.0 ms for GEMMs,
2.6 ms for GDN, 1.8 ms for other GPU work and 0.6 ms for remaining request cost.
This allocation guides experiments. Eager profiler sums and HTTP wall time are
different measurements; the release gate uses actual HTTP latency.

Halving GEMM time alone would reduce the current 20.23 ms eager kernel sum to
about 12.87 ms. Reaching the primary HTTP target therefore also needs progress
outside GEMM or a larger GEMM reduction. Candidate speedups cannot be multiplied
together when they optimize overlapping work.

## 3. Interpret the computation and bandwidth numbers correctly

The machine sustained 5.77 TB/s in a large streaming-copy test and 2,785 TOPS in
an 8192³ library INT8 GEMM. These are reference measurements for those tests.
They are not guaranteed rates for each model projection.

The current representative projection probe uses `M=192, N=34816, K=5120`.
It delivers about 652 useful TOPS in 105 µs, with 368 MB of external L2 read
traffic, versus 178 MB of weights. Its read amplification is about 2.06× and
its MFMA utilization counter is 17.8%.

There are two separate padding effects to measure:

1. The existing engine benchmark computes **144 live tokens** for `readme3` but
   selects a **192-row bucket**.
2. Its selected 128-row GEMM tile issues **256 matrix rows** for that live count.
   Live rows occupy only 56.25% of those issued rows, before considering N-axis
   padding. The synthetic 192-live-row probe occupies 75% instead.

The current HTTP and engine harnesses render through different entry points.
Freeze matching decision settings and record HTTP-derived live counts before
using an engine shape as the HTTP workload's geometry.

For every kernel result, record live rows, allocated bucket, issued tile rows,
logical N/K, useful operations and estimated issued operations. For the current
non-split, whole-K case, a useful accounting model is:

```text
useful_ops = 2 * live_rows * N * K
issued_rows = ceil(live_rows / BM) * BM
issued_columns = ceil(N / BN) * BN
estimated_issued_ops = 2 * issued_rows * issued_columns * K
```

Report actual dispatch behavior when a new scheduler changes this model.
`FetchSize` measures L2/Fabric traffic and can include Infinity Cache service;
it does not directly measure HBM-only traffic. Lower traffic and lower latency
can accompany a lower observed bandwidth rate.

## 4. Ordered implementation work

### P0 — Freeze a stronger measurement and correctness harness

**Purpose:** make each experiment comparable to the current optimized version.

- Freeze request bodies, rendering/none-option settings, GGUF/tokenizer/config
  identities, calibration identity and the baseline commit.
- Extend the matrix harness to the actual engine layouts: output strides,
  sliced projections, row-scale layouts, pointer alignment and live-row masks.
  Include allocated/live pairs such as `192/144`, `384/361`, and `1280/1185`,
  plus zero, one and full live counts.
- Cover all 27B projection shapes, body buckets and read buckets. Include the
  last-layer query-only and KV-only slices, rather than only contiguous random
  matrices.
- Keep weight rotation beyond twice the 256 MB Infinity Cache for streaming
  microbenchmarks. Measure hot-weight cases separately and label them.
- Collect traces and hardware counters in separate runs from release timings.
  Record VGPRs, spills, LDS, active workgroups, MFMA utilization and external
  read/write bytes for the top kernels.
- Add a candidate selector that compares **current tuned** kernels with each
  proposed implementation. The existing `baseline` switch selects the older
  untuned rule, so it is insufficient by itself for next-version claims.

**Exit gate:** repeatable current-version results, exact-output references and
a shape-by-shape time/traffic/padding breakdown. Investigate >3% drift before
ranking candidates. Keep checkpoint loading and calibration outside timings.

### P1 — Compare a library INT32 route on the real projection shapes

**Hypothesis:** some shapes benefit from the installed library's INT8 GEMM
implementation, even after paying for its output processing.

1. Benchmark `torch._int_mm` at the real M/N/K values, layouts and live counts.
   The measured 8192³ result alone does not establish a win at these shapes.
2. For store epilogues, add an ordered scale/conversion kernel and measure the
   complete product-plus-epilogue path, including INT32 scratch traffic.
3. For residual-add epilogues, evaluate returning raw INT32 sums through the
   existing `w8a8.Deferred` interface. Let the next norm/fold reader perform the
   existing scale and residual operations.
4. Allocate scratch before graph capture, reuse it with explicit lifetimes and
   a documented memory bound, and test repeated graph replay. Verify support
   for every selected shape, stride, output buffer and capture mode.
5. Preserve `float(acc) * activation_scale`, then `* weight_scale`, then the
   residual add, with the existing rounding boundaries. Verify accumulation
   range for supported K and extreme signed inputs.

**Landing gate:** exact outputs, stable memory use, ≥15% gain on the selected
complete projection path and a repeatable ≥5% improvement in an affected
whole-request workload. Keep the current route for shapes that fail this gate.

**Likely files:** `kernels/w8a8.py`, `kernels/quant.py`,
`engine/model.py`, and a separate ROCm projection implementation module.

### P2 — Build GEMMs that reuse weights and handle tails efficiently

**Hypothesis:** reducing repeated weight fetches and padded rows produces the
largest gain beyond the current tile table. This is a kernel-design change.

Run bounded experiments in this order:

1. **Persistent N-tile processing.** Reuse a weight panel across multiple M
   fragments within a workgroup. Compare the saved external traffic against
   additional accumulators, LDS use and occupancy loss.
2. **Mixed main/tail fragments.** Prototype a 128-row main fragment plus a
   16-row tail for a 144-live-row case, sharing staged weights where possible.
   Compare with both current 256-issued-row work and independent tail kernels.
   Check generated instructions and counters; nominal dimensions alone do not
   establish that the compiler shares the loads.
3. **Small-body specialization.** Optimize the 55/64 and 77/128 live/bucket
   cases explicitly. The small last-layer read head also needs coverage, but
   head-only wins must be weighted by their actual contribution to the request.
4. **Workgroup ordering and XCD locality.** Test grouping that improves reuse
   in the XCD-local caches. Verify the effect through counters and timings.
5. **Deterministic split-K when parallelism is insufficient.** Accumulate
   partial products in INT32 scratch, reduce them exactly, then scale once.
   Benchmark the reduction, scratch reset and final consumer together. Preserve
   graph replay and the `Deferred`/tensor-parallel contracts. Keep FP32 atomic
   accumulation out of the exact path.
6. Inspect MFMA instruction selection, memory vectorization, pipeline overlap
   and bank conflicts on the winning candidates. Evaluate a HIP implementation
   if generated-code evidence identifies a concrete compiler limitation.

**Diagnostic target for the fixed 192-live-row probe:** ≤65 µs, or roughly
1,050 useful TOPS, and external read amplification ≤1.5× weight bytes. These are
experiment targets. Independently measure the real 144-live-row case and the
complete request. Higher utilization caused by extra padded work is not a win.

**Landing gate:** exact outputs, safe masked loads/stores, no scratch spills
that erase the gain, ≥20% gain on the affected dominant shapes and ≥10% gain
on an affected request relative to the accumulated next-version candidate.
Retain a shape-specific fallback. Avoid an unbounded startup search.

### P3 — Tune GDN for MI355X resources

The current `gfx950` GDN launch parameters match the conservative Radeon
profile: `HREC_BV=16`, `OUT_BV=32`, and one stage for those kernels. MI355X
measurements justify a separate search.

- Evaluate recurrence value widths 16/32/64 and output widths 32/64/128, with
  bounded stage and warp choices. Record register/LDS pressure and occupancy.
- Profile intra-chunk solving separately from recurrence and output. Optimize
  the largest measured component first.
- Keep `CHUNK=64`, segment boundaries, parent-state ordering, convolution taps,
  FP32 recurrent state and existing BF16 conversion points unchanged initially.
- Test cached prefixes, shared/unshared evidence, branch segments, graph replay
  and prefill/decode state isolation for every winning configuration.

**Target:** reduce the measured GDN contribution from about 3.74 to ≤2.6 ms on
the fixed primary workload. **Landing gate:** exact state/output checks and a
repeatable whole-request gain. Regrouping floating-point recurrence operations
requires independent numerical qualification and cannot silently replace the
exact path.

### P4 — Remove remaining intermediate traffic and launch work

Proceed after P1–P3 establish the remaining profile.

- Expand deferred residual folding where it removes an intermediate write and
  read while preserving the existing scale/add/norm order.
- Evaluate producer/consumer fusion for gated norm, quantization and MLP
  intermediates. Include any required row-max reduction and workspace traffic
  in the timing.
- Preserve the full-row activation scale. The existing W8A8 `EPI=2` path uses
  tile-local SwiGLU scales; it is not a direct replacement for the engine's
  current full-row SmoothQuant path.
- Evaluate a small number of additional graph buckets only if measured tail
  waste remains material. Include planner buffer sizing, chunk capacity, graph
  memory and cold-start cost in that experiment.
- Reprofile request rendering, staging and response handling with matched
  prompts. Allocate host-side work according to measured time saved.

**Landing gate:** exact outputs and ≥3% whole-request gain, with bounded graph
and scratch memory. Record smaller valid wins separately; they should not
displace a larger unresolved bottleneck.

## 5. Release validation

### Correctness

Require all of the following before enabling a candidate by default:

- Kernel products and epilogues match the current INT8 arithmetic bit for bit,
  including extreme inputs, sliced/non-contiguous outputs, partial live rows,
  zero live rows and repeated graph replay.
- Complete logits, probabilities, typed answers and usage match the frozen
  current-version fixtures. Preserve exact batch isolation and cache behavior.
- Cover cache miss/hit/eviction, shared/unshared evidence, mixed batches, wide
  option lists, vision coexistence and FP8-KV combinations that the affected
  path supports.
- If projection/deferred interfaces change, rerun INT32 shard tests and real
  two-/four-GPU parity checks. If recurrent state changes, rerun teacher forcing,
  joining sequences and seeded AutoThink isolation checks.
- Retain 4B/9B fallback smoke checks, unsupported-runtime fallback checks and
  the existing numerical thresholds. Expand the current 33-test suite to cover
  each new implementation; a fixed test count is not the acceptance criterion.

### Performance and reproducibility

- Compare current commit `4534c11` against the candidate with identical inputs
  and API settings. Do not regenerate calibration, change precision, omit
  questions or substitute cache hits during the comparison.
- Use an otherwise unoccupied visible GPU and stable power/clock policy. Record
  stack, device, memory use and relevant telemetry; do not change shared-host
  controls for the experiment.
- Use at least five paired engine rounds, alternating mode order, with at
  least 100 measured calls per suite per mode per round after warmup. Report
  round medians, pooled p50/p95 and variation across rounds.
- Run the same HTTP payloads through both servers with at least three
  alternating server pairs. Report p50/p95, errors and timeouts. Investigate
  outliers; retain them in the raw data.
- Include fresh evidence and a small mixed batch/concurrency set to detect
  regressions. Report throughput separately from the single-client target.
- Run at least 10,000 requests across the qualification suite to check stable
  graph/workspace memory and service behavior. Treat this as a finite regression
  check, not proof of long-duration production reliability.
- Keep hardware-counter runs separate from timing runs. Publish cache level,
  traffic definitions, useful/issued operation accounting and profiler overhead.

### Rollout and fallback

Select implementations through the existing backend dispatcher with qualified
architecture, GPU, runtime, source and layout metadata. Keep the `4534c11`
tuned implementation available as the next-version reference/fallback, distinct
from the older untuned `baseline` mode. Select before graph capture and require
restart or recapture when changing modes.

Record additional scratch and graph-memory bounds. Update the Docker image,
qualification tables and kernel identities together. An unsupported or unmatched
configuration must retain a validated implementation.

## 6. Milestones and deliverables

| Milestone | Deliverable | Exit condition |
|---|---|---|
| M0 | Frozen fixtures, current-version selector and live-row/traffic ledger | Reproducible `4534c11` results and exact references |
| M1 | Qualified library route and first reuse/tail prototypes | Primary HTTP p50 ≤16 ms, or a recorded experiment failure with the next hypothesis |
| M2 | Selected persistent/tail kernels and MI355X GDN configuration | Primary HTTP p50 ≤12 ms and the workload targets in section 2 |
| M3 | Complete correctness, mixed-workload and service qualification | No correctness failures, no unexplained >3% regressions, stable memory and usable fallback |
| Stretch | Further reuse/fusion experiments | Primary HTTP p50 ≤9 ms with the same numerical contract |

Track each experiment with its hypothesis, code/config identity, affected
shapes, memory cost, correctness result, kernel result and whole-request result.
Reject a candidate that loses end-to-end latency even if its standalone GEMM
rate improves. Time-box a hypothesis that repeatedly fails its landing gate and
move to the next measured bottleneck.

Expected implementation touchpoints:

- `startlux_prism/kernels/w8a8.py` and a separate MI355X GEMM module.
- `startlux_prism/kernels/quant.py`, especially deferred folds and row scales.
- `startlux_prism/kernels/gdn.py` and `engine/model.py`.
- `bench/mi355x_*.py`, `bench/profile.py` and the MI355X dispatch tests.
- The versioned qualification tables, Docker image and performance report.

A release report should include a raw-results manifest, exact-output summary,
per-suite latency table, counter/roofline analysis, memory/cold-start costs,
enabled dispatch ranges and remaining gaps against the targets above.
