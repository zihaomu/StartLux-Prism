# MI355X next-version speedup plan: native MXFP4 first

Status: proposed. Date: 2026-10-26.

The next version should prioritize **native MXFP4 on MI355X** to reduce
single-GPU 27B decision latency. The primary release target is **12 ms HTTP p50
for `readme3`**, compared with the current INT8 result of 20.54 ms. A 9 ms result
is a stretch goal. These are engineering targets, not measured FP4 results or
promised gains. No native MXFP4 runtime has been implemented or measured here.

The plan has two tracks:

- **Primary: native MXFP4 weights and activations**, with a separately qualified
  accuracy contract and selective higher precision for sensitive operations.
- **Reference and fallback: exact INT8**, retaining the current numerical
  contract and the kernel improvements below. Shared GDN and request-overhead
  improvements can benefit both tracks.

The comparison baseline is commit **`4534c11`**, including its 67 qualified
MI355X tiles. The earlier untuned port is useful historical context; release
claims must compare against the current version.

## 1. Workload and precision contracts

Keep the following fixed across the main comparisons:

- One AMD Instinct MI355X (`gfx950`), one model replica, one HTTP client.
- The same StartLux-Decision-v1.1-27B-AutoThink model revision, matching tokenizer
  and configuration, thinking disabled, text input and GPU graphs enabled.
- Cross-request prefix caching disabled; retain Prism's normal constant system
  prefix. Keep within-request evidence sharing and API decision settings fixed.
- Python 3.12, PyTorch `2.14.0+rocm10.1.0`, HIP `7.16.26385`, Triton `3.8.0`.
  Qualify a different stack as a separate experiment.

### Native MXFP4 track

Use packed FP4 E2M1 values with one E8M0 scale per 32 K-dimension values, for both
weights and dynamically quantized activations in selected projections. Execute
the product through `gfx950` scaled MFMA instructions. Prefer conversion from
the original BF16 model weights; identify and obtain that exact source revision
before conversion. Converting the existing Q8 or Q4 artifact adds quantization
error and must be labeled as a separate ablation, not as the original-weight
recipe. The existing INT8 SmoothQuant calibration cannot be assumed suitable.

Initially keep norms, softmax, GDN recurrent state and decision readout at their
validated BF16/FP32 precision. Quantize large MLP projections first, then extend
coverage only after per-layer and model-level sensitivity checks. Report the
actual layer coverage; a hybrid model is not an all-FP4 model.

FP4 changes the represented model and activations. Its logits need not equal
INT8 bit for bit. It must pass the independent kernel and decision-quality
gates in section 5, while preserving API semantics, usage, state isolation and
supported cache behavior. INT8 remains available when FP4 fails these gates.

### Exact INT8 track

Keep `StartLux-Decision-v1.1-27B-AutoThink-Q8_0.gguf` and the existing SmoothQuant
calibration fixed. Preserve exact INT32 projection sums, ordered FP32
scale/residual operations, activation quantization and BF16 conversion points.
This track must reproduce current logits, answers and affected seeded-generation
results exactly. Report its speedups separately from the FP4 track.

## 2. Measured baseline and release targets

The baseline measurements are in
[`mi355x_performance.md`](mi355x_performance.md) and
[`bench/results/mi355x_int8_20261026.json`](../bench/results/mi355x_int8_20261026.json).

| HTTP workload | Current INT8 p50 | Next-version target | Required speedup |
|---|---:|---:|---:|
| `yesno` | 15.96 ms | ≤10 ms | 1.60× |
| `review1` | 18.33 ms | ≤11 ms | 1.67× |
| `readme3`, primary | 20.54 ms | ≤12 ms | 1.71× |
| `startlux3` | 34.62 ms | ≤23 ms | 1.51× |
| `longpage2` | 96.03 ms | ≤65 ms | 1.48× |

Every row is a release target for the qualified next-version precision profile;
native MXFP4 is the primary candidate. Name the precision and coverage beside
every result. Record an unmet target as open; keep any partial improvement
explicitly scoped. Also require no >3% p50 or p95 regression in the expanded
request suite after accounting for run-to-run variation. FP4 must additionally
pass its quality gates; latency alone cannot qualify that profile.

The current `readme3` eager profile supplies the priority order:

| Component | Current GPU kernel time | Approximate share |
|---|---:|---:|
| INT8 GEMMs | 14.72 ms | 72.8% |
| GDN intra, recurrence, output, convolution prep and gated norm | 3.74 ms | 18.5% |
| Other kernels | 1.78 ms | 8.8% |

A working allocation for the 12 ms target is approximately 7.0 ms for complete
projection paths, 2.6 ms for GDN, 1.8 ms for other GPU work and 0.6 ms for remaining
request cost. Include FP4 activation quantization and scale/packing work in the
projection allocation, without counting it again under other kernels.
This allocation guides experiments. Eager profiler sums and HTTP wall time are
different measurements; the release gate uses actual HTTP latency.

Halving GEMM time alone would reduce the current 20.23 ms eager kernel sum to
about 12.87 ms. Reaching the primary HTTP target therefore also needs progress
outside GEMM or a larger GEMM reduction. Candidate speedups cannot be multiplied
together when they optimize overlapping work.

## 3. Interpret the computation and bandwidth numbers correctly

### Native FP4 opportunity and limits

AMD specifies **10.0663 PFLOPS dense MXFP4**, versus **5.0332 POPS dense INT8**
and **2.5166 PFLOPS BF16** for MI355X. The theoretical arithmetic-rate ratios are
approximately 2× versus INT8 and 4× versus BF16. The native FP4 route therefore
deserves priority, but these ratios do not predict whole-request speedups.
See the [AMD MI355X datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf).

For MXFP4, a 32-value block needs 16 bytes of payload and one byte of scale:
**4.25 bits/value**, or **0.53125 bytes/value** before padding and other metadata.
That is about 47% smaller than a one-byte INT8 payload, equivalent to a 1.88×
payload reduction factor. Measure complete resident storage and
traffic, including scales, mixed-precision layers and activation quantization.
The format and scale layout are documented in the
[OCP microscaling specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
and [Triton `dot_scaled` reference](https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html).

**Existing Q4 support does not supply this native FP4 path.**
`kernels/w4a8.py` consumes GGUF Q4_K/Q6_K integer weights with INT8 arithmetic;
`kernels/amd_w4a16.py` dequantizes k-quant weights to FP16 before multiplication.
Neither is packed E2M1/E8M0 native MXFP4. The current server accepts `bf16`,
`int8`, `w4a8` and `w4a16`, with `auto` selection; it has no `mxfp4` profile.
GGUF Q4_K_M cannot simply be reinterpreted as MXFP4.

To use the full FP4 compute-rate comparison, qualify native FP4 operands on
both sides of the product. A storage-only FP4 route or a mixed FP4/FP8 route
needs its own measured ceiling. Activation quantization, scale loads, layouts,
small M, padding, GDN and request overhead still matter. Do not infer native
execution from a Python API name: `dot_scaled` can also use software emulation.

### Current INT8 measurements and geometry

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

Start with P0, then F1–F3. Complete an early FP4 feasibility and quality review
before investing deeply in new INT8 GEMM designs. P3 GDN tuning benefits both
tracks and follows the largest remaining measured bottleneck. P1/P2 remain
the exact INT8 alternative if FP4 quality or native execution is inadequate.

### P0 — Freeze a stronger measurement and correctness harness

**Purpose:** make each experiment comparable to the current optimized version.

- Freeze request bodies, rendering/none-option settings, model/tokenizer/config
  identities, calibration identities and the baseline commit. Preserve the
  current INT8 fixtures and create an original-BF16 quality reference before
  evaluating FP4. Separate calibration/development data from held-out evaluation.
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

**Exit gate:** repeatable current-version results, exact INT8 references,
BF16 decision-quality references and a shape-by-shape time/traffic/padding
breakdown. Investigate >3% drift before ranking candidates. Keep checkpoint
loading and calibration outside timings; include runtime activation quantization.

### F1 — Prove native MXFP4 execution and qualify a quantization recipe

1. Compile a minimal packed E2M1/E8M0 GEMM on the installed `gfx950` stack.
   Evaluate Triton `tl.dot_scaled` and an applicable AITER kernel; verify their
   actual availability, supported shapes and capture behavior before choosing.
   Inspect disassembly for scaled FP4 MFMA instructions and rule out a full
   BF16/FP16 dequantization-and-GEMM fallback. A successful API call alone is
   insufficient. Pin any necessary stack change as a separate experiment.
2. Build an independent pack/unpack and scale reference. Cover scale axes,
   nibble order, rounding, saturation, zero blocks, extreme values, partial
   blocks and alignment. Specify handling of non-finite inputs. Confirm the
   kernel consumes the same quantized operands as this reference.
3. Convert original BF16 weights offline with a reproducible recipe and a
   versioned payload/scale manifest. Check available Quark tooling for this
   model rather than assuming a published recipe for another model transfers.
   Never substitute GGUF Q4_K bytes for FP4 payloads.
4. Measure weight-only error first, then the additional error from dynamic
   per-32 activation quantization. Use representative short, long and
   multi-question calibration prompts. Examine outlier blocks, each layer's
   output error and final decision margins before expanding FP4 coverage.
5. Keep sensitive projections at higher precision when an ablation shows a
   quality loss. Evaluate rescaling/clipping recipes on development data and
   freeze the selected recipe before running the held-out release suite.

**Exit gate:** demonstrated native scaled-MFMA execution, validated encoding,
an original-weight conversion manifest and a feasible recipe within section 5's
quality budget. If unavailable or inaccurate, record the specific blocker and
continue exact INT8 optimization; do not relabel W4A8 as native FP4.

### F2 — Optimize complete native FP4 projections for decision shapes

- Benchmark actual live/bucket pairs and projection layouts from P0, especially
  144 live rows in a 192-row bucket. Include body, read and sliced final-layer
  projections. Large square GEMM results cannot qualify these kernels.
- Time **activation quantization + packing/scales + GEMM + epilogue**, including
  all intermediate traffic. Compare separate and fused quantization only when
  their quantized values and scale semantics agree.
- Tune payload/scale layouts together, tile geometry, tail handling, weight
  reuse and XCD locality. Include layout conversion and scratch in the cost;
  prepack static weights outside the timed request.
- Record useful FP4 FLOPS, issued work, scale overhead, external read/write
  traffic, register/LDS use and memory bounds. Rotate weights beyond cache and
  separate hot-weight tests, as for INT8.
- Reprofile the model after each coverage change. Prioritize the next largest
  contributor, including GDN, rather than maximizing isolated FP4 peak rate.

**Landing gate:** section 5's numerical/quality checks and a repeatable ≥1.5×
whole-request gain for the primary workload versus the current tuned INT8
version. This is an intermediate gate; the release target remains ≤12 ms.
Keep INT8 dispatch for shapes or layers that fail the full-path latency gate.

### F3 — Integrate an explicit FP4 precision profile

- Add a proposed `--precision mxfp4` profile with explicit model-artifact,
  quantization-recipe and kernel identities. This option does not exist today.
  Report effective layer precisions in runtime metadata and benchmark results.
- Allocate payload, scale and scratch buffers before graph capture. Preserve
  lifetimes across replay, live-row masking, evidence sharing and state reuse.
- Validate artifact format, dimensions, scales and `gfx950`/runtime support at
  load. Do not silently interpret Q4_K_M as MXFP4. If FP4 cannot run, provide a
  clear error or an explicitly reported validated INT8 fallback with its own
  matching artifact; never change precision invisibly.
- Start with the requested single-GPU text decision workload. Qualify AutoThink,
  vision, tensor parallelism and other model sizes separately before advertising
  FP4 support for them. Preserve their existing validated paths.

**Exit gate:** graph-enabled end-to-end FP4 decisions, explicit coverage and
fallback metadata, passing quality/isolation checks and measured latency on
all five request suites. A standalone FP4 GEMM is not model-level support.

### P1 — Compare a library INT32 route on the real projection shapes

This is the exact INT8 track; its arithmetic requirements do not apply to FP4.

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

This work is shared by the INT8 and MXFP4 tracks.

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

Proceed after native FP4 integration or the selected INT8 improvements, and
GDN tuning, establish the remaining profile.

- Expand deferred residual folding where it removes an intermediate write and
  read while preserving the existing scale/add/norm order.
- Evaluate producer/consumer fusion for gated norm, quantization and MLP
  intermediates. Include any required row-max reduction and workspace traffic
  in the timing.
- Preserve the full-row activation scale. The existing W8A8 `EPI=2` path uses
  tile-local SwiGLU scales; it is not a direct replacement for the engine's
  current full-row SmoothQuant path. For FP4, preserve its independently
  qualified per-32 scale/rounding recipe when fusing producers or consumers.
- Evaluate a small number of additional graph buckets only if measured tail
  waste remains material. Include planner buffer sizing, chunk capacity, graph
  memory and cold-start cost in that experiment.
- Reprofile request rendering, staging and response handling with matched
  prompts. Allocate host-side work according to measured time saved.

**Landing gate:** exact outputs and ≥3% whole-request gain, with bounded graph
and scratch memory. Record smaller valid wins separately; they should not
displace a larger unresolved bottleneck.

## 5. Release validation

### Exact INT8 correctness

Require all of the following for the exact INT8 candidate:

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

### Native MXFP4 correctness and decision quality

Do not require FP4 logits to match INT8 exactly. Require the following instead:

- Compare encoding and scale generation against the independent reference.
  Compare GEMM/epilogue arithmetic against a high-precision computation on the
  **same quantized tensors**, separating kernel error from quantization error.
  Fix justified absolute/relative error limits before tuning; FP32 accumulation
  order need not match a CPU reduction bit for bit.
- Evaluate final decisions against original BF16 and current INT8 on a frozen,
  representative held-out suite: short/long evidence, multiple questions,
  wide option lists, none-options and near ties. Report task accuracy and
  per-suite counts, decision agreement, probability changes and margin flips.
  Existing 11-case fixtures are regression checks, not sufficient quality data.
- Use **provisional planning budgets** of ≤0.5 percentage-point aggregate task
  accuracy loss and ≤1 point for any suite versus original BF16. Freeze the
  final task-specific budgets and sufficiently large evaluation set before
  experimentation; these figures are proposals, not approved quality claims.
  Publish sample sizes and uncertainty, and compare against INT8 as well.
- Retain the validated precision of recurrent state, norms and decision readout
  unless separately qualified. Record all higher-precision exceptions and
  weight/activation calibration identities; do not tune on held-out failures.
- Check deterministic replay, usage and exact batch/cache isolation within the
  FP4 profile, including mixed live counts and shared evidence. A numerically
  approximate model still needs correct request and state isolation. If exact
  batch invariance cannot be retained, document that gap and keep the profile
  experimental rather than inheriting the existing guarantee.
- Exercise the common cache, graph, API and fallback cases above. Rerun TP,
  AutoThink or vision checks before extending FP4 support to those features.

Enable FP4 by default only after both quality and latency qualification.
Passing a microbenchmark or matching quantized GEMM arithmetic alone is
insufficient. Retain the exact INT8 path if FP4 does not meet the agreed budget.

### Performance and reproducibility

- Compare current commit `4534c11` against the candidate with identical inputs
  and API settings. Do not omit questions or substitute cache hits. For exact
  INT8 comparisons, keep weights, precision and calibration unchanged. For the
  FP4 comparison, precision and its conversion/calibration recipe intentionally
  change; freeze them before timing and report these changes with quality
  results. Use the same source model revision and matched precision of other
  features. Label hybrid coverage and FP4-specific stack changes explicitly.
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

Keep FP4 opt-in until its quality and feature coverage qualify it for default
selection. Store the INT8 fallback artifact separately and include its memory
and load costs if resident fallback is offered. Do not claim simultaneous
FP4/INT8 residency is free.

## 6. Milestones and deliverables

| Milestone | Deliverable | Exit condition |
|---|---|---|
| M0 | Frozen INT8/BF16 references, selector, live-row ledger and FP4 instruction probe | Reproducible `4534c11` results and demonstrated native scaled FP4 MFMA |
| M1 | Reproducible BF16-to-MXFP4 conversion and sensitivity/quality report | Valid encoding, independent kernel reference and feasible frozen quality recipe |
| M2 | Graph-enabled native FP4 projections and explicit precision profile | ≥1.5× primary request speedup versus tuned INT8, passing quality gates; all suite results reported |
| M3 | Selected FP4/hybrid coverage, GDN improvements and full qualification | Primary HTTP p50 ≤12 ms, all section 2 targets and quality gates, stable memory and usable INT8 fallback |
| INT8 alternative | Library/reuse/tail improvements under the original numerical contract | Separately reported exact-output results; same latency targets, no FP4 claim |
| Stretch | Further native FP4 reuse, GDN and fusion work | Primary HTTP p50 ≤9 ms while retaining the qualified FP4 quality contract |

Track each experiment with its hypothesis, code/config identity, affected
shapes, memory cost, correctness result, kernel result and whole-request result.
Reject a candidate that loses end-to-end latency even if its standalone GEMM
rate improves. Time-box a hypothesis that repeatedly fails its landing gate and
move to the next measured bottleneck.

Expected implementation touchpoints:

- A new MXFP4 packing/quantization and native GEMM implementation, model loader
  support and an offline conversion tool with a versioned artifact manifest.
- `startlux_prism/server.py` and precision options/dispatch in the engine.
- `startlux_prism/kernels/w8a8.py` and a separate MI355X INT8 GEMM module.
- `startlux_prism/kernels/quant.py`, especially deferred folds and row scales.
- `startlux_prism/kernels/gdn.py` and `engine/model.py`.
- `bench/mi355x_*.py`, `bench/profile.py` and the MI355X dispatch tests.
- The versioned qualification tables, Docker image and performance report.

A release report should include a raw-results manifest, separate INT8 exactness
and FP4 quality summaries, per-suite latency table, counter/roofline analysis,
quantization/coverage details, memory/cold-start costs, enabled dispatch ranges
and remaining gaps against the targets above.

## 7. Native FP4 references

- [AMD MI355X datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf): dense arithmetic peaks, HBM bandwidth and cache capacity.
- [OCP microscaling formats specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf): MXFP4 representation and block scaling.
- [Triton `dot_scaled` API](https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html): operand formats, packing and E8M0 scale layout; also documents possible software emulation.
- [AMD native MXFP4 serving example](https://rocm.blogs.amd.com/software-tools-optimization/kimi-k25-mxfp4-atom/README.html): a concrete `gfx950` scaled-MFMA integration using dynamic FP4 activations. This is evidence of the hardware/software route, not a ready-made StartLux integration or transferable accuracy result.
