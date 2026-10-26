# MI355X next-version execution results

Date: 2026-10-26. Branch: `feat/mi355x`.
Plan: [native MXFP4 first](mi355x_next_version_speedup_plan.md).

This is the incremental execution record. Each completed part adds its evidence,
measured outcome and remaining limitations. Projection results and model-level
latency/quality are reported separately.

## Baseline

The tuned INT8 implementation is `4534c11`; the approved work plan is `afeab60`.
Its recorded `readme3` HTTP p50 is 20.54 ms. See the
[INT8 performance report](mi355x_performance.md) for all five suites and raw data.
The next-version target is 12 ms; no FP4 model-level gain has been established.
The [frozen manifest](../bench/results/mi355x_fp4_frozen_20261026.json) records
the Q8 file/calibration/tokenizer hashes, seven exact HTTP request bodies and
the original BF16 revision and shard hashes.

## Part 1: native MXFP4 instruction feasibility

**Completed:** a new experimental `kernels/mxfp4.py` packs E2M1 payloads and E8M0
scales and executes `tl.dot_scaled` on the existing MI355X stack. Disassembly
contains `v_mfma_scale_f32_16x16x128_f8f6f4`, with `cbsz:4 blgp:4` selecting FP4
operands. This establishes native matrix execution, rather than a software
dequantization followed by BF16 GEMM. It does not establish model accuracy.

Stack unchanged: PyTorch `2.14.0+rocm10.1.0`, HIP `7.16.26385`, Triton `3.8.0`,
GPU `gfx950`. Tests use only `HIP_VISIBLE_DEVICES=0`; foreign jobs are left alone.

The original BF16 model source is
[`startlux-models/StartLux-Decision-v1.1-27B-AutoThink`](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-AutoThink/tree/dd62d5cd84a483d75bdd548880bbf08bdd3f67a4),
pinned at `dd62d5cd84a483d75bdd548880bbf08bdd3f67a4`. All 14 shards downloaded.

Independent validation: **13 tests passed**, covering FP32/BF16/FP16 encoding,
midpoint ties, signed zero, non-finite blocks, three K sizes and three native
GEMM configurations, partial output tiles and changing live counts under graph
replay. Products are checked against a CPU FP64 reference on independently
quantized operands, with an accumulation-error bound.

Reproduce: `HIP_VISIBLE_DEVICES=0 python3 -m pytest -q tests/test_mxfp4.py`.

## Part 2: initial streaming projection measurements

**Completed:** the first six-configuration sweep uses 144 live rows in a
192-row bucket. Each weight rotation exceeds 512 MiB, twice the chip's Infinity
Cache. Timings include dynamic activation quantization, packing and scales.
These are synthetic projection measurements, not whole-model speedups.

| Projection | Best GEMM | Quantization + GEMM | Useful FP4 TFLOPS, complete path |
|---|---:|---:|---:|
| MLP gate/up | 55.68 µs | 58.86 µs | 872 |
| MLP down | 61.54 µs | 70.47 µs | 364 |
| GDN input | 33.84 µs | 38.05 µs | 639 |
| GDN output | 23.60 µs | 28.46 µs | 318 |

The [initial sweep data](../bench/results/mi355x_fp4_initial_20261026.json) records
all six configurations. Timings are the lowest of five warmed graph timings,
used for candidate selection rather than release p50 estimates. Initial kernels use
ordinary row-major payloads/scales. MLP down has a long K and warrants further
layout/tile work. The earlier INT8 105 µs probe used 192 live rows, so it is not
a matching useful-work comparison with this 144-live-row sweep.

Next: matched INT8 comparisons, more tile/layout candidates and original-BF16
model quality/latency checks. The model-level 1.5× intermediate gate is open.
