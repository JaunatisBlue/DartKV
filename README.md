# DartKV

DartKV is a PyTorch reference implementation for experimenting with compressed
key/value (KV) caches. It prioritizes clear tensor semantics and numerical
checks over custom kernels. It supports asymmetric group-wise 2/4/8-bit
quantization, bit-packed storage, separate key/value quantization axes, an
optional full-precision sink prefix, page-sized buffering, and incremental
cache updates.

## Scope of this baseline

The cache accepts tensors in `[batch, kv_heads, sequence, head_dim]` layout.
Values are quantized along `head_dim`; keys use token groups along the sequence
axis, which matches the reference K/V distinction used by the second-stage
experiments. Dart promotion stores low two bits for every key channel and high
bits only for selected channels. `DartHFCache` implements the public
Transformers `Cache` lifecycle and materializes tensors for standard eager or
SDPA attention. The Qwen3-specific `dart-engine` adapter keeps prefill on the
selected Transformers backend and routes one-token decode through Dart's
packed-page fused attention; Python page orchestration remains a reference
implementation, so its throughput is not yet claimed to equal Kitty's
production Triton engine.

## Environment and installation

Run these commands from `/home/yx/DartKV`:

```bash
conda activate dartkv
python --version                 # Python 3.10.x
python -m pip install -r requirements.txt
python -m pip install -e .
```

For an exact snapshot of the verified environment, use
`requirements-lock.txt` in place of `requirements.txt` and then install the
editable project.

The standard-task smoke evaluator is optional and can be installed with
`python -m pip install -r requirements-eval.txt`; it is not needed for the
PyTorch cache or Qwen generation runner.

For the Kitty paper/system path, install the checked-in Kitty package and its
optional evaluation tools:

```bash
python -m pip install -e reference/code/Kitty --no-deps
python -m pip install matplotlib seaborn hqq
```

The paper used FlashAttention 2.7.4.post1.  It is a CUDA extension rather than
a portable wheel, so on an A100 build only the required `sm80` kernel (the
same command was used for the verified environment):

```bash
CUDA_HOME=/usr/local/cuda \
FLASH_ATTN_CUDA_ARCHS=80 \
MAX_JOBS=4 \
python -m pip install --no-build-isolation flash-attn==2.7.4.post1
```

Verify both the extension and the Transformers integration before a long run:

```bash
python -c 'import flash_attn, torch; print(flash_attn.__version__, torch.cuda.get_device_name())'
```

The latency benchmark defaults to `--attn-implementation flash_attention_2`,
matching Kitty's official script.  For `kitty-engine`, FlashAttention handles
prefill and Kitty's own Triton attention kernel handles decode.

If the environment is missing, create it first with `conda create -n dartkv
python=3.10 -y`. Do not install the project into `base`. Verify the hardware
and PyTorch runtime before running GPU work:

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
```

The pinned PyTorch wheel was selected for the reference implementation. A
compatible NVIDIA driver is still required; the code also runs on CPU for
unit tests and small smoke tests. The model integration additionally uses
`transformers==4.53.2`, `safetensors==0.8.0`, and `accelerate==1.14.0`.

## Smoke test and tests

```bash
python -m dart --device auto --bits 2 --group-size 32 --sink-tokens 2
python -m pytest -q
```

The smoke test prints the reconstructed shape, maximum absolute error, dense
and packed storage sizes, and the compression ratio. The GPU test is skipped
automatically when CUDA is unavailable.

## Synthetic attention baseline

The first reproducible baseline uses fixed-seed synthetic K/V states and one
query token. It compares dense attention with attention over the materialized
quantized cache and reports output error, CPU/GPU timing, and storage savings:

```bash
conda activate dartkv
python examples/run_baseline.py --device auto --tokens 256 --heads 8 --head-dim 128
```

This is deliberately a small reference experiment rather than a claim of
paper-level model accuracy. Once its numbers are stable, a model-specific
Transformers runner is available at `examples/run_qwen3.py`.

## Local Qwen3 baseline

The second-stage model runner uses the local model at
`/opt/model/Qwen/Qwen-8B`, so it does not download weights during a run. It
records generated text, prefill/decode timing, peak CUDA memory, and actual
cache storage under `results/` (which is ignored by Git):

```bash
python examples/run_qwen3.py --model /opt/model/Qwen/Qwen-8B \
  --cache dense --device cuda:0 --dtype bf16 --max-new-tokens 32

python examples/run_qwen3.py --model /opt/model/Qwen/Qwen-8B \
  --cache dart --device cuda:0 --dtype bf16 --sink-tokens 32 \
  --page-size 128 --key-group-size 128 --value-group-size 64 \
  --promote-ratio 0.25 --hold-partial-pages
```

The dense and Dart modes must use the same prompt, seed, dtype and attention
backend when comparing generated output. Dart's current adapter deliberately
materializes dense K/V for model attention; its storage result is meaningful,
but its latency is not evidence of a fused-kernel speedup.

## Page-streaming reference attention

For a kernel-oriented reference that does not call `cache.get()` first, run:

```bash
python examples/profile_reference.py --device cuda:0 --tokens 1024 \
  --kv-heads 8 --query-heads 32 --head-dim 128 --page-size 128 \
  --promote-ratio 0.25 --trace --output results/profile
```

The page-streaming path is deliberately slower than dense PyTorch attention at
this stage. It is the numerical and lifecycle oracle for future Triton/CUDA
kernels; see [docs/PROFILE.md](docs/PROFILE.md) for the recorded
error, memory and profiling results. The page field order and element-stride
contract is documented in [docs/PAGE_LAYOUT.md](docs/PAGE_LAYOUT.md); when
Triton is importable on CUDA, the reference uses its small dequantization
kernels before the still-unfused attention loop. For a single query token,
`dart.fused_dart_attention` additionally fuses page dequantization, QK, online
softmax state update and SV within each page; it is an intermediate kernel
oracle, not yet the model's Transformers attention backend.

## Minimal API

```python
import torch
from dart import DartKVCache, DartKVCacheConfig

cache = DartKVCache(DartKVCacheConfig(bits=2, group_size=64, sink_tokens=32))
keys = torch.randn(1, 8, 128, 128, device="cuda", dtype=torch.float16)
values = torch.randn_like(keys)
cache.update(keys, values)
next_keys = torch.randn(1, 8, 1, 128, device="cuda", dtype=torch.float16)
cache.update(next_keys, next_keys)
all_keys, all_values = cache.get()
```

`DartKVCache` owns detached inference tensors. `storage_bytes` counts tensor
storage (packed values, channel indices, and quantization metadata), while
`dense_bytes` is the size of the equivalent uncompressed K and V tensors.
Page descriptors are lifecycle-cached by device and invalidated after append,
`clear()` or `to(device)`:

```python
table = cache.page_table(device=keys.device)
runs = cache.page_runs(device=keys.device)
```

The returned page runs are consumed by the single-token fused attention
reference; mixed, sink, and pending boundary pages continue through the
per-page path.

## Kitty paper reproduction

The local reproduction runner mirrors Kitty's accuracy matrix (`K16V16`,
`KIVI-K2V2`, `KIVI-K2V2*`, `Kitty`, and `Kitty-Pro`) through the installed
`lm-eval` tasks. Dart keeps Kitty's one-token key PostQuant delay and an exact
128-token value Local buffer; these distinct K/V boundaries are required to
match the simulation lifecycle:

```bash
python examples/reproduce_kitty.py \
  --model /opt/model/Qwen/Qwen-8B \
  --task gsm8k_cot_llama --variant kitty-pro \
  --backend kitty-reference --protocol paper \
  --device cuda:0 --dtype fp16 --repeats 10 --max-new-tokens 4096
```

`--backend kitty-reference` executes the checked-in Kitty simulation semantics;
`--backend dart` runs the transported Dart cache. `--protocol paper` uses the
sampling settings stated in Section 5.1 and the 25% Kitty-Pro ratio from Table
3. `--protocol artifact` preserves the checked-in script's 20% ratio and lets
the task YAML control sampling. This distinction is intentional: the paper and
artifact disagree on both fields. Use `--limit 1` for a smoke run; limited runs
are isolated under `smoke_limit_1` and cannot be mistaken for full results.

The target cells, environment, known artifact conflicts, and Figure 4/5 setup
are recorded in `experiments/kitty_paper_manifest.json`. Audit completed full
results against the paper's reported maximum-deviation ranges with:

```bash
python examples/check_kitty_reproduction.py \
  --results results/kitty_repro --table table3 \
  --model-key Qwen3-8B --model-dir Qwen-8B \
  --protocol paper --backend kitty-reference
```

Figure 4's 11-point random/magnitude promotion sweep is resumable through:

```bash
python examples/sweep_kitty.py --model /opt/model/Qwen/Qwen-8B \
  --device cuda:0 --repeats 3 --max-new-tokens 4096
```

Use `--limit 8` to validate the sweep wiring before scheduling the full
GSM8K/MATH-Algebra curves.

Each repeat is checkpointed. Resumption only accepts an identical experiment
signature, including task limit and generation length. Result JSON records all
four seeds, cache settings, protocol, Python/PyTorch/Transformers/lm-eval/CUDA
versions, and GPU name.

For the paper's system-side generation comparison, use the local benchmark
counterpart. It also supports `--cache kitty-reference`, the Kitty simulation
cache, for a direct semantic baseline:

```bash
python examples/benchmark_kitty.py --model /opt/model/Qwen/Qwen-8B \
  --cache dart --protocol paper --batch-size 32 --max-seq-len 8192 \
  --warmup-runs 2 --repeat-runs 3 --device cuda:0

python examples/benchmark_kitty.py --model /opt/model/Qwen/Qwen-8B \
  --cache dense --protocol paper --attn-implementation flash_attention_2 \
  --batch-size 1 --max-seq-len 8192 --device cuda:0

python examples/benchmark_kitty.py --model /opt/model/Qwen/Qwen-8B \
  --cache static --protocol paper --batch-size 1 --max-seq-len 8192 \
  --device cuda:0

python examples/benchmark_kitty.py --model /opt/model/Qwen/Qwen-8B \
  --cache quanto --protocol paper --quantized-bits 4 --batch-size 1 \
  --max-seq-len 8192 --device cuda:0
```

The default `--protocol paper` retains Kitty's reference prompt and chat suffix
but fits it to exactly 100 tokens, matching the prompt length stated in Figure
5. `--protocol artifact` keeps the full checked-in `prompt_choice=1` GSM8K
few-shot prompt, which is 1179 Qwen3 chat-template tokens. This paper/artifact
distinction is recorded in every result; use `--no-chat-template` only for an
explicitly different protocol.

The benchmark reports both complete-sequence and newly-generated tokens/s,
peak allocated/reserved memory, storage ratio, and a prefix of generated token
IDs. The historical `tokens_per_second` field aliases complete-sequence
throughput, the Figure 5 metric. `kitty-reference` is the simulation cache and
does not claim the custom Kitty Triton engine's throughput.

`--cache dart-engine` attaches Dart's Qwen3 attention adapter: prefill uses the
selected Transformers backend and one-token decode consumes Dart packed pages
through the fused Triton attention path. This is the implementation-under-test;
`--cache kitty-engine` remains the checked-in Kitty reference engine.

The first complete A100 endpoint check (one warmup and one measured run) found
the following maximum successful batches. The Kitty endpoint provides the
paper's 8x batch-size gain and reaches 5.03x the local HF Dynamic FP16
throughput, meeting rather than falling short of the paper's reported 2.1x--4.1x
range. Exact commands, package versions, storage, and timing values are tracked
in `experiments/kitty_figure5_reproduction.json`.

| cache | maximum batch | sequence tokens/s | generated tokens/s |
| --- | ---: | ---: | ---: |
| HF Dynamic FP16 + FlashAttention 2 | 32 | 314.28 | 310.45 |
| Kitty engine + FlashAttention 2 prefill | 256 | 1579.93 | 1560.64 |

To reproduce Figure 5's increasing-batch protocol, run the subprocess-isolated
sweep (an OOM batch is recorded and terminates the sweep cleanly):

```bash
python examples/sweep_benchmark_kitty.py --model /opt/model/Qwen/Qwen-8B \
  --cache dense --protocol paper --attn-implementation flash_attention_2 \
  --max-seq-len 8192 --device cuda:0

python examples/sweep_benchmark_kitty.py --model /opt/model/Qwen/Qwen-8B \
  --cache kitty-engine --protocol paper --max-seq-len 8192 --device cuda:0
```

`--cache static` and `--cache quanto` reproduce the other two HF baselines in
Kitty's checked-in latency script. `--cache hqq` is also available for the HQQ
package used by the accuracy artifact; the paper's latency command labels its
INT4 baseline as HQQ in prose but passes `backend="quanto"` in code, so results
retain the selected backend name rather than merging the two.

Aggregate completed points and render the two Figure 5 panels with:

```bash
python examples/plot_kitty_figure5.py \
  --results results/kitty_figure5_paper \
  --output results/kitty_figure5_paper/figure5.png
```

The plot defaults to a consistent peak-allocated memory series. Passing
`--memory-metric reserved` plots only points that actually recorded peak
reserved memory; it never mixes allocated fallbacks into the same line. The
adjacent summary JSON retains both fields for audit.

The complete local package/build audit is tracked in
`experiments/kitty_install_manifest.json`; it includes FlashAttention and
Quanto `sm80` build validation, the math extras required by Kitty's MATH/AIME
tasks, and the latest test/`pip check` status.

For a bounded kernel/launch breakdown, profile a short decode window. The
default writes only a compact aggregate table; add `--profile-trace` only when
a potentially gigabyte-scale Chrome trace is needed:

```bash
python examples/benchmark_kitty.py --model /opt/model/Qwen/Qwen-8B \
  --cache kitty-engine --protocol paper --batch-size 8 --max-seq-len 256 \
  --profile --device cuda:0
```

## Reference provenance

The design was informed by the Kitty artifact and the papers in
`reference/paper`, especially their use of group-wise affine quantization,
channel promotion, and sink/page buffering. Kitty is licensed under MIT;
DartKV does not import its package or copy its CUDA/Triton implementation.
The current model adapter is intentionally a PyTorch reference path. Planned
follow-up work is fused dequantization/attention, stronger task evaluation,
and profiling-guided Triton/CUDA kernels.
