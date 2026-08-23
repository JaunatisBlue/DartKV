# DartKV

DartKV is a small PyTorch reference implementation for experimenting with
compressed key/value (KV) caches. The first version prioritizes clear tensor
semantics and numerical checks over custom kernels. It supports asymmetric
group-wise 2/4/8-bit quantization, bit-packed storage, an optional full
precision sink prefix, and incremental cache updates.

## Scope of this baseline

The cache accepts tensors in `[batch, kv_heads, sequence, head_dim]` layout.
Each appended non-sink chunk is quantized independently along `head_dim`; the
cache reconstructs the complete sequence when `get()` is called. This is a
correctness baseline, not a production attention kernel. It intentionally does
not depend on Kitty's private Transformers branch, Triton, flash-attn, HQQ, or
any container image.

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

If the environment is missing, create it first with `conda create -n dartkv
python=3.10 -y`. Do not install the project into `base`. Verify the hardware
and PyTorch runtime before running GPU work:

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
```

The pinned PyTorch wheel was selected for the reference implementation. A
compatible NVIDIA driver is still required; the code also runs on CPU for
unit tests and small smoke tests.

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
Transformers runner can be added without changing the cache/quantizer API.

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
storage (packed values plus quantization metadata), while `dense_bytes` is the
size of the equivalent uncompressed K and V tensors.

## Reference provenance

The design was informed by the Kitty artifact and the papers in
`reference/paper`, especially their use of group-wise affine quantization and
sink/local buffering. Kitty is licensed under MIT; DartKV does not import its
package or copy its CUDA/Triton implementation. Planned follow-up work is
tracked separately: mixed-precision channel promotion, paged storage, model
integration, and kernel profiling.
