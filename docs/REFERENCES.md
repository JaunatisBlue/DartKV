# Reference notes

## Code

- `reference/code/Kitty`: MIT-licensed reference artifact. Relevant pieces are
  `src/kitty_sim/utils_quant.py` (channel scoring and fake group quantization),
  `src/kitty/kvcache/kitty.py` (sink, local, and quantized cache lifecycle),
  and `src/kitty/kvcache/kernels/` (packed 2-bit/4-bit CUDA/Triton layout).
- DartKV's first baseline is intentionally a device-agnostic PyTorch version.
  It uses one uniform bit width per appended chunk and keeps metadata in a
  floating-point tensor. It does not claim to reproduce Kitty's paged kernel
  layout or latency.
- The second-stage path separates key token-axis groups from value
  head-dimension groups, and adds a PyTorch implementation of low two-bit
  storage plus selected four-bit key channels. `DartHFCache` uses the public
  Transformers cache interface but returns materialized tensors to attention;
  it is not a fused kernel.

## Papers

The repository currently contains papers on sliding-window KV quantization,
dynamic channel precision, query-aware mixed precision, rate-distortion
allocation, and transform coding. Before reporting a reproduction result,
record the exact paper, model checkpoint, tokenizer, context length, dataset,
random seed, bit budget, and metric. A paper's algorithmic statement and an
artifact's implementation details should be treated as separate sources of
truth.
