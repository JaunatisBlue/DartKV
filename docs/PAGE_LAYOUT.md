# DartKV page metadata 与 stride 约定

第四轮冻结的是 PyTorch reference 与 Triton 对照所使用的逻辑布局。它不是
最终 GPU page allocator 的 ABI，但所有后续 kernel 都必须先遵守这套 token 顺序、
字段顺序和 element stride。

## 统一逻辑顺序

输入和恢复结果统一为 `[batch, kv_heads, sequence, head_dim]`。cache 中的
segment 按 sequence 顺序排列：

1. `sink_tokens`（若配置且仍在 sink 范围内）是 dense tensor；
2. 其后每个 full page 都是一个独立 quantized segment，长度不超过
   `page_size`；
3. `hold_partial_pages=True` 时，最后不足一页的是 dense pending tensor，下一次
   update 补足后才会变成 quantized page。

`DartKVCache._quantize_pages` 会逐页切分输入，即使一次 update 同时包含多个
page，也不会把多个 page 合并成一个量化对象。`cache.page_metadata()` 返回每个
segment 的 `index`、`sequence_start`、`token_count`、逻辑 shape，以及所有存储
字段的 shape 和 element stride。字段 stride 直接来自 PyTorch tensor，不能当作
byte stride 使用。

## Packed 字段

设 `G=ceil(T/group_size)`，`P=ceil(group_size/(8/bits))`，`K` 是被 promotion
的 channel 数。字段的最后一维是连续 packed byte，低位元素先写入低 bit。

| segment | 字段 | shape（逻辑 batch/head 省略） | 量化轴 |
| --- | --- | --- | --- |
| uniform key | `values` | `[B,H,D,G,P]` | token `T` |
| uniform key | `scale`, `zero_point` | `[B,H,D,G]` | token `T` |
| mixed key | `low_values` | `[B,H,D,G,P]` | token `T`，每 channel 2-bit |
| mixed key | `high_values` | `[B,H,K,G,P]` | promoted channel 的高 2-bit |
| mixed key | `channel_indices` | `[B,H,K]` | channel index |
| mixed key | `scale`, `zero_point` | `[B,H,D,G]` | token `T` |
| uniform value | `values` | `[B,H,T,P]` | head dimension `D` |
| uniform value | `scale`, `zero_point` | `[B,H,T,G]` | head dimension `D` |

`QuantizedTensor` 的 key segment 保留 `axis=-2`，value segment 保留
`axis=-1`；`MixedQuantizedKey` 固定为低 2-bit 与 promoted 高 2-bit。每个 page
独立计算 scale/min，不能跨 page 复用 metadata。

## Triton 对照接口

`dart.triton_dequantize(segment)` 在 CUDA 且 Triton 可导入时执行小型解包 kernel，
否则回退到 PyTorch reference。当前支持 uniform key/value 和 mixed key，输出
与 segment 的 `original_shape` 相同。`streamed_dart_attention` 已使用这个接口，
但尚未融合 QK、online softmax 和 SV；因此 Triton 调用次数多时可能比纯 PyTorch
reference 更慢。

逐元素对照：

```bash
conda activate dartkv
python -m pytest -q tests/test_triton_ops.py
```

GPU 对照允许 FP16 affine expression 的一个舍入单位误差（`atol=4e-3`）；这是
因为 PyTorch reference 可能先在 metadata dtype 中完成乘法，而 Triton reference
在 FP32 中累积后再写回输出 dtype。该误差不能替代 attention 输出和生成结果的
回归检查。
