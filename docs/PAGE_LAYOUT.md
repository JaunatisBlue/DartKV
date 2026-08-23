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
第四轮又增加 `fused_dart_attention`：它按 page 直接读取 packed fields，在同一
个 Triton program 中完成该 page 的解包、QK、online softmax 状态更新和 SV；
Python wrapper 只负责顺序遍历 page，尚未把 page table 也放入 kernel。因此它仍
是单 token 的中间 reference，不是完整模型 fused attention。

逐元素对照：

```bash
conda activate dartkv
python -m pytest -q tests/test_triton_ops.py
```

GPU 对照允许 FP16 affine expression 的一个舍入单位误差（`atol=4e-3`）；这是
因为 PyTorch reference 可能先在 metadata dtype 中完成乘法，而 Triton reference
在 FP32 中累积后再写回输出 dtype。该误差不能替代 attention 输出和生成结果的
回归检查。

单 token fused reference 的入口：

```python
from dart import fused_dart_attention

output = fused_dart_attention(query[:, :, -1:, :], cache, fallback=False)
```

`fallback=False` 会在 CPU、非单 token 或 Triton 不可用时直接报错，适合 kernel
对照；默认 `fallback=True` 会回退到 PyTorch streaming reference。

## Page table 使用方式

```python
table = cache.page_table(device=query.device).validate()
output = fused_dart_attention(query, cache, page_table=table, fallback=False)
```

`page_ids` 的形状为 `[B, N]`，对应 Kitty 的 batch-indexed page table；当前每个
row 的 ID 是 segment index，packed tensor 仍由 `DartKVCache` 持有。`sequence_starts`
和 `token_counts` 是 page-major 的 int32 tensor，`key_modes`/`value_modes` 使用
`DENSE_PAGE`、`QUANTIZED_PAGE`、`MIXED_PAGE` 三个 Dart 常量。page table 可以在
beam reorder 时调用 `reorder(indices)`，只重排 batch 行，不复制 packed page。

当前 fused wrapper 已从 page table 读取 page 数、token 数和 batch/device 元数据。
`DartKVCache.page_table()` 与 `DartKVCache.page_runs()` 会按 device 缓存描述对象；
每次 cache append、`clear()` 或 `to(device)` 都递增 `layout_version` 并使旧缓存
失效，因此 decode loop 不应在每个 token 内手动重新 stack。连续的 uniform full
page 可以进一步调用 `table.uniform_quantized_runs(cache)`，得到 `DartPageRun`：
所有 packed K/V 与 scale/min 都沿第 0 维按 page stack，`page_indices` 仍保留原
table 的逻辑 ID。

```python
table = cache.page_table(device=query.device).validate()
runs = cache.page_runs(device=query.device)
output = fused_dart_attention(
    query, cache, page_table=table, page_runs=runs, fallback=False
)
```

run 只合并连续、相同布局且长度等于 `page_size` 的 uniform quantized K/V；dense
sink、mixed key 和 pending tail 会终止 run，继续走逐页 kernel。传入 `page_runs=()`
可强制逐页对照。当前 run kernel 在一个 `(batch, query_head)` Triton program 内
循环 page，并跨页更新 online softmax 的 `m/l/o`；Python 仍负责生命周期与逻辑 ID
校验，物理 allocator 和 mixed run 留待后续里程碑。

`DartHFCache.page_table(layer_idx)` 与 `page_runs(layer_idx)` 透传对应层的缓存
描述；`reorder_cache(beam_idx)` 会重建被选 beam 的 layer cache，因此旧 table/run
不会继续复用。若要强制刷新当前逻辑 layout，可传 `rebuild=True`，但正常 decode
路径不需要这样做。

当需要复现 Kitty 的 accuracy simulation 时，将 `local_tokens` 设为
`buffer_length`。这样 prefill/decode 会持续保留最新 dense Q/Local suffix，只有
更早的完整 page 才进入 packed quantized storage；`local_tokens=0` 则保留 DartKV
原本的“只保留不足一页 pending”参考行为。
