# 第三阶段参考路径与 profiling 记录

本阶段先实现可逐页面解量化的 PyTorch reference attention，再根据结果决定
是否写 Triton/CUDA kernel。入口是 `examples/profile_reference.py`，不会把完整
K/V 重新作为 attention 的输入；`streamed_dart_attention` 使用 online softmax
累积器逐段处理 `DartKVCache.iter_segments()` 返回的 packed page。

## 运行命令

```bash
conda activate dartkv
python examples/profile_reference.py \
  --device cuda:0 --tokens 1024 --page-size 128 \
  --key-group-size 128 --value-group-size 64 --sink-tokens 32 \
  --kv-heads 8 --query-heads 32 --head-dim 128 \
  --promote-ratio 0.25 --repeats 3 --output results/profile
```

需要 Chrome trace 时增加 `--trace`；它会分别导出 Python streaming 和 fused
page attention 两份 trace，trace 和 JSON 都保存在 `results/`，不会加入版本库。
固定 seed 后，streaming/fused 路径应与“对同一个量化 cache 先完整 materialize
再 dense attention”的结果一致；另行报告的 quantization error 则是量化 cache
与原始 FP16 K/V attention 的差异。

## 已验证的合成结果

条件为 `cuda:0`、seed `20260823`、`[B,Hkv,T,D]=[1,8,1024,128]`、
`Hq=32`、page/group size 分别为 `128/128/64`、32 个 sink token、25% key
channel promotion：

| 指标 | 结果 |
| --- | ---: |
| quantize and store (quantize + pack) | 224.18 ms |
| page segment unpack + dequantize | 0.517 ms |
| complete cache materialize (dequantize + concat) | 0.565 ms |
| dense attention | 0.157 ms |
| page-streamed attention | 51.66 ms |
| streamed vs materialized max abs error | `3.05e-5` |
| streamed vs materialized RMSE | `5.52e-7` |
| quantized vs original attention max abs error | `0.1272` |
| quantized vs original attention RMSE | `0.03276` |
| dense cache bytes | 4,194,304 |
| Dart packed cache bytes | 1,127,424 |
| logical compression ratio | 3.72× |
| segment dequantize temporary peak | 43,660,288 bytes |
| complete materialize temporary peak | 43,660,288 bytes |
| dense attention temporary peak | 51,934,208 bytes |
| streamed attention temporary peak | 43,713,024 bytes |

这里的 quantize-and-pack 是一次性 cache update，其他延迟为重复 3 次的平均值。
这个 reference 实现的 streaming 延迟明显高于 dense 矩阵乘法，因为 Python 循环
和逐页解量化尚未融合；它的价值是确定 kernel 的正确性、页面边界、GQA head
映射和显存生命周期。后续 kernel 优化必须先对齐这条路径，再报告性能收益。

## Qwen3 prompt suite

`examples/prompts.txt` 提供固定的一行一 prompt 文件，
`examples/run_qwen3_suite.py` 会保存每条 prompt 的生成、cache bytes、显存和
prefill/decode timing，并确保每条记录保留自己的 prompt 配置。模型加载仍按
prompt 重建，适合小规模验证；需要正式吞吐测试时应改成一次加载、多 prompt
批处理。

```bash
python examples/run_qwen3_suite.py \
  --model /opt/model/Qwen/Qwen-8B --cache dart --device cuda:0 \
  --dtype bf16 --max-new-tokens 32 --limit 4 \
  --sink-tokens 32 --page-size 128 --key-group-size 128 \
  --value-group-size 64 --promote-ratio 0.25 \
  --hold-partial-pages --output results/qwen3-suite
```

## 当前结论

- page-streaming reference 已能在不调用完整 cache materialize 的情况下完成
  GQA attention；
- 量化误差和 streaming 数值误差已经分离记录；
- 当前性能瓶颈是 Python page 循环、PyTorch unpack 和每段临时张量，而不是
  模型或 GPU 是否可用；
- 暂不直接复制 Kitty 的 Triton kernel。下一步应固定 page metadata/stride，
  再以小 kernel 对照测试逐步迁移 low/high bit unpack 和 attention。

## 第四轮 page split 与 Triton 对照

第四轮修正了一个页面边界问题：一次 update 中的多个完整 page 现在会逐页生成
segment，不能再把多个 page 合并为一个量化对象。布局、字段和 stride 约定见
[`docs/PAGE_LAYOUT.md`](PAGE_LAYOUT.md)。`streamed_dart_attention` 和 profile
入口在 CUDA 上会优先调用 `dart.triton_dequantize`；它仍然只是解包/反量化小
kernel，不是 fused attention。`fused_dart_attention` 则把单 page 的解包、QK、
online softmax 状态更新和 SV 放进同一个 Triton program。

profile 还会单独记录 `page_table_build_ms`。page table 是 Kitty 风格的
`[batch, page_count]` device descriptor，应在 cache append 或 beam reorder 后缓存，
不应在每个 decode token 内重复构造；本次小尺寸 smoke 的构建耗时约 `41 ms`，
说明它属于生命周期操作，不应混入稳定 decode kernel latency。

修正后的 A100 结果（`cuda:0`、seed `20260823`、`[1,8,1024,128]`、
`Hq=32`、page/key-group/value-group 为 `128/128/64`、32 sink、25% promotion、
重复 3 次）为：quantize+pack `232.89 ms`，分段 Triton unpack/dequantize
`1.11 ms`，完整 materialize `3.39 ms`，dense attention `0.151 ms`，page
streaming `136.28 ms`。streaming 对量化 dense 的 max abs/RMSE 为
`1.83e-4`/`3.59e-5`，packed storage `1,133,568` bytes，相对 dense
`4,194,304` bytes 为 `3.70×`；streaming 临时峰值为 `35,510,784` bytes。

在 mixed-key 配置下（25% promotion）不会错误合并 page，仍使用逐页 kernel；
`page_run_count=0` 是预期结果。uniform 配置（promotion=0）则把 sink 之后的
15 个 full page 合并为一个 run。A100、`[1,8,2048,128]`、32 query heads、
128 page、128/64 K/V group、32 sink、重复 5 次的实测为：

| path | average latency | temporary peak | max abs vs quantized dense | RMSE |
| --- | ---: | ---: | ---: | ---: |
| dense attention | 0.206 ms | 95,898,624 B | — | — |
| Python page streaming | 320.50 ms | 62,429,696 B | `1.53e-4` | `2.35e-5` |
| Triton fused page（逐页） | 4.42 ms | 61,845,504 B | `1.53e-4` | `2.15e-5` |
| Triton fused page-run（15 pages/launch） | 2.32 ms | 61,845,504 B | `1.53e-4` | `2.15e-5` |

page-run 相对逐页 fused 约减少 47% 延迟；本次重复 3 次的首次构建为
`page_table_build_ms=52.53 ms`、`page_run_build_ms=0.89 ms`，而缓存命中 lookup
分别只有 `0.0105 ms` 与 `0.0044 ms`。因此两类描述都应在 cache append 后缓存，
不计入稳定 decode token latency。它相对 dense attention 仍慢约一个数量级，说明
后续重点仍是更大的 query tile、物理 page table 和 mixed-run 融合，而不是把这组
reference 数字解读为端到端模型加速。

这个结果不能与第三轮的单 segment profile 直接比较：第四轮按真实 page 边界
拆分，并且 fused reference 仍由 Python wrapper 顺序发射 page kernel。下一步的
性能工作应把 page table、多个 page 的循环和请求 batch 进一步放进 kernel，而不
是把这个小 kernel 的单独延迟当作最终端到端结果。
