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

需要 Chrome trace 时增加 `--trace`；trace 和 JSON 都保存在 `results/`，不会
加入版本库。固定 seed 后，streaming 路径应与“对同一个量化 cache 先完整
materialize 再 dense attention”的结果一致；另行报告的 quantization error
则是量化 cache 与原始 FP16 K/V attention 的差异。

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
