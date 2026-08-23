# Second-stage experiment record

The runner writes raw JSON to `results/`, which is intentionally ignored by
Git because it contains generated text and machine-specific timings. The
commands and representative observations below are kept in the repository so
the experiment can be repeated.

## Qwen-8B local generation

Environment: `dartkv`, PyTorch 2.7.1+cu126, Transformers 4.53.2, one NVIDIA
A100 80GB (`cuda:0`), BF16, eager attention, seed `20260823`. The local model
snapshot is `/opt/model/Qwen/Qwen-8B`; its Hugging Face cache tree identifies
revision `b968826d9c46dd6066d109eabc6255188de91218`.

Prompt:

```text
Give a concise definition of a key-value cache.
```

Dense command:

```bash
python examples/run_qwen3.py \
  --model /opt/model/Qwen/Qwen-8B --cache dense --device cuda:0 \
  --dtype bf16 --max-new-tokens 4 --output results/qwen3
```

Dart command:

```bash
python examples/run_qwen3.py \
  --model /opt/model/Qwen/Qwen-8B --cache dart --device cuda:0 \
  --dtype bf16 --max-new-tokens 4 --sink-tokens 2 --page-size 8 \
  --key-group-size 8 --value-group-size 64 --promote-ratio 0.25 \
  --hold-partial-pages --output results/qwen3
```

Uniform K2V2 command (the same run with promotion disabled):

```bash
python examples/run_qwen3.py \
  --model /opt/model/Qwen/Qwen-8B --cache dart --device cuda:0 \
  --dtype bf16 --max-new-tokens 4 --sink-tokens 2 --page-size 8 \
  --key-group-size 8 --value-group-size 64 --promote-ratio 0 \
  --hold-partial-pages --output results/qwen3
```

All three modes generated the same four-token continuation, `A key-value cache`.
The observed cache and timing values were:

| mode | prefill ms | decode ms/token | peak allocated | cache bytes | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 573.95 | 52.26 | 16,397,081,600 | 1,916,928 | 1.00× |
| Dart uniform K2V2 | 738.58 | 88.71 | 16,396,233,728 | 1,050,624 | 1.82× |
| Dart mixed K2V2 + promoted K4 | 822.09 | 96.34 | 16,396,289,024 | 1,105,920 | 1.73× |

All three modes generated the same four-token continuation, `A key-value
cache`. The Dart storage reduction is a real packed-cache measurement for this
short run. The latency is not a speedup result: `DartHFCache` materializes
dense K/V tensors before standard model attention, and quantization work is
included in the reference path. Longer prompts, repeated measurements, and a
fused attention kernel are required before drawing performance conclusions.

## Standard-task smoke evaluation

The optional official `lm-eval==0.4.12` stack was installed separately from
the core dependencies. A one-sample `hellaswag` smoke run was executed with
the dense local model (the harness warns that `--limit` is for testing and the
number is not a publishable metric):

```bash
HF_HOME=/opt/model/.cache/huggingface \
HF_DATASETS_CACHE=/opt/model/.cache/huggingface/datasets \
lm_eval --model hf \
  --model_args pretrained=/opt/model/Qwen/Qwen-8B,dtype=bfloat16,local_files_only=True \
  --device cuda:0 --tasks hellaswag --limit 1 --batch_size 1 \
  --output_path results/lm-eval/qwen8b-hellaswag-1 --log_samples
```

The run completed successfully and produced `acc=0` and `acc_norm=0` for the
single selected example, with raw sample output saved under
`results/lm-eval/qwen8b-hellaswag-1`. This is an integration smoke check, not
a quality claim; a meaningful result requires the full validation split and
identical evaluation settings for every cache variant.

## Reproducibility requirements

Every future result should keep the JSON file from `results/` together with
the exact command, model revision or local model path, tokenizer version,
dtype, attention backend, seed, prompt/data identifier, input/new-token
lengths, cache configuration, and GPU selection. Do not commit model weights,
generated bulk data, or private tokens.

## Fourth-stage Qwen3 smoke

为确认 page split 修正没有破坏 Transformers cache 生命周期，使用本地
`/opt/model/Qwen/Qwen3-0.6B`、BF16、eager attention、seed `20260823`、相同
prompt `Define a KV cache in one sentence.`，比较 dense 与 Dart mixed（sink 2、
page 8、key/value group 8、25% promotion），均生成 `A KV cache`。这只是 cache
adapter smoke，不是 Triton fused attention 结果；模型 adapter 仍在每层 attention
前物化 K/V。

| mode | input/new tokens | cache bytes | dense bytes | ratio | generated |
| --- | ---: | ---: | ---: | ---: | --- |
| dense | 8 / 3 | 1,146,880 | 1,146,880 | 1.00× | `A KV cache` |
| Dart mixed | 8 / 3 | 616,448 | 1,146,880 | 1.86× | `A KV cache` |

## Fourth-stage fused page attention

在同一份固定 seed 的合成数据上，`examples/profile_reference.py` 同时运行
Python streaming、逐页 fused 和 page-run fused。后者直接读取 page-major packed
page，在一个 Triton program 内循环连续 uniform page，并跨页更新 online softmax
state。`[1,8,2048,128]`、32 query heads、128 page、128/64 K/V group、32 sink、
无 promotion、A100 `cuda:0`、重复 5 次的结果：

| path | average latency | temporary peak | max abs vs quantized dense | RMSE |
| --- | ---: | ---: | ---: | ---: |
| dense attention | 0.151 ms | 51,964,928 B | — | — |
| Python page streaming | 320.50 ms | 62,429,696 B | `1.53e-4` | `2.35e-5` |
| Triton fused page（逐页） | 4.42 ms | 61,845,504 B | `1.53e-4` | `2.15e-5` |
| Triton fused page-run（15 pages/launch） | 2.32 ms | 61,845,504 B | `1.53e-4` | `2.15e-5` |

相对 Python page streaming，page-run reference 约快 `138×`，相对逐页 fused
约减少 `47%` 延迟；相对 dense attention 仍慢约一个数量级。后续重复 3 次的
生命周期 profile 显示，首次 page table/page-run 构建为 `52.53 ms`/`0.89 ms`，
缓存命中 lookup 仅 `0.0105 ms`/`0.0044 ms`，因此应在 append 后缓存。该差距仍来自
packed metadata 访问、单 query-head program 和尚未融合的物理 page table/batch
调度，不能把这组数据写成端到端模型加速结论。Chrome trace 可用 profile 脚本的
`--trace` 导出 streaming 与 page-run attention 路径。

本次 page-table smoke 使用 `T=128,page=32,Hkv=2,Hq=4,D=32`，构建并验证
`DartPageTable` 的耗时为 `41.15 ms`。该成本来自 device tensor 创建和完整
不变量检查，应该在 cache append/reorder 后缓存；不能把它放入每 token decode 的
kernel latency。当前 fused wrapper 已接受预构建的 `page_table=` 参数，下一步
会把 descriptor tensor 直接传入 page-run kernel。

## Kitty accuracy reproduction entrypoint

论文第 5 节的准确率实验要求同一模型、task、few-shot、seed 下比较 K16V16、
KIVI-K2V2、KIVI-K2V2*、Kitty 和 Kitty-Pro。DartKV 新增
`examples/reproduce_kitty.py`，使用本地 `lm-eval` task，并将
`local_tokens=buffer_length=128` 映射为 Kitty 的 dense Q/Local buffer；变体配置
分别是 sink `0/32`、key/value 2-bit 和 key promotion `0/12.5%/25%`。首次建议
用 `--limit 1` 验证模型、task 和 cache 生命周期，再去掉 limit 运行完整数据集。

在本地 Qwen3-0.6B smoke（FP16、seed `20260823`、sink=2、buffer/page=8、
promotion=25%、生成 4 tokens）中，Kitty reference cache 与 DartHFCache 生成的
token ID 序列完全一致；这验证了 local suffix、quantization page 边界和
Transformers generation adapter 的基本对齐。该 smoke 不是论文 Table 3 的
8B accuracy 结果，完整结果仍需对应的 8B/任务数据与论文重复次数。
