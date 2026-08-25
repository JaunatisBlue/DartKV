#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/yx/DartKV"
results_dir="${1:-results/kitty_qwen8_paper_b16}"
device="${2:-cuda:0}"
variant="${3:-kitty-pro}"
datasets_cache="/home/yx/.cache/dartkv/humaneval_hf_datasets"
humaneval_cache="/home/yx/.cache/huggingface/datasets/openai___openai_humaneval"

mkdir -p "${repo_root}/${results_dir}"
mkdir -p "${datasets_cache}"
if [[ ! -d "${datasets_cache}/openai___openai_humaneval" ]]; then
  cp -a "${humaneval_cache}" "${datasets_cache}/"
fi
exec bwrap \
  --ro-bind / / \
  --dev-bind /dev /dev \
  --proc /proc \
  --tmpfs /tmp \
  --unshare-net \
  --bind "${datasets_cache}" /home/yx/.cache/huggingface/datasets \
  --bind "${repo_root}/${results_dir}" "${repo_root}/${results_dir}" \
  --chdir "${repo_root}" \
  env HF_ALLOW_CODE_EVAL=1 HF_METRICS_CACHE=/tmp/hf_metrics \
  HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  /home/yx/miniconda3/envs/dartkv/bin/python examples/reproduce_kitty.py \
    --model /opt/model/Qwen/Qwen-8B \
    --task humaneval_instruct \
    --variant "${variant}" \
    --backend kitty-reference \
    --protocol paper \
    --device "${device}" \
    --batch-size 16 \
    --repeats 1 \
    --max-new-tokens 4096 \
    --confirm-run-unsafe-code \
    --output "${results_dir}"
