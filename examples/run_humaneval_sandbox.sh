#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/yx/DartKV"
results_dir="${1:-results/kitty_qwen8_paper_b1}"
device="${2:-cuda:0}"
variant="${3:-kitty-pro}"

mkdir -p "${repo_root}/${results_dir}"
exec bwrap \
  --ro-bind / / \
  --dev-bind /dev /dev \
  --proc /proc \
  --tmpfs /tmp \
  --unshare-net \
  --bind "${repo_root}/${results_dir}" "${repo_root}/${results_dir}" \
  --chdir "${repo_root}" \
  env HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false \
  /home/yx/miniconda3/envs/dartkv/bin/python examples/reproduce_kitty.py \
    --model /opt/model/Qwen/Qwen-8B \
    --task humaneval_instruct \
    --variant "${variant}" \
    --backend kitty-reference \
    --protocol paper \
    --device "${device}" \
    --batch-size 1 \
    --repeats 3 \
    --max-new-tokens 4096 \
    --confirm-run-unsafe-code \
    --output "${results_dir}"
