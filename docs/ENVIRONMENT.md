# Verified environment

This snapshot was captured on 2026-08-23 in `/home/yx/DartKV`.

| Item | Verified value |
| --- | --- |
| Conda environment | `dartkv` |
| Python | 3.10.20 |
| PyTorch | 2.7.1+cu126 |
| PyTorch CUDA runtime | 12.6 |
| Triton | 3.3.1 |
| CUDA available | `True` |
| GPU count | 2 |
| GPU | NVIDIA A100 80GB PCIe × 2 |
| Driver reported by `nvidia-smi` | 590.48.01 |
| NumPy | 2.2.6 |
| PyTest | 8.4.1 |
| Transformers | 4.53.2 |
| Safetensors | 0.8.0 |
| Accelerate | 1.14.0 |
| lm-eval (optional) | 0.4.12 |

The PyTorch installation pulls its CUDA 12.6 runtime wheels from the package
index. This is independent of the host toolkit, but still requires a
compatible NVIDIA driver. The reference path does not require flash-attn,
HQQ, or a container image. The optional evaluation stack additionally pulls
datasets, pandas, pyarrow and related metric packages. The verified local Qwen3 model is
`/opt/model/Qwen/Qwen-8B`; model weights are external to this repository.

To repeat the checks:

```bash
conda activate dartkv
python --version
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
```
