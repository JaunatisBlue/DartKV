# Verified environment

This snapshot was captured on 2026-08-23 in `/home/yx/DartKV`.

| Item | Verified value |
| --- | --- |
| Conda environment | `dartkv` |
| Python | 3.10.20 |
| PyTorch | 2.7.1+cu126 |
| PyTorch CUDA runtime | 12.6 |
| CUDA available | `True` |
| GPU count | 2 |
| GPU | NVIDIA A100 80GB PCIe × 2 |
| Driver reported by `nvidia-smi` | 590.48.01 |
| NumPy | 2.2.6 |
| PyTest | 8.4.1 |

The PyTorch installation pulls its CUDA 12.6 runtime wheels from the package
index. This is independent of the host toolkit, but still requires a
compatible NVIDIA driver. The base implementation does not require
`transformers`, Triton kernels, flash-attn, HQQ, or a container image.

To repeat the checks:

```bash
conda activate dartkv
python --version
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
```
