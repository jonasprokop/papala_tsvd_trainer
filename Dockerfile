# =============================================================================
# TVSD macaque V1 environment — phase 1: Docker and data reconnaissance.
# Starts from the original, verified Sinz V1 environment; no Musa code.
# Mount trainer code and data at runtime. This image does not modify either.
# =============================================================================

ARG BASE_IMAGE=sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_MODEL_ID=facebook/convnextv2-atto-1k-224 \
    HF_REVISION=2a62c34e37c1092d9a99cba587d5428837495d7c

WORKDIR /project

# Exact PyTorch stack observed in the reference container.
RUN python3.9 -m pip install --no-cache-dir --force-reinstall \
        torch==1.13.1+cu117 \
        torchvision==0.14.1+cu117 \
        torchaudio==0.13.1 \
        --extra-index-url https://download.pytorch.org/whl/cu117

# Hugging Face / numerical core.
RUN python3.9 -m pip install --no-cache-dir \
        numpy==1.22.4 \
        transformers==4.35.2 \
        huggingface-hub==0.27.1 \
        safetensors==0.5.2 \
        tokenizers==0.15.2

# Image preprocessing / released Cadena data loader.
RUN python3.9 -m pip install --no-cache-dir \
        scipy==1.13.1 \
        scikit-image==0.19.1 \
        Pillow==10.4.0 \
        pandas==2.2.3 \
        tqdm==4.61.2

# Inspection + test tools. Data-format-specific packages (e.g. Neo/NIX) are
# deliberately postponed until we have examined spikes_filtered.
RUN python3.9 -m pip install --no-cache-dir \
        PyYAML==6.0.2 \
        h5py==3.10.0 \
        pytest==8.4.2

# Enforce authoritative NumPy version after dependency resolution.
RUN python3.9 -m pip install --no-cache-dir --force-reinstall numpy==1.22.4

# Build-time environment check.
RUN python3.9 - <<'PY'
import sys
import numpy
import torch
import torchvision
import torchaudio
import transformers
import scipy
import h5py
import yaml
import pytest
import skimage
import PIL
import huggingface_hub
import safetensors
import tokenizers
import pandas
import tqdm

print("=" * 78)
print("V1 ConvNeXtV2 reference environment")
print("=" * 78)

versions = {
    "Python": sys.version.split()[0],
    "NumPy": numpy.__version__,
    "Torch": torch.__version__,
    "Torch CUDA build": torch.version.cuda,
    "Torchvision": torchvision.__version__,
    "Torchaudio": torchaudio.__version__,
    "cuDNN": torch.backends.cudnn.version(),
    "Transformers": transformers.__version__,
    "SciPy": scipy.__version__,
    "h5py": h5py.__version__,
    "PyYAML": yaml.__version__,
    "pytest": pytest.__version__,
    "scikit-image": skimage.__version__,
    "Pillow": PIL.__version__,
    "huggingface-hub": huggingface_hub.__version__,
    "safetensors": safetensors.__version__,
    "tokenizers": tokenizers.__version__,
    "pandas": pandas.__version__,
    "tqdm": tqdm.__version__,
}

for name, value in versions.items():
    print(f"{name:20s}: {value}")

assert sys.version.split()[0] == "3.9.6", sys.version
assert numpy.__version__ == "1.22.4"
assert torch.__version__ == "1.13.1+cu117", torch.__version__
assert torch.version.cuda == "11.7", torch.version.cuda
assert torchvision.__version__ == "0.14.1+cu117", torchvision.__version__
assert torchaudio.__version__.startswith("0.13.1"), torchaudio.__version__
assert transformers.__version__ == "4.35.2"
assert scipy.__version__ == "1.13.1"
assert h5py.__version__ == "3.10.0"
assert yaml.__version__ == "6.0.2"
assert pytest.__version__ == "8.4.2"
assert skimage.__version__ == "0.19.1"
assert PIL.__version__ == "10.4.0"
assert huggingface_hub.__version__ == "0.27.1"
assert safetensors.__version__ == "0.5.2"
assert tokenizers.__version__ == "0.15.2"
assert pandas.__version__ == "2.2.3"
assert tqdm.__version__ == "4.61.2"

cudnn = torch.backends.cudnn.version()
if cudnn is not None:
    assert cudnn == 8500, cudnn

print("=" * 78)
print("Environment verification PASSED")
print("=" * 78)
PY

# Historical backend settings. Set these again in the Python training entrypoint.
ENV V1_CUDNN_BENCHMARK=0 \
    V1_CUDNN_DETERMINISTIC=0 \
    V1_MATMUL_TF32=0 \
    V1_CUDNN_TF32=1

# No COPY on purpose: mount source/data into /project at runtime.
WORKDIR /project
CMD ["bash"]
