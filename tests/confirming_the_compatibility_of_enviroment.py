import os
import sys
import subprocess

import numpy as np
import torch
import torchvision
import torchaudio
import transformers
import scipy
import skimage
import PIL
import huggingface_hub
import safetensors
import tokenizers
import pandas


def check(name, actual, expected):
    ok = actual == expected
    print(
        f"{'OK' if ok else 'FAIL':4s} | "
        f"{name:24s} | actual={str(actual):18s} | expected={expected}"
    )
    return ok


print("=" * 90)
print("V1 CONVNEXT REPLICATION — ENVIRONMENT COMPATIBILITY")
print("=" * 90)

results = []

# ---------------------------------------------------------------------
# Python / numerical stack
# ---------------------------------------------------------------------

results.append(check("Python", sys.version.split()[0], "3.9.6"))
results.append(check("NumPy", np.__version__, "1.22.4"))

results.append(
    check("PyTorch", torch.__version__, "1.13.1+cu117")
)
results.append(
    check("Torch CUDA build", torch.version.cuda, "11.7")
)
results.append(
    check("torchvision", torchvision.__version__, "0.14.1+cu117")
)

# torchaudio may expose the local CUDA suffix depending on packaging.
ta_ok = torchaudio.__version__.startswith("0.13.1")
print(
    f"{'OK' if ta_ok else 'FAIL':4s} | "
    f"{'torchaudio':24s} | actual={torchaudio.__version__:18s} | expected=0.13.1*"
)
results.append(ta_ok)

results.append(
    check("Transformers", transformers.__version__, "4.35.2")
)

results.append(
    check("SciPy", scipy.__version__, "1.13.1")
)
results.append(
    check("scikit-image", skimage.__version__, "0.19.1")
)
results.append(
    check("Pillow", PIL.__version__, "10.4.0")
)
results.append(
    check("huggingface-hub", huggingface_hub.__version__, "0.27.1")
)
results.append(
    check("safetensors", safetensors.__version__, "0.5.2")
)
results.append(
    check("tokenizers", tokenizers.__version__, "0.15.2")
)
results.append(
    check("pandas", pandas.__version__, "2.2.3")
)

# ---------------------------------------------------------------------
# CUDA / cuDNN
# ---------------------------------------------------------------------

print("\n" + "=" * 90)
print("CUDA / GPU")
print("=" * 90)

cuda_available = torch.cuda.is_available()
print("CUDA available:           ", cuda_available)

if not cuda_available:
    results.append(False)
else:
    print("Visible CUDA devices:     ", torch.cuda.device_count())
    print("CUDA device inside Docker:", torch.cuda.current_device())
    print("GPU name:                 ", torch.cuda.get_device_name(0))
    print("GPU capability:           ", torch.cuda.get_device_capability(0))

    # Physical host GPU 1 should be the only GPU exposed, and therefore
    # appears as cuda:0 inside the container.
    results.append(
        check("Visible GPU count", torch.cuda.device_count(), 1)
    )

print("NVIDIA_VISIBLE_DEVICES:   ", os.getenv("NVIDIA_VISIBLE_DEVICES"))
print("CUDA_VERSION env:         ", os.getenv("CUDA_VERSION"))
print("CUDNN_VERSION env:        ", os.getenv("CUDNN_VERSION"))

cudnn = torch.backends.cudnn.version()
print("PyTorch cuDNN runtime:    ", cudnn)
results.append(check("cuDNN runtime", cudnn, 8500))

# ---------------------------------------------------------------------
# Backend flags from original runtime
# ---------------------------------------------------------------------

print("\n" + "=" * 90)
print("PYTORCH BACKEND FLAGS")
print("=" * 90)

# Explicitly reproduce the reference runtime.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True

results.append(
    check("cudnn.enabled", torch.backends.cudnn.enabled, True)
)
results.append(
    check("cudnn.benchmark", torch.backends.cudnn.benchmark, False)
)
results.append(
    check("cudnn.deterministic", torch.backends.cudnn.deterministic, False)
)
results.append(
    check(
        "matmul.allow_tf32",
        torch.backends.cuda.matmul.allow_tf32,
        False,
    )
)
results.append(
    check(
        "cudnn.allow_tf32",
        torch.backends.cudnn.allow_tf32,
        True,
    )
)

# ---------------------------------------------------------------------
# Actual GPU computation
# ---------------------------------------------------------------------

print("\n" + "=" * 90)
print("GPU COMPUTE TEST")
print("=" * 90)

if cuda_available:
    try:
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x
        torch.cuda.synchronize()

        print("GPU matrix multiplication: OK")
        print("result mean:", y.mean().item())
        results.append(True)

    except Exception as exc:
        print("GPU matrix multiplication: FAIL")
        print(exc)
        results.append(False)

# ---------------------------------------------------------------------
# Transformers ConvNeXtV2 availability
# ---------------------------------------------------------------------

print("\n" + "=" * 90)
print("CONVNEXTV2 IMPLEMENTATION")
print("=" * 90)

try:
    from transformers import ConvNextV2Model, ConvNextV2Config

    print("ConvNextV2Model import:   OK")
    print("ConvNextV2Config import:  OK")
    results.append(True)

except Exception as exc:
    print("ConvNextV2 import: FAIL")
    print(exc)
    results.append(False)

# ---------------------------------------------------------------------
# pip dependency consistency
# ---------------------------------------------------------------------

print("\n" + "=" * 90)
print("PIP CHECK")
print("=" * 90)

pip_result = subprocess.run(
    [sys.executable, "-m", "pip", "check"],
    text=True,
    capture_output=True,
)

print(pip_result.stdout.strip() or pip_result.stderr.strip())

# Report it, but don't yet make this fatal:
# the historical Sinz base contains lots of unrelated old packages.
if pip_result.returncode == 0:
    print("pip dependency consistency: OK")
else:
    print(
        "pip dependency consistency: WARN "
        "(may be inherited historical-base baggage)"
    )

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

print("\n" + "=" * 90)

if all(results):
    print("ENVIRONMENT COMPATIBILITY: PASS")
else:
    print("ENVIRONMENT COMPATIBILITY: FAIL")
    failed = len(results) - sum(results)
    print(f"{failed} hard compatibility check(s) failed.")

print("=" * 90)