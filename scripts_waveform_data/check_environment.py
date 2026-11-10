"""Verify the TVSD container runtime, including *one visible GPU*.

The Compose device_ids=['0'] reservation exposes physical host GPU 0 as
cuda:0 inside this container. Run on the actual Docker host.
"""
import os
import sys
import importlib
from pathlib import Path

EXPECTED = {
    "numpy": "1.22.4",
    "torch": "1.13.1+cu117",
    "torchvision": "0.14.1+cu117",
    "transformers": "4.35.2",
    "scipy": "1.13.1",
    "h5py": "3.10.0",
    "yaml": "6.0.2",
}


def main():
    failures = []
    print("Python:", sys.version.split()[0])
    for module, expected in EXPECTED.items():
        try:
            mod = importlib.import_module(module)
            actual = mod.__version__
            good = actual == expected
            print(f"{'PASS' if good else 'FAIL'} {module}: {actual} (expected {expected})")
            if not good:
                failures.append(module)
        except Exception as exc:
            failures.append(module)
            print(f"FAIL {module}: {exc}")

    try:
        import torch
        count = torch.cuda.device_count()
        print("CUDA torch build:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())
        print("Visible GPU count:", count)
        print("NVIDIA_VISIBLE_DEVICES:", os.getenv("NVIDIA_VISIBLE_DEVICES"))
        if not torch.cuda.is_available() or count != 1:
            failures.append("GPU visibility (expected precisely host GPU 0 only)")
        else:
            print("Inside container cuda:0:", torch.cuda.get_device_name(0))
            torch.cuda.set_device(0)
            a = torch.randn(64, 64, device="cuda:0")
            b = (a @ a).sum().item()
            print("CUDA matmul test:", b)
        if torch.version.cuda != "11.7":
            failures.append("CUDA torch build")
    except Exception as exc:
        print("FAIL GPU test:", exc)
        failures.append("GPU")

    source = Path("/source/tvsd")
    destination = Path("/project/data")
    print("TVSD source present:", source.is_dir())
    print("TVSD source readable:", os.access(str(source), os.R_OK))
    print("Writable destination:", destination, os.access(str(destination), os.W_OK))
    if not source.is_dir():
        failures.append("source mount")
    if not os.access(str(destination), os.W_OK):
        failures.append("writable working directory")

    if failures:
        print("CHECKS FAILED:", ", ".join(failures))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
