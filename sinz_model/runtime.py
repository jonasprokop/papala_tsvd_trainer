from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a mapping")
    return cfg


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_sha256(cfg: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()


def save_resolved_config(cfg: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(copy.deepcopy(cfg), f, sort_keys=False)


# ---------------------------------------------------------------------------

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_state_dict(state_dict) -> str:
    h = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tensor.dtype).encode("ascii"))
        h.update(str(tuple(tensor.shape)).encode("ascii"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def _run_git(args):
    try:
        return subprocess.check_output(
            ["git"] + list(args),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def capture_git_state() -> Dict[str, Any]:
    commit = _run_git(["rev-parse", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    diff = _run_git(["diff", "--binary"]) or ""
    status = _run_git(["status", "--porcelain"]) or ""
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "diff": diff,
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


def capture_environment() -> Dict[str, Any]:
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": (
            torch.backends.cuda.matmul.allow_tf32
            if hasattr(torch.backends, "cuda")
            else None
        ),
        "cudnn_allow_tf32": (
            torch.backends.cudnn.allow_tf32
            if hasattr(torch.backends.cudnn, "allow_tf32")
            else None
        ),
    }
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except Exception:
        env["transformers"] = None
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    else:
        env["gpu_name"] = None
        env["gpu_capability"] = None
    return env


def seed_everything(seed: int, cfg: Dict[str, Any]) -> Dict[str, int]:
    seed = int(seed)
    seeds = {
        "root": seed,
        "python": seed,
        "numpy": seed,
        "torch_cpu": seed,
        "torch_cuda": seed,
        "dataloader": seed + 1,
    }

    random.seed(seeds["python"])
    np.random.seed(seeds["numpy"])
    torch.manual_seed(seeds["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeds["torch_cuda"])

    repro = cfg["reproducibility"]
    torch.backends.cudnn.benchmark = bool(repro["cudnn_benchmark"])
    torch.backends.cudnn.deterministic = bool(repro["cudnn_deterministic"])
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = bool(
            repro["cuda_matmul_allow_tf32"]
        )
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = bool(repro["cudnn_allow_tf32"])

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(
            bool(repro["deterministic_algorithms"])
        )
    return seeds


def capture_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------

import os
from pathlib import Path
from typing import Any, Dict, Optional
import torch

def atomic_torch_save(payload: Dict[str, Any], path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp = path_obj.with_name(path_obj.name + ".tmp")
    torch.save(payload, str(tmp))
    os.replace(str(tmp), str(path_obj))

def save_checkpoint(path, model, optimizer, epoch, phase, run_id, config,
                    split_sha256, rng_state, metrics=None, best_score=None,
                    best_epoch=None, completed_epoch=True, scheduler=None):
    payload = {
        "format_version": 2,
        "epoch": int(epoch),
        "completed_epoch": bool(completed_epoch),
        "next_epoch": int(epoch)+1 if completed_epoch else int(epoch),
        "phase": str(phase), "run_id": str(run_id), "config": config,
        "split_sha256": split_sha256,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": rng_state, "metrics": metrics or {},
        "best_score": best_score, "best_epoch": best_epoch,
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
    }
    atomic_torch_save(payload, path)
    return {"file_sha256": sha256_file(path),
            "parameter_sha256": hash_state_dict(model.state_dict())}

def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


# ---------------------------------------------------------------------------

import statistics
from collections import deque
from typing import Optional
import torch
from tqdm.auto import tqdm

def format_seconds(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    s = int(round(seconds)); h, s = divmod(s,3600); m,s = divmod(s,60)
    return f"{h:02d}:{m:02d}:{s:02d}"

class EpochTiming:
    def __init__(self, window=5):
        self.durations = deque(maxlen=int(window))
    def add(self, seconds):
        self.durations.append(float(seconds))
    def eta(self, epochs_remaining):
        if not self.durations: return None
        return statistics.median(self.durations)*max(int(epochs_remaining),0)

def gpu_memory_gb():
    if not torch.cuda.is_available(): return "cpu"
    a=torch.cuda.memory_allocated()/(1024**3); r=torch.cuda.memory_reserved()/(1024**3)
    return f"{a:.1f}/{r:.1f}G"

def batch_bar(iterable, epoch, total_epochs, phase, disable=False):
    return tqdm(iterable, desc=f"{phase} e{epoch+1:03d}/{total_epochs:03d}",
                unit="batch", leave=False, dynamic_ncols=True, disable=disable)

def epoch_bar(iterable, total, initial=0, disable=False):
    return tqdm(iterable, total=total, initial=initial, desc="epochs",
                unit="epoch", leave=True, dynamic_ncols=True, disable=disable)
