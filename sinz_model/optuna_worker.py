"""PBS worker for one trial. Does not import torch until GPU UUID is fixed."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import socket
import subprocess
import time
import traceback
from pathlib import Path


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    try:
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def fix_cuda_visible_devices():
    """PyTorch 1.13 fails to parse PBS's single GPU UUID; map to nvidia-smi index."""
    original = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if original.startswith("GPU-") and "," not in original:
        report = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True,
        )
        matches = [index.strip() for index, uuid in csv.reader(io.StringIO(report))
                   if uuid.strip() == original]
        if len(matches) != 1:
            raise RuntimeError(f"Cannot uniquely resolve allocated GPU UUID {original}: {matches}")
        os.environ["CUDA_VISIBLE_DEVICES"] = matches[0]
    print(f"PBS GPU mask: {original!r} -> {os.environ.get('CUDA_VISIBLE_DEVICES')!r}", flush=True)


def execute(experiment: Path, batch: str, index: int):
    root = experiment.resolve()
    if not batch.startswith("batch-") or not batch[6:].isdigit() or index < 1:
        raise ValueError("Invalid batch or array index")
    manifest = json.loads((root / "batches" / batch / "manifest.json").read_text())
    records = {r["index"]: r for r in manifest["records"]}
    record = records[index]
    trial_dir = Path(record["container_trial_dir"]).resolve()
    if trial_dir.parent != root / "trials":
        raise ValueError("Unexpected trial directory outside experiment")
    cfg = json.loads((trial_dir / "config.json").read_text())
    if _digest(cfg) != record["config_sha256"]:
        raise RuntimeError("Trial configuration changed after sampling")
    existing = trial_dir / "result.json"
    if existing.exists():
        payload = json.loads(existing.read_text())
        if payload.get("status") == "completed" and payload.get("config_sha256") == record["config_sha256"]:
            print("Completed trial result already exists: %s" % existing, flush=True)
            return
    lock = trial_dir / ".worker-running"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Trial lock exists: {lock}; verify PBS jobs before retry") from exc
    try:
        with os.fdopen(fd, "w") as f:
            f.write(f"job={os.environ.get('PBS_JOBID', 'manual')} host={socket.gethostname()}\n")
        started = time.monotonic()
        try:
            fix_cuda_visible_devices()
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError(f"Exactly one CUDA device expected; got {torch.cuda.device_count()}")
            print("GPU:", torch.cuda.get_device_name(0), flush=True)
            from sinz_model.orchestrator import RunRegistry, candidate_config_sha256
            from sinz_model.trainer import train_member
            registry = RunRegistry(cfg["project"]["sqlite_path"])
            registry.initialize()
            workflows = registry.list_workflows()
            if workflows:
                if len(workflows) != 1:
                    raise RuntimeError("Expected one workflow per trial registry")
                workflow = workflows[0]["workflow_id"]
            else:
                workflow = registry.create_workflow(f"optuna-trial-{record['number']}")
            candidates = registry.list_candidates(workflow_id=workflow)
            if candidates:
                if len(candidates) != 1:
                    raise RuntimeError("Expected one candidate per trial registry")
                candidate_id = candidates[0]["candidate_id"]
                if registry.get_candidate(candidate_id)["config_sha256"] != candidate_config_sha256(cfg):
                    raise ValueError("Candidate config has changed")
            else:
                candidate_id = registry.create_candidate(workflow, "optuna", cfg, overrides=record["params"])["candidate_id"]
            runs = registry.candidate_runs(candidate_id)
            if len(runs) > 1:
                raise RuntimeError("Unexpected multiple runs for single-seed screening trial")
            if runs and runs[0]["status"] == "completed":
                run_id = runs[0]["run_id"]
            elif runs:
                run_id = runs[0]["run_id"]
                latest = Path(runs[0]["run_dir"]) / "checkpoints/latest.pt"
                if not latest.exists():
                    raise RuntimeError("Previous incomplete run lacks a recovery checkpoint; inspect manually")
                train_member(cfg, 0, candidate_id=candidate_id, resume_run_id=run_id,
                             evaluate_test=False)
            else:
                run_id = train_member(cfg, 0, candidate_id=candidate_id, evaluate_test=False)
            run = registry.get_run(run_id)
            if run["status"] != "completed":
                raise RuntimeError(f"Training run status is {run['status']}")
            registry.finish_candidate(candidate_id, status="completed")
            best = registry.best_validation(run_id)
            value = float(best["val_corr_mean"])
            if not math.isfinite(value):
                raise ValueError("Validation metric is not finite")
            summary = {"trial_number": record["number"], "config_sha256": record["config_sha256"],
                       "status": "completed", "objective": value,
                       "best_epoch": best["epoch"], "val_corr_median": best["val_corr_median"],
                       "run_id": run_id, "candidate_id": candidate_id,
                       "pbs_job_id": os.environ.get("PBS_JOBID"), "host": socket.gethostname(),
                       "gpu": torch.cuda.get_device_name(0), "seconds": time.monotonic() - started}
            _atomic_json(existing, summary)
            print(f"TRIAL COMPLETE: {record['number']} validation r={value}", flush=True)
        except Exception:
            err = traceback.format_exc()
            _atomic_json(existing, {"trial_number": record["number"],
                                    "config_sha256": record["config_sha256"],
                                    "status": "failed", "error": err,
                                    "pbs_job_id": os.environ.get("PBS_JOBID"),
                                    "host": socket.gethostname()})
            print(err, flush=True)
            raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    p.add_argument("--batch", required=True)
    p.add_argument("--index", type=int, required=True)
    a = p.parse_args()
    execute(Path(a.experiment), a.batch, a.index)


if __name__ == "__main__":
    main()
