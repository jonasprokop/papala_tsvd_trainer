#!/usr/bin/env python3
"""GPU integration smoke test of the *existing manual* training CLI (no Optuna).

Exercises two epochs: one frozen-core/readout epoch, then one joint epoch.
Uses bounded training batches but full validation to cover both macaques.
All outputs are isolated in one new project directory.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def select_pbs_gpu() -> None:
    """Translate PBS's UUID mask before anything can import torch."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if value.startswith("GPU-") and "," not in value:
        report = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True,
        )
        matches = [index.strip() for index, uuid in csv.reader(io.StringIO(report))
                   if uuid.strip() == value]
        if len(matches) != 1:
            raise RuntimeError("Cannot uniquely resolve PBS GPU UUID %r: %r" % (value, matches))
        os.environ["CUDA_VISIBLE_DEVICES"] = matches[0]
    print("CUDA_VISIBLE_DEVICES: %r -> %r" % (
        value, os.environ.get("CUDA_VISIBLE_DEVICES", "")), flush=True)


def build_config(base_config, output, data_root, image_root, dataset_module):
    from sinz_model.runtime import load_config

    cfg = load_config(str(base_config))
    cfg["project"].update(
        name="tvsd-model-smoke",
        output_root=str(output / "runs"),
        sqlite_path=str(output / "results.sqlite"),
        result_sheet_path=str(output / "runs.csv"),
    )
    cfg["data"]["root"] = str(data_root)
    cfg["data"]["image_root"] = str(image_root)
    cfg["data"]["dataset_module_path"] = str(dataset_module)
    cfg["data"]["preprocessing"]["normalization_path"] = str(data_root / "normalization.json")
    cfg["training"]["total_epochs"] = 2
    cfg["training"]["readout_only_epochs"] = 1
    cfg["training"]["ensemble_members"] = 1
    cfg["training"]["member_seeds"] = [8000]
    cfg["training"]["progress"]["enabled"] = True
    # The existing CLI defaults to evaluate_test=False; we do not pass its
    # --evaluate-test flag. Both epochs perform complete validation.
    return cfg


def verify(reg, candidate_id, *, train_batches):
    import torch

    runs = reg.candidate_runs(candidate_id)
    if len(runs) != 1:
        raise AssertionError("Expected exactly one physical run, got %d" % len(runs))
    run = runs[0]
    if run["status"] != "completed":
        raise AssertionError("Run status: %s" % run["status"])
    run_dir = Path(run["run_dir"])
    with reg.connect() as con:
        rows = [dict(r) for r in con.execute(
            "SELECT epoch, phase, train_loss, val_corr_mean FROM epochs WHERE run_id=? ORDER BY epoch",
            (run["run_id"],),
        )]
    if [(x["epoch"], x["phase"]) for x in rows] != [
        (0, "readout_only"), (1, "joint_finetune")
    ]:
        raise AssertionError("Two training phases not recorded: %r" % rows)
    if not all(math.isfinite(float(x[k])) for x in rows for k in ("train_loss", "val_corr_mean")):
        raise AssertionError("Non-finite training/validation metric: %r" % rows)

    phase1 = run_dir / "checkpoints" / "phase1_final_epoch_001.pt"
    final = run_dir / "checkpoints" / "last.pt"
    best = run_dir / "checkpoints" / "best.pt"
    latest = run_dir / "checkpoints" / "latest.pt"
    checkpoints = (phase1, final, best, latest)
    if not all(p.is_file() for p in checkpoints):
        raise AssertionError("Missing checkpoint(s): %r" %
                             [str(p) for p in checkpoints if not p.is_file()])

    # Confirm phase 2 actually updated core *parameters*, not just BatchNorm buffers.
    before = torch.load(phase1, map_location="cpu")["model_state_dict"]
    final_payload = torch.load(final, map_location="cpu")
    if final_payload["epoch"] != 1 or final_payload["phase"] != "joint_finetune":
        raise AssertionError("Unexpected final checkpoint phase/epoch")
    after = final_payload["model_state_dict"]
    names = [name for name, tensor in before.items()
             if name.startswith("core.") and tensor.is_floating_point()
             and (name.endswith("weight") or name.endswith("bias"))]
    changed = [name for name in names if not torch.equal(before[name], after[name])]
    if not changed:
        raise AssertionError("Joint-training epoch did not change any core weight/bias")

    validation = reg.best_validation(run["run_id"])
    return {
        "status": "passed",
        "run_id": run["run_id"],
        "candidate_id": candidate_id,
        "registry": str(reg.sqlite_path),
        "run_dir": str(run_dir),
        "train_batches_per_epoch": train_batches,
        "validation": "full (F and N)",
        "epochs": rows,
        "best_val_corr_mean": float(validation["val_corr_mean"]),
        "core_parameters_changed": len(changed),
        "test_evaluation_performed": False,
    }


def run(args) -> None:
    select_pbs_gpu()
    # No CUDA/PyTorch imports before fixing CUDA_VISIBLE_DEVICES.
    import torch
    from sinz_model.orchestrator import RunRegistry
    from sinz_model.runtime import save_resolved_config

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one usable CUDA device required, got %d" % torch.cuda.device_count())
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    data_root = Path(args.data_root).resolve()
    image_root = Path(args.image_root).resolve()
    dataset_module = Path(args.dataset_module).resolve()
    for item in (data_root / "dataset_info.json", data_root / "sample_index.csv",
                 data_root / "unit_to_neuron.csv", data_root / "normalization.json", dataset_module):
        if not item.is_file():
            raise FileNotFoundError(item)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an existing test run.
    cfg = build_config(Path(args.config), output, data_root, image_root, dataset_module)
    metadata = json.loads((data_root / "dataset_info.json").read_text())
    widths = {animal: int(metadata["n_neurons"][animal]) for animal in ("F", "N")}
    configured = {animal: int(cfg["model"]["readout"]["outdims_by_animal"][animal])
                  for animal in ("F", "N")}
    if configured != widths or sum(widths.values()) != int(cfg["model"]["readout"]["outdims"]):
        raise ValueError("Dataset/readout mismatch: %r vs %r" % (widths, configured))
    save_resolved_config(cfg, str(output / "resolved_config.yaml"))

    reg = RunRegistry(cfg["project"]["sqlite_path"])
    reg.initialize()
    workflow = reg.create_workflow("metacentrum-manual-training-smoke")
    candidate = reg.create_candidate(workflow, "smoke", cfg, notes="2-epoch trainer integration test")
    candidate_id = candidate["candidate_id"]
    reg.mark_candidate_running(candidate_id)

    # Deliberately use the public MANUAL CLI; Optuna is not imported or involved.
    command = [sys.executable, "-m", "sinz_model.cli", "train",
               "--output-root", str(output),
               "--candidate-id", candidate_id,
               "--member", "0", "--limit-train-batches", str(args.train_batches),
               "--notes", "model-smoke; test set disabled"]
    print("Manual training command:", " ".join(command), flush=True)
    # Use the actual public CLI, not a second implementation of training.
    # Keep git-state capture anchored at the mounted repository.
    subprocess.run(command, check=True, cwd=str(Path(__file__).resolve().parents[1]))
    summary = verify(reg, candidate_id, train_batches=args.train_batches)
    reg.finish_candidate(candidate_id, status="completed")
    reg.finish_workflow(workflow, status="completed")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("MODEL SMOKE PASSED\n" + json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True, help="New, nonexistent output directory")
    p.add_argument("--data-root", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--dataset-module", required=True)
    p.add_argument("--train-batches", type=int, default=2)
    args = p.parse_args()
    if args.train_batches < 1:
        p.error("--train-batches must be positive")
    run(args)


if __name__ == "__main__":
    main()
