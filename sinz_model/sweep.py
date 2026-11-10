"""Cartesian structural/model sweep from the selected SEARCH founder.

SWEEP changes only config paths explicitly listed in a separate sweep-space YAML.
The Cartesian product is over *candidate configurations*.  Ensemble seeds are
replicates nested inside each candidate and never participate in the product.

Example sweep-space YAML::

    sweep:
      model.core.layer_name:
        - convnextv2.encoder.stages.1.layers.0
        - convnextv2.encoder.stages.1.layers.1
      model.readout.type:
        - gaussian
        - point_pooled

Each candidate is fully trained for all configured ensemble members and then
ensemble-evaluated on test.  No winner is selected automatically.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from .evaluate import evaluate_ensemble, save_evaluation
from .orchestrator import RunRegistry, candidate_config_sha256
from .trainer import train_member


def _registry(output_root: str) -> RunRegistry:
    return RunRegistry(str(Path(output_root).resolve() / "results.sqlite"))


def _workflow_file(output_root: str) -> Path:
    return Path(output_root).resolve() / "workflow.json"


def _load_workflow_id(output_root: str) -> str:
    path = _workflow_file(output_root)
    if not path.exists():
        raise FileNotFoundError(
            "No workflow.json in %s; SEARCH must establish the workflow first"
            % Path(output_root).resolve()
        )
    return str(json.loads(path.read_text())["workflow_id"])


def _set_existing(cfg: Dict[str, Any], dotted_path: str, value: Any) -> None:
    if dotted_path.startswith("project.") or dotted_path in (
        "training.member_seeds",
        "training.ensemble_members",
    ):
        raise ValueError("Sweeping %s is reserved for orchestration" % dotted_path)
    target: Any = cfg
    names = dotted_path.split(".")
    for name in names[:-1]:
        if not isinstance(target, dict) or name not in target:
            raise ValueError("Unknown sweep parameter %s" % dotted_path)
        target = target[name]
    if not isinstance(target, dict) or names[-1] not in target:
        raise ValueError("Unknown sweep parameter %s" % dotted_path)
    target[names[-1]] = copy.deepcopy(value)


def _load_space(path: str) -> Dict[str, List[Any]]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Sweep-space file must be a mapping")
    grid = raw.get("sweep", raw)
    if not isinstance(grid, dict) or not grid:
        raise ValueError("sweep space must be a non-empty mapping")
    out: Dict[str, List[Any]] = {}
    for key, spec in grid.items():
        values = spec.get("values") if isinstance(spec, dict) else spec
        if not isinstance(values, list) or not values:
            raise ValueError("Sweep parameter %s requires a non-empty value list" % key)
        out[str(key)] = copy.deepcopy(values)
    return out


def make_plan(
    base_cfg: Dict[str, Any], grid: Dict[str, List[Any]], max_candidates: int
) -> List[Dict[str, Any]]:
    keys = sorted(grid)
    count = 1
    for key in keys:
        # Validate path on an expendable config while counting combinations.
        probe = copy.deepcopy(base_cfg)
        _set_existing(probe, key, grid[key][0])
        count *= len(grid[key])
    if count > int(max_candidates):
        raise ValueError(
            "Grid contains %d candidate configurations; increase --max-candidates explicitly"
            % count
        )

    plan: List[Dict[str, Any]] = []
    seen = set()
    for index, selection in enumerate(itertools.product(*(grid[key] for key in keys))):
        overrides = dict(zip(keys, selection))
        cfg = copy.deepcopy(base_cfg)
        for key, value in overrides.items():
            _set_existing(cfg, key, value)
        sha = candidate_config_sha256(cfg)
        if sha in seen:
            # Different textual grid choices resolving to the same config are not
            # distinct scientific candidates.
            continue
        seen.add(sha)
        plan.append(
            {
                "index": index,
                "overrides": overrides,
                "config_sha256": sha,
                "config": cfg,
            }
        )
    return plan


def _existing_member(reg: RunRegistry, candidate_id: str, member_index: int):
    rows = reg.candidate_runs(candidate_id, member_index=int(member_index))
    if len(rows) > 1:
        raise RuntimeError(
            "Candidate %s has duplicate member_index=%d" % (candidate_id, member_index)
        )
    return rows[0] if rows else None


def _ensure_member(
    reg: RunRegistry, candidate_id: str, member_index: int, notes: str
) -> str:
    existing = _existing_member(reg, candidate_id, member_index)
    if existing is not None and existing["status"] == "completed":
        return str(existing["run_id"])
    if existing is not None:
        latest = Path(existing["run_dir"]) / "checkpoints" / "latest.pt"
        if not latest.exists():
            raise RuntimeError(
                "Existing %s run %s for candidate %s member %d has no recovery checkpoint"
                % (existing["status"], existing["run_id"], candidate_id, member_index)
            )
        execution_cfg = json.loads(existing["config_json"])
        return train_member(
            execution_cfg,
            member_index,
            candidate_id=candidate_id,
            notes=notes,
            resume_run_id=str(existing["run_id"]),
            evaluate_test=False,
        )
    cfg = reg.get_candidate_config(candidate_id)
    return train_member(
        cfg,
        member_index,
        candidate_id=candidate_id,
        notes=notes,
        evaluate_test=False,
    )


def _ensemble_output_dir(output_root: str, candidate_id: str) -> Path:
    return Path(output_root).resolve() / "ensembles" / candidate_id


def _evaluate_candidate_ensemble(
    reg: RunRegistry, output_root: str, candidate_id: str
) -> Dict[str, Any]:
    candidate = reg.get_candidate(candidate_id)
    cfg = reg.get_candidate_config(candidate_id)
    expected = int(cfg["training"]["ensemble_members"])
    completed = sorted(
        reg.candidate_runs(candidate_id, status="completed"),
        key=lambda row: int(row["member_index"]),
    )
    indices = [int(row["member_index"]) for row in completed]
    required = list(range(expected))
    if indices != required:
        raise RuntimeError(
            "Candidate %s ensemble incomplete: have %s, require %s"
            % (candidate_id, indices, required)
        )
    checkpoints = [reg.best_checkpoint(str(row["run_id"])) for row in completed]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metrics, collected = evaluate_ensemble(cfg, checkpoints, device)

    out = _ensemble_output_dir(output_root, candidate_id)
    eval_dir = out / "evaluation"
    summary = save_evaluation(
        str(eval_dir),
        candidate_id,
        metrics,
        collected,
        save_predictions=bool(cfg["evaluation"]["save_predictions"]),
    )
    provenance = {
        "candidate_id": candidate_id,
        "workflow_id": candidate["workflow_id"],
        "parent_candidate_id": candidate["parent_candidate_id"],
        "member_run_ids": [str(row["run_id"]) for row in completed],
        "member_indices": indices,
        "checkpoints": checkpoints,
        "aggregation": "arithmetic_mean_predictions",
    }
    out.mkdir(parents=True, exist_ok=True)
    provenance_path = out / "ensemble_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    reg.clear_candidate_metrics(candidate_id, evaluation_kind="ensemble", split="test")
    for metric_name, summary_key in (
        ("corr_to_average", "corr_to_average_summary"),
        ("single_trial_corr", "single_trial_summary"),
    ):
        sm = metrics[summary_key]
        reg.log_candidate_metric(
            candidate_id, "ensemble", "test", metric_name, "mean", sm["mean"]
        )
        reg.log_candidate_metric(
            candidate_id, "ensemble", "test", metric_name, "median", sm["median"]
        )
        reg.log_candidate_metrics(
            candidate_id,
            "ensemble",
            "test",
            metric_name,
            [
                {
                    "scope": "neuron",
                    "neuron_id": int(neuron_id),
                    "value": None if not np.isfinite(value) else float(value),
                }
                for neuron_id, value in enumerate(metrics[metric_name])
            ],
        )

    reg.log_candidate_artifact(
        candidate_id, "ensemble_evaluation_summary", str(eval_dir / "metrics.json")
    )
    reg.log_candidate_artifact(
        candidate_id, "ensemble_per_neuron_metrics", str(eval_dir / "per_neuron.csv")
    )
    reg.log_candidate_artifact(
        candidate_id, "ensemble_provenance", str(provenance_path)
    )
    reg.finish_candidate(candidate_id, "completed")
    return summary


def _resolve_parent(
    reg: RunRegistry,
    output_root: str,
    workflow_id: Optional[str],
    parent_candidate_id: Optional[str],
):
    if parent_candidate_id is not None:
        parent = reg.get_candidate(parent_candidate_id)
        if workflow_id is not None and parent["workflow_id"] != workflow_id:
            raise ValueError("parent candidate belongs to a different workflow")
        return parent
    resolved_workflow = workflow_id or _load_workflow_id(output_root)
    return reg.get_selected_candidate(resolved_workflow, stage="search")


def run_plan(args) -> None:
    root = Path(args.output_root).resolve()
    reg = _registry(str(root))
    reg.initialize()
    parent = _resolve_parent(reg, str(root), args.workflow_id, args.parent_candidate_id)
    base = reg.get_candidate_config(parent["candidate_id"])
    grid = _load_space(args.space)
    plan = make_plan(base, grid, args.max_candidates)
    n_members = int(base["training"]["ensemble_members"])
    for entry in plan:
        print(
            json.dumps(
                {
                    "index": entry["index"],
                    "config_sha256": entry["config_sha256"],
                    "overrides": entry["overrides"],
                    "ensemble_members": n_members,
                },
                sort_keys=True,
            )
        )
    print(
        "TOTAL %d candidates | %d member runs | parent=%s"
        % (len(plan), len(plan) * n_members, parent["candidate_id"])
    )


def run_sweep(args) -> None:
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    reg = _registry(str(root))
    reg.initialize()
    parent = _resolve_parent(reg, str(root), args.workflow_id, args.parent_candidate_id)
    workflow_id = str(parent["workflow_id"])
    base = reg.get_candidate_config(parent["candidate_id"])
    grid = _load_space(args.space)
    plan = make_plan(base, grid, args.max_candidates)

    print(
        "SWEEP workflow %s | founder=%s | %d candidate configurations"
        % (workflow_id, parent["candidate_id"], len(plan)),
        flush=True,
    )
    for i, entry in enumerate(plan):
        candidate = reg.get_or_create_candidate(
            workflow_id,
            "sweep",
            entry["config"],
            parent_candidate_id=str(parent["candidate_id"]),
            overrides=entry["overrides"],
            notes="SWEEP Cartesian candidate",
        )
        candidate_id = str(candidate["candidate_id"])
        reg.set_candidate_promoted(candidate_id, True)
        cfg = reg.get_candidate_config(candidate_id)
        n = int(cfg["training"]["ensemble_members"])
        print(
            "SWEEP %d/%d candidate=%s members=%d overrides=%s"
            % (i + 1, len(plan), candidate_id, n, candidate["overrides_json"]),
            flush=True,
        )
        try:
            for member_index in range(n):
                _ensure_member(
                    reg,
                    candidate_id,
                    member_index,
                    "SWEEP ensemble member",
                )
            summary = _evaluate_candidate_ensemble(reg, str(root), candidate_id)
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        except KeyboardInterrupt:
            reg.export_candidates(str(root / "sweep_candidates.csv"), workflow_id, "sweep")
            raise
        except Exception as exc:
            reg.finish_candidate(candidate_id, "failed")
            print("SWEEP FAILED candidate=%s error=%r" % (candidate_id, exc), flush=True)
            if args.fail_fast:
                reg.export_candidates(
                    str(root / "sweep_candidates.csv"), workflow_id, "sweep"
                )
                raise
        reg.export_candidates(str(root / "sweep_candidates.csv"), workflow_id, "sweep")
        reg.export_candidates(str(root / "candidates.csv"), workflow_id, None)

    print(str(root / "sweep_candidates.csv"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sinz-v1-sweep")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, func in (("plan", run_plan), ("run", run_sweep)):
        p = sub.add_parser(name)
        p.add_argument("--output-root", required=True)
        p.add_argument("--space", required=True)
        p.add_argument("--workflow-id", default=None)
        p.add_argument("--parent-candidate-id", default=None,
                       help="defaults to the selected SEARCH founder")
        p.add_argument("--max-candidates", type=int, default=64)
        if name == "run":
            p.add_argument("--fail-fast", action="store_true")
        p.set_defaults(func=func)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
