"""Candidate-based calibration search around a reproduction baseline.

Lifecycle
---------
1. ``screen`` creates many SEARCH candidates from an explicit variation-space
   YAML. Every candidate owns a full resolved config and is trained once using
   member 0 at the *final* epoch budget.
2. ``promote`` ranks screening results and optionally includes named controls.
3. ``replicate`` trains additional seeds for each promoted candidate and ranks
   candidates by validation correlations averaged across seeds.
4. ``ensemble`` evaluates every promoted five-member ensemble on test.
5. ``select`` chooses the founder from those ensemble test results.

Example search-space YAML::

    search:
      training.phase2_lr:
        type: loguniform
        low: 0.0001
        high: 0.001
      training.optimizer.weight_decay:
        type: choice
        values: [0.0, 0.00001, 0.0001, 0.001]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from .evaluate import evaluate_ensemble, save_evaluation
from .orchestrator import RunRegistry, candidate_config_sha256
from .runtime import load_config, save_resolved_config
from .trainer import train_member


def _registry(output_root: str) -> RunRegistry:
    return RunRegistry(str(Path(output_root).resolve() / "results.sqlite"))


def _set_existing(cfg: Dict[str, Any], dotted_path: str, value: Any) -> None:
    if dotted_path.startswith("project.") or dotted_path in (
        "training.member_seeds",
        "training.ensemble_members",
    ):
        raise ValueError("Searching %s is reserved for orchestration" % dotted_path)
    target: Any = cfg
    names = dotted_path.split(".")
    for name in names[:-1]:
        if not isinstance(target, dict) or name not in target:
            raise ValueError("Unknown search parameter %s" % dotted_path)
        target = target[name]
    if not isinstance(target, dict) or names[-1] not in target:
        raise ValueError("Unknown search parameter %s" % dotted_path)
    target[names[-1]] = copy.deepcopy(value)




def _validate_existing_path(cfg: Dict[str, Any], dotted_path: str) -> None:
    if dotted_path.startswith("project.") or dotted_path in (
        "training.member_seeds",
        "training.ensemble_members",
    ):
        raise ValueError("Searching %s is reserved for orchestration" % dotted_path)
    target: Any = cfg
    names = dotted_path.split(".")
    for name in names:
        if not isinstance(target, dict) or name not in target:
            raise ValueError("Unknown search parameter %s" % dotted_path)
        target = target[name]


def _load_space(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Search-space file must be a mapping")
    space = raw.get("search", raw)
    if not isinstance(space, dict) or not space:
        raise ValueError("search space must be a non-empty mapping")
    out: Dict[str, Dict[str, Any]] = {}
    for key, spec in space.items():
        if not isinstance(spec, dict):
            raise ValueError("Search parameter %s must map to a specification" % key)
        kind = str(spec.get("type", "")).lower()
        if kind not in ("choice", "uniform", "loguniform", "randint"):
            raise ValueError(
                "Search parameter %s has unsupported type %r" % (key, kind)
            )
        checked = copy.deepcopy(spec)
        checked["type"] = kind
        if kind == "choice":
            values = checked.get("values")
            if not isinstance(values, list) or not values:
                raise ValueError("choice %s requires non-empty values" % key)
        else:
            if "low" not in checked or "high" not in checked:
                raise ValueError("%s %s requires low and high" % (kind, key))
            low, high = checked["low"], checked["high"]
            if float(high) < float(low):
                raise ValueError("%s has high < low" % key)
            if kind == "loguniform" and (float(low) <= 0 or float(high) <= 0):
                raise ValueError("loguniform bounds must be positive for %s" % key)
        out[str(key)] = checked
    return out


def _load_explicit_plan(path: str) -> Optional[List[Dict[str, Any]]]:
    """An optional list of one-at-a-time experiments, instead of random sampling.

    YAML schema::
        candidates:
          - name: smaller_gamma
            overrides:
              training.loss.gamma_readout: 1.0

    The untouched baseline is inserted automatically as candidate zero.
    ``--trials`` counts that baseline; subsequent invocations with a larger
    ``--trials`` count reuse completed candidates and their checkpoints.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or "candidates" not in raw:
        return None
    if "search" in raw:
        raise ValueError("Use either candidates or search in a plan, not both")
    items = raw["candidates"]
    if not isinstance(items, list) or not items:
        raise ValueError("candidates must be a nonempty list")
    result = []
    names = set()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("candidate %d must be a mapping" % i)
        name = item.get("name")
        overrides = item.get("overrides")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("candidate names must be unique, nonempty strings")
        if not isinstance(overrides, dict) or not overrides:
            raise ValueError("candidate %s needs at least one override" % name)
        if not all(isinstance(key, str) for key in overrides):
            raise ValueError("candidate %s override keys must be strings" % name)
        names.add(name)
        result.append({"name": name, "overrides": overrides})
    return result


def _sample_value(rng: random.Random, spec: Dict[str, Any]) -> Any:
    kind = spec["type"]
    if kind == "choice":
        return copy.deepcopy(rng.choice(spec["values"]))
    if kind == "uniform":
        return rng.uniform(float(spec["low"]), float(spec["high"]))
    if kind == "loguniform":
        lo, hi = math.log(float(spec["low"])), math.log(float(spec["high"]))
        return math.exp(rng.uniform(lo, hi))
    if kind == "randint":
        return rng.randint(int(spec["low"]), int(spec["high"]))
    raise AssertionError(kind)


def _sample_overrides(
    rng: random.Random, space: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    return {key: _sample_value(rng, spec) for key, spec in sorted(space.items())}


def _apply_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    for key, value in overrides.items():
        _set_existing(cfg, key, value)
    return cfg


def _prepare_base_config(
    config_path: str,
    output_root: str,
    ensemble_members: int,
    member_seed: int,
    epochs: Optional[int],
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    root = Path(output_root).resolve()
    cfg["project"]["output_root"] = str(root / "runs")
    cfg["project"]["sqlite_path"] = str(root / "results.sqlite")
    cfg["project"]["result_sheet_path"] = str(root / "runs.csv")
    if epochs is not None:
        cfg["training"]["total_epochs"] = int(epochs)
    n = int(ensemble_members)
    if n < 1:
        raise ValueError("ensemble_members must be >= 1")
    cfg["training"]["ensemble_members"] = n
    cfg["training"]["member_seeds"] = [int(member_seed) + i for i in range(n)]
    return cfg


def _workflow_file(output_root: str) -> Path:
    return Path(output_root).resolve() / "workflow.json"


def _load_workflow_id(output_root: str) -> str:
    path = _workflow_file(output_root)
    if not path.exists():
        raise FileNotFoundError(
            "No workflow.json in %s; run SEARCH screen first" % Path(output_root).resolve()
        )
    payload = json.loads(path.read_text())
    return str(payload["workflow_id"])


def _create_or_resolve_workflow(
    reg: RunRegistry,
    output_root: str,
    workflow_id: Optional[str],
    workflow_name: str,
) -> str:
    path = _workflow_file(output_root)
    if workflow_id is not None:
        reg.get_workflow(workflow_id)
        resolved = str(workflow_id)
    elif path.exists():
        resolved = str(json.loads(path.read_text())["workflow_id"])
        reg.get_workflow(resolved)
    else:
        resolved = reg.create_workflow(workflow_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"workflow_id": resolved, "name": reg.get_workflow(resolved)["name"]},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return resolved


def _existing_member(reg: RunRegistry, candidate_id: str, member_index: int):
    rows = reg.candidate_runs(candidate_id, member_index=int(member_index))
    if len(rows) > 1:
        raise RuntimeError(
            "Candidate %s has duplicate member_index=%d" % (candidate_id, member_index)
        )
    return rows[0] if rows else None


def _ensure_member(
    reg: RunRegistry,
    candidate_id: str,
    member_index: int,
    notes: str,
) -> str:
    existing = _existing_member(reg, candidate_id, member_index)
    if existing is not None and existing["status"] == "completed":
        return str(existing["run_id"])

    if existing is not None:
        run_dir = Path(existing["run_dir"])
        latest = run_dir / "checkpoints" / "latest.pt"
        if not latest.exists():
            raise RuntimeError(
                "Existing %s run %s for candidate %s member %d has no recovery "
                "checkpoint; remove/restart that run explicitly"
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


def _log_screening_score(reg: RunRegistry, candidate_id: str, run_id: str) -> None:
    best = reg.best_validation(run_id)
    reg.clear_candidate_metrics(
        candidate_id, evaluation_kind="screening", split="val", metric="corr"
    )
    reg.log_candidate_metric(
        candidate_id, "screening", "val", "corr", "mean", best["val_corr_mean"]
    )
    reg.log_candidate_metric(
        candidate_id,
        "screening",
        "val",
        "corr",
        "median",
        best["val_corr_median"],
    )


def _candidate_ids(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _ensemble_output_dir(output_root: str, candidate_id: str) -> Path:
    return Path(output_root).resolve() / "ensembles" / candidate_id


def evaluate_candidate_ensemble(
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
            "Candidate %s ensemble incomplete: have members %s, require %s"
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
        rows = []
        for neuron_id, value in enumerate(metrics[metric_name]):
            rows.append(
                {
                    "scope": "neuron",
                    "neuron_id": int(neuron_id),
                    "value": None if not np.isfinite(value) else float(value),
                }
            )
        reg.log_candidate_metrics(
            candidate_id, "ensemble", "test", metric_name, rows
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


def run_screen(args) -> None:
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    reg = _registry(str(root))
    reg.initialize()
    workflow_id = _create_or_resolve_workflow(
        reg, str(root), args.workflow_id, args.workflow_name
    )
    explicit_plan = _load_explicit_plan(args.space)
    space = {} if explicit_plan is not None else _load_space(args.space)
    base = _prepare_base_config(
        args.config,
        str(root),
        ensemble_members=args.ensemble_members,
        member_seed=args.member_seed,
        epochs=args.epochs,
    )

    # Validate every declared path before spending GPU time.
    override_keys = (
        {key for entry in explicit_plan for key in entry["overrides"]}
        if explicit_plan is not None else space.keys()
    )
    for key in override_keys:
        _validate_existing_path(base, key)

    baseline = reg.get_or_create_candidate(
        workflow_id,
        "search",
        base,
        parent_candidate_id=None,
        overrides={},
        notes="SEARCH baseline",
    )
    baseline_id = str(baseline["candidate_id"])

    rng = random.Random(int(args.search_seed))
    candidates: List[str] = [baseline_id]
    seen = {candidate_config_sha256(base)}
    attempts = 0
    target = int(args.trials)
    if target < 1:
        raise ValueError("trials must be >= 1")

    if explicit_plan is not None:
        if target > len(explicit_plan) + 1:
            raise ValueError("Plan has %d total candidates (including baseline); requested %d"
                             % (len(explicit_plan) + 1, target))
        for entry in explicit_plan[:target - 1]:
            overrides = entry["overrides"]
            cfg = _apply_overrides(base, overrides)
            sha = candidate_config_sha256(cfg)
            if sha in seen:
                raise ValueError("Candidate %s resolves to a duplicate configuration"
                                 % entry["name"])
            seen.add(sha)
            candidate = reg.get_or_create_candidate(
                workflow_id, "search", cfg, parent_candidate_id=baseline_id,
                overrides=overrides,
                notes="SEARCH controlled candidate: " + entry["name"],
            )
            candidates.append(str(candidate["candidate_id"]))
    else:
        while len(candidates) < target:
            attempts += 1
            if attempts > max(1000, target * 100):
                raise RuntimeError(
                    "Could not generate %d unique search candidates from this space" % target
                )
            overrides = _sample_overrides(rng, space)
            cfg = _apply_overrides(base, overrides)
            sha = candidate_config_sha256(cfg)
            if sha in seen:
                continue
            seen.add(sha)
            candidate = reg.get_or_create_candidate(
                workflow_id,
                "search",
                cfg,
                parent_candidate_id=baseline_id,
                overrides=overrides,
                notes="SEARCH sampled candidate",
            )
            candidates.append(str(candidate["candidate_id"]))

    print(
        "SEARCH workflow %s | %d candidates | %d epochs each | screen member 0"
        % (workflow_id, len(candidates), int(base["training"]["total_epochs"])),
        flush=True,
    )
    for i, candidate_id in enumerate(candidates):
        candidate = reg.get_candidate(candidate_id)
        print(
            "SEARCH %d/%d candidate=%s overrides=%s"
            % (i + 1, len(candidates), candidate_id, candidate["overrides_json"]),
            flush=True,
        )
        try:
            run_id = _ensure_member(
                reg, candidate_id, 0, "SEARCH screening member 0"
            )
            _log_screening_score(reg, candidate_id, run_id)
            reg.finish_candidate(candidate_id, "completed")
        except KeyboardInterrupt:
            reg.export_candidates(str(root / "candidates.csv"), workflow_id, "search")
            raise
        except Exception as exc:
            reg.finish_candidate(candidate_id, "failed")
            print("SEARCH FAILED candidate=%s error=%r" % (candidate_id, exc), flush=True)
            if args.fail_fast:
                reg.export_candidates(str(root / "candidates.csv"), workflow_id, "search")
                raise
        reg.export_candidates(str(root / "candidates.csv"), workflow_id, "search")

    print(str(root / "candidates.csv"))


def _screening_rows(reg: RunRegistry, workflow_id: str) -> List[Dict[str, Any]]:
    """Return candidates with a recorded screening validation score.

    Candidate status can later be ``failed`` after a replication attempt; the
    already completed screening result remains valid for promotion.
    """
    with reg.connect() as con:
        rows = con.execute(
            """
            SELECT
                c.candidate_id,
                c.created_at_utc,
                MAX(CASE WHEN m.evaluation_kind='screening' AND m.split='val'
                              AND m.metric='corr' AND m.scope='mean'
                         THEN m.value END) AS val_corr_mean,
                MAX(CASE WHEN m.evaluation_kind='screening' AND m.split='val'
                              AND m.metric='corr' AND m.scope='median'
                         THEN m.value END) AS val_corr_median
            FROM candidates c
            LEFT JOIN candidate_metrics m ON m.candidate_id=c.candidate_id
            WHERE c.workflow_id=? AND c.stage='search'
            GROUP BY c.candidate_id
            ORDER BY c.created_at_utc ASC
            """,
            (workflow_id,),
        ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        if item["val_corr_mean"] is None or item["val_corr_median"] is None:
            continue
        item["val_corr_mean"] = float(item["val_corr_mean"])
        item["val_corr_median"] = float(item["val_corr_median"])
        out.append(item)
    return out


def _promotion_score(row: Dict[str, Any], criterion: str, mean_weight: float) -> float:
    if criterion == "mean":
        return float(row["val_corr_mean"])
    if criterion == "mean_median":
        w = float(mean_weight)
        return w * float(row["val_corr_mean"]) + (1.0 - w) * float(row["val_corr_median"])
    raise ValueError(criterion)


def run_promote(args) -> None:
    """Promote top-k screened candidates plus explicit controls, using validation.

    ``mean`` ranks only by mean validation correlation. ``mean_median`` uses a
    weighted average of mean and median validation correlation, providing a
    simple robustness/fit-quality term without touching the test set.
    """
    root = Path(args.output_root).resolve()
    reg = _registry(str(root))
    reg.initialize()
    workflow_id = args.workflow_id or _load_workflow_id(str(root))
    with reg.connect() as con:
        tested = con.execute(
            "SELECT 1 FROM candidate_metrics m JOIN candidates c "
            "ON c.candidate_id=m.candidate_id "
            "WHERE c.workflow_id=? AND m.split='test' LIMIT 1",
            (workflow_id,),
        ).fetchone()
    if tested is not None:
        raise ValueError("Test already evaluated; cannot change the shortlist")
    rows = _screening_rows(reg, workflow_id)
    if not rows:
        raise ValueError("No completed SEARCH screening candidates found")

    top_k = int(args.top_k)
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if top_k > len(rows):
        raise ValueError(
            "Requested top_k=%d but only %d completed screening candidates exist"
            % (top_k, len(rows))
        )
    mean_weight = float(args.mean_weight)
    if not 0.0 <= mean_weight <= 1.0:
        raise ValueError("mean_weight must be between 0 and 1")

    ranked = []
    for row in rows:
        item = dict(row)
        item["promotion_score"] = _promotion_score(
            item, str(args.criterion), mean_weight
        )
        ranked.append(item)
    ranked.sort(
        key=lambda x: (
            float(x["promotion_score"]),
            float(x["val_corr_mean"]),
            float(x["val_corr_median"]),
        ),
        reverse=True,
    )
    include_ids = list(dict.fromkeys(getattr(args, "include_candidate_id", []) or []))
    by_id = {str(row["candidate_id"]): row for row in ranked}
    unknown = [cid for cid in include_ids if cid not in by_id]
    if unknown:
        raise ValueError("Candidate IDs lack completed screening in this workflow: %s" % unknown)
    selected = list(ranked[:top_k])
    selected_ids = {str(row["candidate_id"]) for row in selected}
    for cid in include_ids:
        if cid not in selected_ids:
            selected.append(by_id[cid])
            selected_ids.add(cid)

    # Promotion is a replaceable shortlist, not an accumulating flag.
    for candidate in reg.list_candidates(workflow_id=workflow_id, stage="search"):
        reg.set_candidate_promoted(
            str(candidate["candidate_id"]),
            str(candidate["candidate_id"]) in selected_ids,
        )

    report = {
        "workflow_id": workflow_id,
        "criterion": str(args.criterion),
        "mean_weight": mean_weight if args.criterion == "mean_median" else None,
        "top_k": top_k,
        "include_candidate_ids": include_ids,
        "selected": selected,
    }
    report_path = root / "promotion.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    reg.export_candidates(str(root / "candidates.csv"), workflow_id, "search")

    print(
        "Promoted %d/%d SEARCH candidates by %s (top %d plus explicit IDs)"
        % (len(selected), len(ranked), args.criterion, top_k)
    )
    for rank, row in enumerate(selected, 1):
        print(
            "%2d  %s  score=%.6f  mean=%.6f  median=%.6f"
            % (
                rank,
                row["candidate_id"],
                row["promotion_score"],
                row["val_corr_mean"],
                row["val_corr_median"],
            )
        )
    print(str(report_path))


def _replication_rows(reg: RunRegistry, workflow_id: str, members: int) -> List[Dict[str, Any]]:
    """Validation scores of the first `members` seeds for each promoted candidate."""
    rows = []
    for candidate in reg.list_candidates(
        workflow_id=workflow_id, stage="search", promoted=True
    ):
        cid = str(candidate["candidate_id"])
        runs = {
            int(run["member_index"]): run
            for run in reg.candidate_runs(cid, status="completed")
        }
        if any(i not in runs for i in range(members)):
            continue
        best = [reg.best_validation(str(runs[i]["run_id"])) for i in range(members)]
        rows.append({
            "candidate_id": cid,
            "members": members,
            "val_corr_mean": float(np.mean([float(b["val_corr_mean"]) for b in best])),
            "val_corr_median": float(np.mean([float(b["val_corr_median"]) for b in best])),
            "member_run_ids": [str(runs[i]["run_id"]) for i in range(members)],
        })
    rows.sort(key=lambda row: (row["val_corr_mean"], row["val_corr_median"]), reverse=True)
    return rows


def run_replicate(args) -> None:
    root = Path(args.output_root).resolve()
    reg = _registry(str(root))
    reg.initialize()
    workflow_id = args.workflow_id or _load_workflow_id(str(root))
    candidates = reg.list_candidates(workflow_id=workflow_id, stage="search", promoted=True)
    if not candidates:
        raise ValueError("No promoted SEARCH candidates found; run search promote first")
    members = int(args.members)
    if members < 2:
        raise ValueError("Replication requires at least 2 members")
    for candidate in candidates:
        cfg = reg.get_candidate_config(str(candidate["candidate_id"]))
        if members > int(cfg["training"]["ensemble_members"]):
            raise ValueError("--members exceeds ensemble_members for %s" % candidate["candidate_id"])

    for i, candidate in enumerate(candidates, 1):
        cid = str(candidate["candidate_id"])
        print("REPLICATE %d/%d candidate=%s members=%d" % (
            i, len(candidates), cid, members
        ), flush=True)
        try:
            for index in range(members):
                _ensure_member(reg, cid, index, "SEARCH validation replication")
            reg.clear_candidate_metrics(cid, evaluation_kind="replication", split="val")
            row = next(row for row in _replication_rows(reg, workflow_id, members)
                       if row["candidate_id"] == cid)
            for scope in ("mean", "median"):
                reg.log_candidate_metric(
                    cid, "replication", "val", "corr", scope,
                    row["val_corr_" + scope],
                )
            reg.finish_candidate(cid, "completed")
            print("REPLICATE val_mean=%.6f" % row["val_corr_mean"], flush=True)
        except KeyboardInterrupt:
            reg.export_candidates(str(root / "candidates.csv"), workflow_id, "search")
            raise
        except Exception as exc:
            reg.finish_candidate(cid, "failed")
            print("REPLICATE FAILED candidate=%s error=%r" % (cid, exc), flush=True)
            if args.fail_fast:
                reg.export_candidates(str(root / "candidates.csv"), workflow_id, "search")
                raise
        reg.export_candidates(str(root / "candidates.csv"), workflow_id, "search")

    ranked = _replication_rows(reg, workflow_id, members)
    report_path = root / "replication.json"
    report_path.write_text(json.dumps({
        "workflow_id": workflow_id, "members": members, "ranking": ranked,
    }, indent=2, sort_keys=True) + "\n")
    print(str(report_path))


def run_ensemble(args) -> None:
    root = Path(args.output_root).resolve()
    reg = _registry(str(root))
    reg.initialize()
    workflow_id = args.workflow_id or _load_workflow_id(str(root))
    ids = [row["candidate_id"] for row in reg.list_candidates(
        workflow_id=workflow_id, stage="search", promoted=True
    )]
    if not ids:
        raise ValueError("No promoted candidates; run search promote first")

    for i, candidate_id in enumerate(ids):
        candidate = reg.get_candidate(candidate_id)
        if candidate["workflow_id"] != workflow_id or candidate["stage"] != "search":
            raise ValueError("Candidate set must belong to one SEARCH workflow")
        cfg = reg.get_candidate_config(candidate_id)
        n = int(cfg["training"]["ensemble_members"])
        print(
            "ENSEMBLE %d/%d candidate=%s members=%d"
            % (i + 1, len(ids), candidate_id, n),
            flush=True,
        )
        try:
            already_tested = candidate_id in {
                row["candidate_id"] for row in _ensemble_test_rows(reg, workflow_id)
            }
            if already_tested:
                print("TEST already evaluated for candidate=%s; skipping" % candidate_id)
            else:
                for member_index in range(n):
                    _ensure_member(
                        reg, candidate_id, member_index,
                        "SEARCH promoted ensemble member",
                    )
                summary = evaluate_candidate_ensemble(reg, str(root), candidate_id)
                print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        except KeyboardInterrupt:
            reg.export_candidates(str(root / "candidates.csv"), workflow_id, "search")
            raise
        except Exception as exc:
            reg.finish_candidate(candidate_id, "failed")
            print("ENSEMBLE FAILED candidate=%s error=%r" % (candidate_id, exc), flush=True)
            if args.fail_fast:
                reg.export_candidates(str(root / "candidates.csv"), workflow_id, "search")
                raise
        reg.export_candidates(str(root / "candidates.csv"), workflow_id, "search")

    ranked = _ensemble_test_rows(reg, workflow_id)
    report_path = root / "ensemble_ranking.json"
    report_path.write_text(json.dumps({
        "workflow_id": workflow_id,
        "criterion": "ensemble_test_corr_to_average_mean",
        "ranking": ranked,
    }, indent=2, sort_keys=True) + "\n")
    for rank, row in enumerate(ranked, 1):
        print("%2d  %s  test_corr=%.6f" % (
            rank, row["candidate_id"], row["test_corr_mean"]
        ))
    print(str(report_path))
    print(str(root / "candidates.csv"))


def _ensemble_test_rows(reg: RunRegistry, workflow_id: str) -> List[Dict[str, Any]]:
    """Fully evaluated promoted ensembles, ordered by test corr-to-average."""
    rows = []
    for candidate in reg.list_candidates(
        workflow_id=workflow_id, stage="search", promoted=True
    ):
        cid = str(candidate["candidate_id"])
        cfg = reg.get_candidate_config(cid)
        expected_members = int(cfg["training"]["ensemble_members"])
        expected_neurons = int(cfg["model"]["readout"]["outdims"])
        completed = {int(run["member_index"]) for run in reg.candidate_runs(
            cid, status="completed"
        )}
        if not set(range(expected_members)).issubset(completed):
            continue
        with reg.connect() as con:
            score = con.execute(
                "SELECT value FROM candidate_metrics WHERE candidate_id=? "
                "AND evaluation_kind='ensemble' AND split='test' "
                "AND metric='corr_to_average' AND scope='mean' "
                "ORDER BY metric_id DESC LIMIT 1", (cid,),
            ).fetchone()
            count = con.execute(
                "SELECT COUNT(*) AS n FROM candidate_metrics WHERE candidate_id=? "
                "AND evaluation_kind='ensemble' AND split='test' "
                "AND metric='corr_to_average' AND scope='neuron'", (cid,),
            ).fetchone()["n"]
        if score is not None and score["value"] is not None and int(count) == expected_neurons:
            value = float(score["value"])
            if math.isfinite(value):
                rows.append({"candidate_id": cid, "test_corr_mean": value})
    rows.sort(key=lambda row: row["test_corr_mean"], reverse=True)
    return rows


def run_select(args) -> None:
    root = Path(args.output_root).resolve()
    reg = _registry(str(root))
    reg.initialize()
    workflow_id = _load_workflow_id(str(root))
    promoted = reg.list_candidates(workflow_id=workflow_id, stage="search", promoted=True)
    if any(int(reg.get_candidate_config(str(row["candidate_id"]))["training"]["ensemble_members"])
           != int(args.members) for row in promoted):
        raise ValueError("--members must match the ensemble size of every promoted candidate")
    ranked = _ensemble_test_rows(reg, workflow_id)
    if len(ranked) != len(promoted):
        raise ValueError("Test evaluation has not completed for every promoted ensemble")
    if not ranked:
        raise ValueError("No promoted ensembles with completed test evaluation")
    candidate_id = ranked[0]["candidate_id"] if args.best else args.candidate_id
    candidate = reg.get_candidate(candidate_id)
    if candidate["workflow_id"] != workflow_id:
        raise ValueError("Candidate is not part of this SEARCH workflow")
    if candidate["stage"] != "search":
        raise ValueError("Founder must be a SEARCH candidate")
    if not bool(candidate["is_promoted"]):
        raise ValueError("Founder must be chosen from the promoted SEARCH shortlist")
    cfg = reg.get_candidate_config(candidate_id)
    if candidate_id not in {row["candidate_id"] for row in ranked}:
        raise ValueError("Candidate has no complete ensemble test evaluation")
    reg.set_candidate_selected(candidate_id, True)
    founder_path = root / "founder_config.yaml"
    save_resolved_config(cfg, str(founder_path))
    (root / "selection.json").write_text(json.dumps({
        "workflow_id": workflow_id,
        "criterion": "ensemble_test_corr_to_average_mean",
        "selected_candidate_id": candidate_id,
        "ranking": ranked,
    }, indent=2, sort_keys=True) + "\n")
    reg.export_candidates(
        str(root / "candidates.csv"), candidate["workflow_id"], "search"
    )
    print("Selected founder %s by ensemble test corr-to-average (mean=%.6f)" % (
        candidate_id,
        next(row["test_corr_mean"] for row in ranked if row["candidate_id"] == candidate_id),
    ))
    print(str(founder_path))


def run_export(args) -> None:
    root = Path(args.output_root).resolve()
    reg = _registry(str(root))
    reg.initialize()
    workflow_id = args.workflow_id or _load_workflow_id(str(root))
    output = Path(args.output) if args.output else root / "candidates.csv"
    reg.export_candidates(str(output), workflow_id, "search")
    print(str(output))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sinz-v1-search")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("screen")
    p.add_argument("--config", required=True)
    p.add_argument("--space", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--trials", type=int, default=100,
                   help="total candidates including the unchanged baseline")
    p.add_argument("--epochs", type=int, default=None,
                   help="override total_epochs for every candidate")
    p.add_argument("--ensemble-members", type=int, default=5)
    p.add_argument("--member-seed", type=int, default=8000,
                   help="member seeds are member_seed+i and are shared across candidates")
    p.add_argument("--search-seed", type=int, default=107173,
                   help="deterministic seed for hyperparameter sampling only")
    p.add_argument("--workflow-id", default=None)
    p.add_argument("--workflow-name", default="reproduction calibration")
    p.add_argument("--fail-fast", action="store_true")
    p.set_defaults(func=run_screen)

    p = sub.add_parser("promote")
    p.add_argument("--output-root", required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--include-candidate-id", action="append", default=[],
                   help="also promote a screened candidate by ID; repeat for controls")
    p.add_argument("--criterion", choices=("mean", "mean_median"), default="mean")
    p.add_argument(
        "--mean-weight", type=float, default=0.5,
        help="weight of mean validation r for mean_median; median gets 1-weight",
    )
    p.add_argument("--workflow-id", default=None)
    p.set_defaults(func=run_promote)

    p = sub.add_parser("replicate")
    p.add_argument("--output-root", required=True)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--workflow-id", default=None)
    p.add_argument("--fail-fast", action="store_true")
    p.set_defaults(func=run_replicate)

    p = sub.add_parser("ensemble")
    p.add_argument("--output-root", required=True)
    p.add_argument("--workflow-id", default=None)
    p.add_argument("--fail-fast", action="store_true")
    p.set_defaults(func=run_ensemble)

    p = sub.add_parser("select")
    p.add_argument("--output-root", required=True)
    select = p.add_mutually_exclusive_group(required=True)
    select.add_argument("--candidate-id")
    select.add_argument("--best", action="store_true", help="select highest ensemble test correlation")
    p.add_argument("--members", type=int, default=5,
                   help="required ensemble size; all members must be evaluated")
    p.set_defaults(func=run_select)

    p = sub.add_parser("export")
    p.add_argument("--output-root", required=True)
    p.add_argument("--workflow-id", default=None)
    p.add_argument("--output", default=None)
    p.set_defaults(func=run_export)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
