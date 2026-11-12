"""Single-writer, staged Optuna optimisation over independent PBS GPU jobs.

Do not import the GPU trainer here. The coordinator operates without CUDA, and
never gives compute-node workers access to the Optuna storage.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sqlite3
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import optuna
import yaml
from optuna.trial import TrialState

OPTUNA_VERSION = optuna.__version__
RESERVED = {"training.member_seeds", "training.ensemble_members"}


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _resolve(data: dict, dotted: str) -> Any:
    if dotted.startswith("project.") or dotted.startswith("data.") or dotted in RESERVED:
        raise ValueError(f"Forbidden optimisation parameter: {dotted}")
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"Unknown parameter in base config: {dotted}")
        node = node[part]
    return node


def _override(data: dict, dotted: str, value: Any) -> None:
    _resolve(data, dotted)
    node = data
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def _space(path: Path, base: dict) -> dict:
    raw = yaml.safe_load(path.read_text())
    specs = raw.get("search") if isinstance(raw, dict) else None
    if not isinstance(specs, dict) or not specs:
        raise ValueError("Space must contain a non-empty 'search:' mapping")
    for name, spec in specs.items():
        base_value = _resolve(base, name)
        if not isinstance(spec, dict):
            raise ValueError(f"Expected mapping for {name}")
        kind = spec.get("type")
        if kind == "choice":
            values = spec.get("values")
            if not isinstance(values, list) or not values or any(isinstance(v, (list, dict)) for v in values):
                raise ValueError(f"Invalid categorical choices: {name}")
            if isinstance(base_value, bool) or isinstance(base_value, str):
                if any(type(v) is not type(base_value) for v in values):
                    raise ValueError(f"Categorical value type mismatch: {name}")
        elif kind in ("loguniform", "uniform"):
            if isinstance(base_value, bool) or not isinstance(base_value, (int, float)):
                raise ValueError(f"Numeric parameter required: {name}")
            low, high = float(spec["low"]), float(spec["high"])
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError(f"Invalid range for {name}")
            if kind == "loguniform" and low <= 0:
                raise ValueError(f"Nonpositive log range for {name}")
        elif kind == "randint":
            if isinstance(base_value, bool) or not isinstance(base_value, int):
                raise ValueError(f"Integer base parameter required: {name}")
            if not isinstance(spec.get("low"), int) or not isinstance(spec.get("high"), int) or spec["high"] <= spec["low"]:
                raise ValueError(f"Invalid integer range for {name}")
        else:
            raise ValueError(f"Unsupported distribution {kind} for {name}")
    return specs


def _sample(trial: optuna.trial.Trial, specs: dict) -> dict:
    out = {}
    for name, spec in sorted(specs.items()):
        kind = spec["type"]
        if kind == "choice":
            value = trial.suggest_categorical(name, spec["values"])
        elif kind == "randint":
            value = trial.suggest_int(name, spec["low"], spec["high"])
        else:
            value = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=(kind == "loguniform"))
        out[name] = value
    return out


@contextmanager
def _lock(root: Path):
    lock = root / ".coordinator.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Coordinator lock exists ({lock}); inspect before removing") from exc
    try:
        with os.fdopen(fd, "w") as f:
            f.write(f"pid={os.getpid()}\n")
        yield
    finally:
        lock.unlink(missing_ok=True)


@contextmanager
def _study(root: Path):
    """Operate on local SQLite; atomically checkpoint a coherent snapshot to Brno2.

    Optuna's SQLite database is NEVER opened for live writes on NFS.
    Only one coordinator can own it at a time. GPU jobs never touch this DB.
    """
    db = root / "study.sqlite"
    with tempfile.TemporaryDirectory(prefix="tvsd-optuna-") as tmpdir:
        local = Path(tmpdir) / "study.sqlite"
        if db.exists():
            shutil.copy2(db, local)
        url = "sqlite:///" + str(local)
        meta = _read(root / "experiment.json")
        sampler = optuna.samplers.TPESampler(seed=int(meta["seed"]), constant_liar=True)
        storage = optuna.storages.RDBStorage(url=url)
        study = optuna.create_study(study_name=meta["name"], direction="maximize", storage=storage,
                                    sampler=sampler, load_if_exists=True)
        # Optuna persists trials, not sampler RNG state. Vary the deterministic
        # seed by the number of existing trials on each batch restart.
        existing_count = len(study.get_trials(deepcopy=False))
        if existing_count:
            study.sampler = optuna.samplers.TPESampler(seed=int(meta["seed"]) + existing_count,
                                                     constant_liar=True)
        try:
            yield study
        finally:
            # Close Optuna connections before backing up and removing local scratch.
            # The on-disk study remains valid on exceptions.
            storage.engine.dispose()
            if local.exists():
                backup_tmp = db.with_name(db.name + f".tmp.{os.getpid()}")
                try:
                    with sqlite3.connect(local) as source, sqlite3.connect(backup_tmp) as dest:
                        source.backup(dest)
                    os.replace(backup_tmp, db)
                finally:
                    backup_tmp.unlink(missing_ok=True)


def _experiment(root: Path) -> dict:
    path = root / "experiment.json"
    if not path.exists():
        raise FileNotFoundError(f"Initialize experiment first: {path}")
    meta = _read(path)
    if meta["schema"] != 1:
        raise ValueError("Unknown experiment schema")
    if meta["optuna_version"] != OPTUNA_VERSION:
        raise RuntimeError("Optuna version changed; use the version recorded at experiment initialization")
    return meta


def initialize(args) -> None:
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _lock(root):
        if (root / "experiment.json").exists():
            raise FileExistsError("Experiment exists; use prepare / collect / status")
        base = yaml.safe_load(Path(args.config).read_text())
        specs = _space(Path(args.space), base)
        if args.epochs is not None:
            epochs = int(args.epochs)
            if epochs < int(base["training"]["readout_only_epochs"]) + 1:
                raise ValueError("Epoch budget must include at least one joint-training epoch")
            base["training"]["total_epochs"] = epochs
        base["training"]["ensemble_members"] = 1
        base["training"]["member_seeds"] = [int(args.member_seed)]
        meta = {"schema": 1, "name": args.name, "seed": int(args.seed),
                "base_digest": _digest(base), "space_digest": _digest(specs),
                "metric": "best_validation_mean_pearson", "optuna_version": OPTUNA_VERSION}
        _write(root / "base.json", base)
        _write(root / "space.json", specs)
        _write(root / "experiment.json", meta)
        with _study(root):
            pass
        print(f"Initialized {root}; manual training remains available")


def _container_path(host_path: Path, workspace: Path) -> str:
    """Map persistent Brno2 paths to the proven /mnt/tvsd Singularity bind."""
    p = Path(host_path)
    if p.is_relative_to(Path("/mnt/tvsd")):
        return str(p)
    try:
        rel = p.resolve().relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError(f"Only paths within workspace {workspace} are available to PBS workers: {p}") from exc
    return str(Path("/mnt/tvsd") / rel)


def _active_batch(root: Path) -> Path | None:
    for path in sorted((root / "batches").glob("*/manifest.json")):
        m = _read(path)
        if not m.get("collected", False):
            return path
    return None


def prepare(args) -> None:
    root = Path(args.root).resolve()
    _experiment(root)
    n = int(args.trials)
    if n <= 0 or n > 1000:
        raise ValueError("trials must be 1..1000")
    with _lock(root):
        if _active_batch(root):
            raise RuntimeError("An uncollected batch exists; run collect before preparing another")
        base, specs = _read(root / "base.json"), _read(root / "space.json")
        # Expect <workspace>/optuna/<study>; all worker-visible paths use the
        # established Singularity bind <workspace>:/mnt/tvsd.
        workspace = root.parents[1]
        if root.parent.name != "optuna":
            raise ValueError("Place experiment under <workspace>/optuna/<study>")
        if _digest(base) != _experiment(root)["base_digest"] or _digest(specs) != _experiment(root)["space_digest"]:
            raise RuntimeError("Frozen experiment configuration changed")
        with _study(root) as study:
            if any(t.state == TrialState.RUNNING for t in study.trials):
                raise RuntimeError("Unaccounted RUNNING trials exist; do not sample another batch")
            batch_index = len(list((root / "batches").glob("*/manifest.json"))) + 1
            batch = root / "batches" / f"batch-{batch_index:03d}"
            if batch.exists():
                raise FileExistsError(batch)
            batch.mkdir(parents=True)
            records = []
            for i in range(1, n + 1):
                trial = study.ask()
                values = _sample(trial, specs)
                trial_dir = root / "trials" / f"trial-{trial.number:05d}"
                cfg = copy.deepcopy(base)
                for key, value in values.items():
                    _override(cfg, key, value)
                cfg["project"]["name"] = f"{base['project']['name']}-optuna-{trial.number}"
                container_trial = Path(_container_path(trial_dir, workspace))
                cfg["project"]["output_root"] = str(container_trial / "runs")
                cfg["project"]["sqlite_path"] = str(container_trial / "results.sqlite")
                cfg["project"]["result_sheet_path"] = str(container_trial / "runs.csv")
                # Repo is mounted at /project; data/results under /mnt/tvsd.
                cfg["data"]["root"] = _container_path(Path(args.data_root or workspace / "data" / "tvsd_waveform_aware_v1" / "model_dataset"), workspace)
                cfg["data"]["image_root"] = _container_path(Path(args.image_root or workspace / "data" / "assets" / "THINGS"), workspace)
                cfg["data"]["preprocessing"]["normalization_path"] = str(Path(cfg["data"]["root"]) / "normalization.json")
                cfg["data"]["dataset_module_path"] = "/project/waveform_aware_tvsd_preprocessing/scripts/dataset.py"
                cfg["training"]["progress"]["enabled"] = False
                _write(trial_dir / "config.json", cfg)
                _write(trial_dir / "params.json", values)
                records.append({"index": i, "number": trial.number, "trial_dir": str(trial_dir),
                                "container_trial_dir": str(container_trial), "params": values,
                                "config_sha256": _digest(cfg)})
            # Commit study before publishing manifest. A crash in between is explicitly detected.
        _write(batch / "manifest.json", {"schema": 1, "batch": batch_index, "records": records, "collected": False,
                                          "pbs_job_id": None})
        print(f"Prepared {len(records)} trials in {batch}")
        print(f"Submit with: python -m sinz_model.optuna_batch submit --root {root}")


def submit(args) -> None:
    root = Path(args.root).resolve()
    _experiment(root)
    with _lock(root):
        manifest_path = _active_batch(root)
        if manifest_path is None:
            raise RuntimeError("No active batch")
        manifest = _read(manifest_path)
        if manifest["pbs_job_id"] or manifest.get("submission_pending"):
            raise RuntimeError(f"Already submitted or submission uncertain: {manifest.get('pbs_job_id')}; inspect qstat")
        n = len(manifest["records"])
        cap = int(args.concurrency)
        if cap <= 0 or cap > n:
            raise ValueError("concurrency must be in 1..number of trials")
        script = Path(__file__).resolve().parents[1] / "jobs" / "optuna_trial.pbs"
        if not script.exists():
            raise FileNotFoundError(script)
        # qsub -v values cannot contain commas; path characters limited for shell safety.
        if "," in str(root) or " " in str(root):
            raise ValueError("Experiment directory must not contain commas or spaces")
        cmd = ["qsub", "-J", f"1-{n}%{cap}", "-v", f"OPTUNA_EXPERIMENT={root},OPTUNA_BATCH={manifest_path.parent.name}", str(script)]
        if args.dry_run:
            print(" ".join(cmd))
            return
        # Crash safety: do not automatically resubmit if qsub succeeded but the
        # coordinator died before storing its returned job ID.
        manifest["submission_pending"] = True
        _write(manifest_path, manifest)
        try:
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError:
            # qsub explicitly rejected the job; it was not scheduled.
            manifest["submission_pending"] = False
            _write(manifest_path, manifest)
            raise
        job_id = result.stdout.strip()
        if not job_id:
            raise RuntimeError(f"No PBS job ID returned: {result.stderr}")
        manifest["pbs_job_id"] = job_id
        manifest["submission_pending"] = False
        _write(manifest_path, manifest)
        print(f"PBS array: {job_id} | {n} trials | concurrency={cap}")


def collect(args) -> None:
    root = Path(args.root).resolve()
    _experiment(root)
    with _lock(root):
        path = _active_batch(root)
        if path is None:
            raise RuntimeError("No uncollected batch")
        manifest = _read(path)
        successes = failed = pending = 0
        with _study(root) as study:
            for record in manifest["records"]:
                num = record["number"]
                trial = study.trials[num]
                if trial.state.is_finished():
                    continue
                result_path = Path(record["trial_dir"]) / "result.json"
                if not result_path.exists():
                    if args.fail_missing:
                        study.tell(num, state=TrialState.FAIL)
                        failed += 1
                    else:
                        pending += 1
                    continue
                result = _read(result_path)
                if result.get("trial_number") != num or result.get("config_sha256") != record["config_sha256"]:
                    raise ValueError(f"Trial result mismatch: {result_path}")
                if result.get("status") == "completed":
                    value = float(result["objective"])
                    if not math.isfinite(value):
                        raise ValueError(f"Nonfinite objective: {num}")
                    study.tell(num, value)
                    successes += 1
                elif result.get("status") == "failed":
                    study.tell(num, state=TrialState.FAIL)
                    failed += 1
                else:
                    raise ValueError(f"Unknown result status for {num}")
        # Mark collected only when all trials have been resolved.
        if pending == 0:
            manifest["collected"] = True
            _write(path, manifest)
        print(f"Collected completed={successes}, failed={failed}, pending={pending}")
        if pending:
            print("Missing results remain RUNNING. Rerun collect after jobs finish, or inspect then --fail-missing.")


def status(args) -> None:
    root = Path(args.root).resolve()
    _experiment(root)
    with _lock(root):
        with _study(root) as study:
            counts = {state.name: sum(t.state == state for t in study.trials) for state in TrialState}
            print(json.dumps({"study": study.study_name, "trials": len(study.trials), "states": counts,
                              "best": ({"number": study.best_trial.number, "value": study.best_value,
                                        "params": study.best_trial.params} if any(t.state == TrialState.COMPLETE for t in study.trials) else None)}, indent=2))
        active = _active_batch(root)
        if active:
            print(f"Pending batch: {active.parent.name}; PBS: {_read(active).get('pbs_job_id')}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init")
    p.add_argument("--root", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--space", required=True)
    p.add_argument("--name", default="tvsd-optuna")
    p.add_argument("--seed", type=int, default=107173)
    p.add_argument("--member-seed", type=int, default=8000)
    p.add_argument("--epochs", type=int, default=None, help="fixed training budget for every trial")
    p.set_defaults(func=initialize)
    p = sub.add_parser("prepare")
    p.add_argument("--root", required=True)
    p.add_argument("--trials", type=int, required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--image-root", default=None)
    p.set_defaults(func=prepare)
    p = sub.add_parser("submit")
    p.add_argument("--root", required=True)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=submit)
    p = sub.add_parser("collect")
    p.add_argument("--root", required=True)
    p.add_argument("--fail-missing", action="store_true", help="Only after verifying PBS jobs are no longer active")
    p.set_defaults(func=collect)
    p = sub.add_parser("status")
    p.add_argument("--root", required=True)
    p.set_defaults(func=status)
    return parser


def main():
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
