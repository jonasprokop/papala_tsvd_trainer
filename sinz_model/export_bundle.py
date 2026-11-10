"""Export one evaluated ensemble as a self-contained inference handoff."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import uuid
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_bundle(sqlite_path: str, candidate_id: str, output_dir: str) -> Path:
    """Copy five best checkpoints, model source, and their resolved metadata.

    The resulting directory can be copied between containers without the
    trainer's database, data, or absolute run paths.
    """
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    connection = sqlite3.connect(f"file:{Path(sqlite_path).resolve()}?mode=ro", uri=True)
    try:
        candidate = connection.execute(
            "SELECT config_json, config_sha256, workflow_id FROM candidates WHERE candidate_id=?",
            (candidate_id,),
        ).fetchone()
        if candidate is None:
            raise ValueError(f"Unknown candidate {candidate_id}")
        cfg = json.loads(candidate[0])
        if int(cfg["training"]["ensemble_members"]) != 5:
            raise ValueError("Inference adapter requires exactly five ensemble members")
        runs = connection.execute(
            "SELECT member_index, run_id, run_dir, status, session_mapping_json "
            "FROM runs WHERE candidate_id=? ORDER BY member_index", (candidate_id,),
        ).fetchall()
        if [row[0] for row in runs] != list(range(5)) or any(row[3] != "completed" for row in runs):
            raise ValueError("Candidate must have five completed members indexed 0..4")
        mappings = [json.loads(row[4]) for row in runs]
        order_hashes = {entry[0]["session_order_sha256"] for entry in mappings}
        if (len(order_hashes) != 1 or any(entry != mappings[0] for entry in mappings)
                or len(mappings[0]) != 32 or mappings[0][-1]["global_end"] != 458):
            raise ValueError("Incompatible 458-neuron session mappings")
        metric_rows = connection.execute(
            "SELECT neuron_id, value FROM candidate_metrics WHERE candidate_id=? "
            "AND evaluation_kind='ensemble' AND split='test' "
            "AND metric='corr_to_average' AND scope='neuron' ORDER BY neuron_id",
            (candidate_id,),
        ).fetchall()
        if [row[0] for row in metric_rows] != list(range(458)) or any(row[1] is None for row in metric_rows):
            raise ValueError("Candidate needs a complete 458-neuron ensemble evaluation")
        summary = connection.execute(
            "SELECT scope, value FROM candidate_metrics WHERE candidate_id=? "
            "AND evaluation_kind='ensemble' AND split='test' "
            "AND metric='corr_to_average' AND scope IN ('mean','median')",
            (candidate_id,),
        ).fetchall()
        if {scope for scope, _ in summary} != {"mean", "median"}:
            raise ValueError("Candidate needs ensemble mean and median evaluation")
    finally:
        connection.close()

    normalizations = []
    for _, _, run_dir, _, _ in runs:
        with (Path(run_dir) / "normalization.json").open() as stream:
            stats = json.load(stream)
        normalizations.append((float(stats["mean"]), float(stats["std"])))
    if len(set(normalizations)) != 1 or normalizations[0][1] <= 0:
        raise ValueError("Members have different or invalid image normalization")
    mean, std = normalizations[0]
    model_source = Path(__file__).with_name("model.py")
    spec = {
        "loader": "sinz_ensemble", "n_neurons": 458,
        "input": {"mean": mean, "std": std},
        "correlations": [float(row[1]) for row in metric_rows],
        "trainer": {
            "candidate_id": candidate_id, "workflow_id": candidate[2],
            "candidate_config_sha256": candidate[1],
            "session_order_sha256": next(iter(order_hashes)),
            "member_run_ids": [row[1] for row in runs],
            "ensemble_test_corr_to_average": dict(summary),
            "training_config": cfg,
            "model_code_sha256": _sha256(model_source),
        },
    }
    staging = destination.with_name(destination.name + ".tmp-" + uuid.uuid4().hex)
    try:
        (staging / "checkpoints").mkdir(parents=True)
        checkpoints = []
        for index, run_id, run_dir, _, _ in runs:
            source = Path(run_dir) / "checkpoints" / "best.pt"
            if not source.is_file():
                raise FileNotFoundError(f"Best checkpoint missing for {run_id}: {source}")
            filename = f"member_{index}_best.pt"
            copied = staging / "checkpoints" / filename
            shutil.copyfile(source, copied)
            checkpoints.append({
                "member_index": index, "run_id": run_id, "filename": filename,
                "sha256": _sha256(copied), "size_bytes": copied.stat().st_size,
            })
        shutil.copyfile(model_source, staging / "model.py")
        (staging / "manifest.json").write_text(json.dumps({
            "format": "sinz_v1_ensemble_bundle_v1", "model": spec,
            "checkpoints": checkpoints,
        }, indent=2, sort_keys=True) + "\n")
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
