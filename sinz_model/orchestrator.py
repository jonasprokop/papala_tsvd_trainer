from __future__ import annotations

import copy
import csv
import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .runtime import canonical_json, config_sha256, save_resolved_config
from .runtime import capture_environment, capture_git_state, sha256_file


SCHEMA_VERSION = 4

# Candidate identity intentionally excludes execution bookkeeping.  The full
# resolved config is still stored on both the candidate and every run.
_EXECUTION_ONLY_CONFIG_PATHS = (
    ("project", "output_root"),
    ("project", "sqlite_path"),
    ("project", "result_sheet_path"),
    ("training", "member_seeds"),
    ("training", "ensemble_members"),
)
_EXECUTION_ONLY_TOP_LEVEL_KEYS = ("search", "sweep")


SCHEMA = r"""
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS workflows (
    workflow_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    finished_at_utc TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    parent_candidate_id TEXT,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    finished_at_utc TEXT,
    config_sha256 TEXT NOT NULL,
    config_json TEXT NOT NULL,
    identity_config_json TEXT NOT NULL,
    overrides_json TEXT NOT NULL,
    is_promoted INTEGER NOT NULL DEFAULT 0,
    is_selected INTEGER NOT NULL DEFAULT 0,
    notes TEXT,
    FOREIGN KEY(workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE,
    FOREIGN KEY(parent_candidate_id) REFERENCES candidates(candidate_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS candidate_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    evaluation_kind TEXT NOT NULL,
    split TEXT NOT NULL,
    metric TEXT NOT NULL,
    scope TEXT NOT NULL,
    neuron_id INTEGER,
    value REAL,
    created_at_utc TEXT NOT NULL,
    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS candidate_artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT,
    size_bytes INTEGER,
    created_at_utc TEXT NOT NULL,
    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    member_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    finished_at_utc TEXT,
    command TEXT,
    config_sha256 TEXT NOT NULL,
    config_json TEXT NOT NULL,
    run_dir TEXT NOT NULL,
    dataset_fingerprint TEXT,
    split_sha256 TEXT,
    session_mapping_json TEXT,
    git_commit TEXT,
    git_branch TEXT,
    git_dirty INTEGER,
    git_diff_sha256 TEXT,
    environment_json TEXT,
    notes TEXT,
    error TEXT,
    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    UNIQUE(candidate_id, member_index)
);

CREATE TABLE IF NOT EXISTS params (
    run_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    PRIMARY KEY (run_id, key),
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS seeds (
    run_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    seed INTEGER,
    PRIMARY KEY (run_id, namespace),
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS epochs (
    run_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    phase TEXT NOT NULL,
    train_loss REAL,
    train_poisson REAL,
    val_corr_mean REAL,
    val_corr_median REAL,
    lr REAL,
    created_at_utc TEXT NOT NULL,
    PRIMARY KEY (run_id, epoch),
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    split TEXT NOT NULL,
    metric TEXT NOT NULL,
    scope TEXT NOT NULL,
    neuron_id INTEGER,
    value REAL,
    created_at_utc TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    phase TEXT NOT NULL,
    path TEXT NOT NULL,
    file_sha256 TEXT NOT NULL,
    parameter_sha256 TEXT NOT NULL,
    is_best INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT,
    size_bytes INTEGER,
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS progress (
    run_id TEXT PRIMARY KEY,
    phase TEXT,
    epoch INTEGER,
    batch INTEGER,
    batches_total INTEGER,
    global_step INTEGER,
    progress_fraction REAL,
    current_loss REAL,
    current_poisson REAL,
    current_lr REAL,
    best_val_r REAL,
    best_epoch INTEGER,
    elapsed_seconds REAL,
    estimated_remaining_seconds REAL,
    latest_checkpoint TEXT,
    last_heartbeat_utc TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);
CREATE INDEX IF NOT EXISTS idx_candidates_workflow ON candidates(workflow_id);
CREATE INDEX IF NOT EXISTS idx_candidates_stage ON candidates(workflow_id, stage);
CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates(status);
CREATE INDEX IF NOT EXISTS idx_candidates_hash ON candidates(config_sha256);
CREATE INDEX IF NOT EXISTS idx_candidate_metrics_lookup
    ON candidate_metrics(candidate_id, evaluation_kind, split, metric, scope);
CREATE INDEX IF NOT EXISTS idx_candidate_artifacts_candidate
    ON candidate_artifacts(candidate_id);
CREATE INDEX IF NOT EXISTS idx_runs_candidate ON runs(candidate_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_member ON runs(candidate_id, member_index);
CREATE INDEX IF NOT EXISTS idx_metrics_lookup ON metrics(metric, split, scope);
CREATE INDEX IF NOT EXISTS idx_params_key ON params(key, value_json);

PRAGMA user_version = 4;
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def flatten_dict(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = "%s.%s" % (prefix, key) if prefix else str(key)
            out.update(flatten_dict(value, next_prefix))
    else:
        out[prefix] = obj
    return out


def candidate_identity_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the config subset that defines candidate identity.

    Paths, result ledgers and ensemble replication bookkeeping do not create a
    different candidate.  Everything else remains part of candidate identity.
    """
    identity = copy.deepcopy(cfg)
    for key in _EXECUTION_ONLY_TOP_LEVEL_KEYS:
        identity.pop(key, None)
    for path in _EXECUTION_ONLY_CONFIG_PATHS:
        target = identity
        for name in path[:-1]:
            if not isinstance(target, dict) or name not in target:
                target = None
                break
            target = target[name]
        if isinstance(target, dict):
            target.pop(path[-1], None)
    return identity


def candidate_config_sha256(cfg: Dict[str, Any]) -> str:
    return config_sha256(candidate_identity_config(cfg))


class RunRegistry:
    def __init__(self, sqlite_path: str) -> None:
        self.sqlite_path = str(sqlite_path)

    def connect(self):
        con = sqlite3.connect(self.sqlite_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        return con

    def initialize(self) -> None:
        """Initialize schema v4.

        There is intentionally no migration from the old run-centric schema.
        If a legacy DB is encountered, delete it or point the registry at a new
        path.  Silent partial upgrades would corrupt the candidate/run model.
        """
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            existing_runs = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
            ).fetchone()
            if existing_runs is not None:
                columns = {
                    row["name"] for row in con.execute("PRAGMA table_info(runs)").fetchall()
                }
                if "candidate_id" not in columns:
                    raise RuntimeError(
                        "Legacy orchestration DB detected at %s. "
                        "Delete it and initialize a fresh schema-v4 database."
                        % self.sqlite_path
                    )
            version = int(con.execute("PRAGMA user_version").fetchone()[0])
            if version not in (0, SCHEMA_VERSION):
                raise RuntimeError(
                    "Unsupported orchestration schema version %d at %s"
                    % (version, self.sqlite_path)
                )
            con.executescript(SCHEMA)

    # ------------------------------------------------------------------
    # Workflow level

    def create_workflow(self, name: str, notes: Optional[str] = None) -> str:
        self.initialize()
        workflow_id = str(uuid.uuid4())
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO workflows(
                    workflow_id,name,status,created_at_utc,notes
                ) VALUES (?,?,?,?,?)
                """,
                (workflow_id, str(name), "running", utcnow(), notes),
            )
        return workflow_id

    def get_workflow(self, workflow_id: str) -> Dict[str, Any]:
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM workflows WHERE workflow_id=?", (workflow_id,)
            ).fetchone()
        if row is None:
            raise KeyError("Unknown workflow_id %s" % workflow_id)
        return dict(row)

    def list_workflows(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM workflows WHERE 1=1"
        args: List[Any] = []
        if status is not None:
            query += " AND status=?"
            args.append(status)
        query += " ORDER BY created_at_utc DESC"
        with self.connect() as con:
            return [dict(row) for row in con.execute(query, args).fetchall()]

    def finish_workflow(
        self, workflow_id: str, status: str = "completed"
    ) -> None:
        with self.connect() as con:
            cur = con.execute(
                """
                UPDATE workflows
                SET status=?, finished_at_utc=?
                WHERE workflow_id=?
                """,
                (status, utcnow(), workflow_id),
            )
            if cur.rowcount == 0:
                raise KeyError("Unknown workflow_id %s" % workflow_id)

    # ------------------------------------------------------------------
    # Candidate/configuration level

    def create_candidate(
        self,
        workflow_id: str,
        stage: str,
        cfg: Dict[str, Any],
        parent_candidate_id: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, str]:
        self.initialize()
        self.get_workflow(workflow_id)
        stage = str(stage).strip().lower()
        if not stage:
            raise ValueError("stage must be non-empty")

        if parent_candidate_id is not None:
            parent = self.get_candidate(parent_candidate_id)
            if parent["workflow_id"] != workflow_id:
                raise ValueError("parent candidate belongs to a different workflow")

        candidate_id = str(uuid.uuid4())
        identity_cfg = candidate_identity_config(cfg)
        cfg_sha = config_sha256(identity_cfg)
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO candidates(
                    candidate_id,workflow_id,parent_candidate_id,stage,status,
                    created_at_utc,config_sha256,config_json,
                    identity_config_json,overrides_json,notes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    candidate_id,
                    workflow_id,
                    parent_candidate_id,
                    stage,
                    "pending",
                    utcnow(),
                    cfg_sha,
                    canonical_json(cfg),
                    canonical_json(identity_cfg),
                    canonical_json(overrides or {}),
                    notes,
                ),
            )
        return {"candidate_id": candidate_id, "config_sha256": cfg_sha}

    def get_or_create_candidate(
        self,
        workflow_id: str,
        stage: str,
        cfg: Dict[str, Any],
        parent_candidate_id: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return an existing identical candidate or create it.

        Identity is scoped by workflow, stage, parent and the normalized candidate
        configuration.  This makes SEARCH/SWEEP plans idempotent on rerun.
        """
        self.initialize()
        stage = str(stage).strip().lower()
        cfg_sha = candidate_config_sha256(cfg)
        query = (
            "SELECT * FROM candidates WHERE workflow_id=? AND stage=? "
            "AND config_sha256=? AND "
            "((parent_candidate_id IS NULL AND ? IS NULL) OR parent_candidate_id=?) "
            "ORDER BY created_at_utc ASC LIMIT 1"
        )
        with self.connect() as con:
            row = con.execute(
                query,
                (workflow_id, stage, cfg_sha, parent_candidate_id, parent_candidate_id),
            ).fetchone()
        if row is not None:
            out = dict(row)
            out["created"] = False
            return out
        created = self.create_candidate(
            workflow_id=workflow_id,
            stage=stage,
            cfg=cfg,
            parent_candidate_id=parent_candidate_id,
            overrides=overrides,
            notes=notes,
        )
        out = self.get_candidate(created["candidate_id"])
        out["created"] = True
        return out

    def get_selected_candidate(
        self, workflow_id: str, stage: Optional[str] = None
    ) -> Dict[str, Any]:
        query = "SELECT * FROM candidates WHERE workflow_id=? AND is_selected=1"
        args: List[Any] = [workflow_id]
        if stage is not None:
            query += " AND stage=?"
            args.append(str(stage).lower())
        query += " ORDER BY created_at_utc DESC"
        with self.connect() as con:
            rows = con.execute(query, args).fetchall()
        if not rows:
            raise KeyError(
                "No selected candidate for workflow %s%s"
                % (workflow_id, " stage %s" % stage if stage is not None else "")
            )
        if len(rows) > 1:
            raise RuntimeError("Multiple selected candidates match the request")
        return dict(rows[0])

    def get_candidate(self, candidate_id: str) -> Dict[str, Any]:
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)
            ).fetchone()
        if row is None:
            raise KeyError("Unknown candidate_id %s" % candidate_id)
        return dict(row)

    def get_candidate_config(self, candidate_id: str) -> Dict[str, Any]:
        return json.loads(self.get_candidate(candidate_id)["config_json"])

    def list_candidates(
        self,
        workflow_id: Optional[str] = None,
        stage: Optional[str] = None,
        status: Optional[str] = None,
        promoted: Optional[bool] = None,
        selected: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM candidates WHERE 1=1"
        args: List[Any] = []
        if workflow_id is not None:
            query += " AND workflow_id=?"
            args.append(workflow_id)
        if stage is not None:
            query += " AND stage=?"
            args.append(str(stage).lower())
        if status is not None:
            query += " AND status=?"
            args.append(status)
        if promoted is not None:
            query += " AND is_promoted=?"
            args.append(1 if promoted else 0)
        if selected is not None:
            query += " AND is_selected=?"
            args.append(1 if selected else 0)
        query += " ORDER BY created_at_utc ASC"
        with self.connect() as con:
            return [dict(row) for row in con.execute(query, args).fetchall()]

    def mark_candidate_running(self, candidate_id: str) -> None:
        with self.connect() as con:
            cur = con.execute(
                """
                UPDATE candidates
                SET status='running', finished_at_utc=NULL
                WHERE candidate_id=?
                """,
                (candidate_id,),
            )
            if cur.rowcount == 0:
                raise KeyError("Unknown candidate_id %s" % candidate_id)

    def finish_candidate(
        self, candidate_id: str, status: str = "completed"
    ) -> None:
        with self.connect() as con:
            cur = con.execute(
                """
                UPDATE candidates
                SET status=?, finished_at_utc=?
                WHERE candidate_id=?
                """,
                (status, utcnow(), candidate_id),
            )
            if cur.rowcount == 0:
                raise KeyError("Unknown candidate_id %s" % candidate_id)

    def set_candidate_promoted(self, candidate_id: str, promoted: bool = True) -> None:
        """Record the decision to spend replication budget on a candidate."""
        with self.connect() as con:
            cur = con.execute(
                "UPDATE candidates SET is_promoted=? WHERE candidate_id=?",
                (1 if promoted else 0, candidate_id),
            )
            if cur.rowcount == 0:
                raise KeyError("Unknown candidate_id %s" % candidate_id)

    def set_candidate_selected(self, candidate_id: str, selected: bool = True) -> None:
        """Mark the founder/selected candidate, exclusively within its workflow stage."""
        candidate = self.get_candidate(candidate_id)
        with self.connect() as con:
            if selected:
                con.execute(
                    """
                    UPDATE candidates SET is_selected=0
                    WHERE workflow_id=? AND stage=? AND candidate_id<>?
                    """,
                    (candidate["workflow_id"], candidate["stage"], candidate_id),
                )
            cur = con.execute(
                "UPDATE candidates SET is_selected=? WHERE candidate_id=?",
                (1 if selected else 0, candidate_id),
            )
            if cur.rowcount == 0:
                raise KeyError("Unknown candidate_id %s" % candidate_id)

    def log_candidate_metric(
        self,
        candidate_id: str,
        evaluation_kind: str,
        split: str,
        metric: str,
        scope: str,
        value: float,
        neuron_id: Optional[int] = None,
    ) -> None:
        """Log a candidate-level metric.

        ``evaluation_kind`` separates cheap single-member screening from the
        replicated ensemble evaluation of the same candidate.  Per-neuron
        ensemble correlations are stored as ordinary rows (scope="neuron",
        neuron_id=<id>); threshold counts and quantiles are derived later.
        """
        self.get_candidate(candidate_id)
        evaluation_kind = str(evaluation_kind).strip().lower()
        if not evaluation_kind:
            raise ValueError("evaluation_kind must be non-empty")
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO candidate_metrics(
                    candidate_id,evaluation_kind,split,metric,scope,
                    neuron_id,value,created_at_utc
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    candidate_id,
                    evaluation_kind,
                    split,
                    metric,
                    scope,
                    neuron_id,
                    (float(value) if math.isfinite(float(value)) else None),
                    utcnow(),
                ),
            )

    def log_candidate_metrics(
        self,
        candidate_id: str,
        evaluation_kind: str,
        split: str,
        metric: str,
        rows: Iterable[Dict[str, Any]],
    ) -> None:
        """Bulk-log candidate metrics, primarily per-neuron ensemble values."""
        self.get_candidate(candidate_id)
        evaluation_kind = str(evaluation_kind).strip().lower()
        if not evaluation_kind:
            raise ValueError("evaluation_kind must be non-empty")
        payload = []
        now = utcnow()
        for row in rows:
            payload.append(
                (
                    candidate_id,
                    evaluation_kind,
                    split,
                    metric,
                    str(row.get("scope", "neuron")),
                    None if row.get("neuron_id") is None else int(row["neuron_id"]),
                    (
                        None
                        if row.get("value") is None
                        else (
                            float(row["value"])
                            if math.isfinite(float(row["value"]))
                            else None
                        )
                    ),
                    now,
                )
            )
        with self.connect() as con:
            con.executemany(
                """
                INSERT INTO candidate_metrics(
                    candidate_id,evaluation_kind,split,metric,scope,
                    neuron_id,value,created_at_utc
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                payload,
            )

    def clear_candidate_metrics(
        self,
        candidate_id: str,
        evaluation_kind: Optional[str] = None,
        split: Optional[str] = None,
        metric: Optional[str] = None,
    ) -> None:
        """Delete a previous candidate evaluation before an intentional re-evaluation."""
        self.get_candidate(candidate_id)
        query = "DELETE FROM candidate_metrics WHERE candidate_id=?"
        args: List[Any] = [candidate_id]
        if evaluation_kind is not None:
            query += " AND evaluation_kind=?"
            args.append(str(evaluation_kind).strip().lower())
        if split is not None:
            query += " AND split=?"
            args.append(split)
        if metric is not None:
            query += " AND metric=?"
            args.append(metric)
        with self.connect() as con:
            con.execute(query, args)

    def log_candidate_artifact(
        self, candidate_id: str, kind: str, path: str
    ) -> None:
        self.get_candidate(candidate_id)
        path_obj = Path(path)
        sha = sha256_file(str(path_obj)) if path_obj.is_file() else None
        size = path_obj.stat().st_size if path_obj.is_file() else None
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO candidate_artifacts(
                    candidate_id,kind,path,sha256,size_bytes,created_at_utc
                ) VALUES (?,?,?,?,?,?)
                """,
                (candidate_id, kind, str(path_obj), sha, size, utcnow()),
            )

    def candidate_runs(
        self,
        candidate_id: str,
        status: Optional[str] = None,
        member_index: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        self.get_candidate(candidate_id)
        return self.list_runs(
            status=status, member_index=member_index, candidate_id=candidate_id
        )

    # ------------------------------------------------------------------
    # Run/execution level

    def start_run(
        self,
        candidate_id: str,
        member_index: int,
        command: str,
        seeds: Dict[str, int],
        dataset_fingerprint: str,
        split_sha256: str,
        session_mapping: List[Dict[str, Any]],
        cfg: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, str]:
        """Start one stochastic realization of a candidate.

        ``cfg`` is the exact execution snapshot.  It may differ from the stored
        candidate config only in execution-only fields (paths and ensemble/member
        bookkeeping).  This lets a search candidate be evaluated with one member
        and later expanded to an ensemble without becoming a different candidate.
        """
        self.initialize()
        candidate = self.get_candidate(candidate_id)
        if cfg is None:
            cfg = json.loads(candidate["config_json"])
        else:
            cfg = copy.deepcopy(cfg)

        if candidate_config_sha256(cfg) != candidate["config_sha256"]:
            raise ValueError(
                "execution config changes candidate identity for %s" % candidate_id
            )

        run_id = str(uuid.uuid4())
        output_root = Path(cfg["project"]["output_root"])
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        git = capture_git_state()
        env = capture_environment()
        cfg_sha = config_sha256(cfg)

        save_resolved_config(cfg, str(run_dir / "resolved_config.yaml"))
        with open(run_dir / "git.diff", "w") as f:
            f.write(git["diff"])
        with open(run_dir / "environment.json", "w") as f:
            json.dump(env, f, indent=2, sort_keys=True)
        with open(run_dir / "session_mapping.json", "w") as f:
            json.dump(session_mapping, f, indent=2, sort_keys=True)
        with open(run_dir / "candidate.json", "w") as f:
            json.dump(
                {
                    "candidate_id": candidate_id,
                    "workflow_id": candidate["workflow_id"],
                    "stage": candidate["stage"],
                    "candidate_config_sha256": candidate["config_sha256"],
                    "member_index": int(member_index),
                },
                f,
                indent=2,
                sort_keys=True,
            )

        with self.connect() as con:
            con.execute(
                """
                INSERT INTO runs (
                    run_id,candidate_id,member_index,status,created_at_utc,
                    command,config_sha256,config_json,run_dir,
                    dataset_fingerprint,split_sha256,session_mapping_json,
                    git_commit,git_branch,git_dirty,git_diff_sha256,
                    environment_json,notes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    candidate_id,
                    int(member_index),
                    "running",
                    utcnow(),
                    command,
                    cfg_sha,
                    canonical_json(cfg),
                    str(run_dir),
                    dataset_fingerprint,
                    split_sha256,
                    json.dumps(session_mapping, sort_keys=True),
                    git["commit"],
                    git["branch"],
                    1 if git["dirty"] else 0,
                    git["diff_sha256"],
                    json.dumps(env, sort_keys=True),
                    notes,
                ),
            )
            con.execute(
                "UPDATE candidates SET status='running', finished_at_utc=NULL WHERE candidate_id=?",
                (candidate_id,),
            )
            for key, value in flatten_dict(cfg).items():
                con.execute(
                    "INSERT INTO params(run_id,key,value_json) VALUES (?,?,?)",
                    (run_id, key, json.dumps(value, sort_keys=True)),
                )
            for namespace, seed in seeds.items():
                con.execute(
                    "INSERT INTO seeds(run_id,namespace,seed) VALUES (?,?,?)",
                    (run_id, namespace, int(seed)),
                )

        return {"run_id": run_id, "run_dir": str(run_dir)}

    def log_epoch(
        self,
        run_id: str,
        epoch: int,
        phase: str,
        train_loss: float,
        train_poisson: float,
        val_corr_mean: float,
        val_corr_median: float,
        lr: float,
    ) -> None:
        with self.connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO epochs(
                    run_id,epoch,phase,train_loss,train_poisson,
                    val_corr_mean,val_corr_median,lr,created_at_utc
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    int(epoch),
                    phase,
                    float(train_loss),
                    float(train_poisson),
                    float(val_corr_mean),
                    float(val_corr_median),
                    float(lr),
                    utcnow(),
                ),
            )

    def log_checkpoint(
        self,
        run_id: str,
        epoch: int,
        phase: str,
        path: str,
        file_sha256: str,
        parameter_sha256: str,
        is_best: bool,
    ) -> None:
        with self.connect() as con:
            if is_best:
                con.execute(
                    "UPDATE checkpoints SET is_best=0 WHERE run_id=?",
                    (run_id,),
                )
            con.execute(
                """
                INSERT INTO checkpoints(
                    run_id,epoch,phase,path,file_sha256,parameter_sha256,is_best
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    int(epoch),
                    phase,
                    path,
                    file_sha256,
                    parameter_sha256,
                    1 if is_best else 0,
                ),
            )

    def log_metric(
        self,
        run_id: str,
        split: str,
        metric: str,
        scope: str,
        value: float,
        neuron_id: Optional[int] = None,
    ) -> None:
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO metrics(
                    run_id,split,metric,scope,neuron_id,value,created_at_utc
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    split,
                    metric,
                    scope,
                    neuron_id,
                    (float(value) if math.isfinite(float(value)) else None),
                    utcnow(),
                ),
            )

    def log_artifact(self, run_id: str, kind: str, path: str) -> None:
        path_obj = Path(path)
        sha = sha256_file(str(path_obj)) if path_obj.is_file() else None
        size = path_obj.stat().st_size if path_obj.is_file() else None
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO artifacts(run_id,kind,path,sha256,size_bytes)
                VALUES (?,?,?,?,?)
                """,
                (run_id, kind, str(path_obj), sha, size),
            )

    def mark_running(self, run_id: str, command=None) -> None:
        with self.connect() as con:
            if command is None:
                con.execute(
                    "UPDATE runs SET status='running', error=NULL WHERE run_id=?",
                    (run_id,),
                )
            else:
                con.execute(
                    "UPDATE runs SET status='running', error=NULL, command=? WHERE run_id=?",
                    (command, run_id),
                )

    def update_progress(
        self,
        run_id,
        phase,
        epoch,
        batch,
        batches_total,
        global_step,
        progress_fraction,
        current_loss,
        current_poisson,
        current_lr,
        best_val_r,
        best_epoch,
        elapsed_seconds,
        estimated_remaining_seconds,
        latest_checkpoint,
    ) -> None:
        with self.connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO progress(
                    run_id,phase,epoch,batch,batches_total,global_step,
                    progress_fraction,current_loss,current_poisson,current_lr,
                    best_val_r,best_epoch,elapsed_seconds,
                    estimated_remaining_seconds,latest_checkpoint,last_heartbeat_utc
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    phase,
                    int(epoch),
                    int(batch),
                    int(batches_total),
                    int(global_step),
                    float(progress_fraction),
                    current_loss,
                    current_poisson,
                    float(current_lr),
                    best_val_r,
                    best_epoch,
                    float(elapsed_seconds),
                    estimated_remaining_seconds,
                    latest_checkpoint,
                    utcnow(),
                ),
            )

    def get_progress(self, run_id: str):
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM progress WHERE run_id=?", (run_id,)
            ).fetchone()
        return None if row is None else dict(row)

    def finish_run(self, run_id: str, status="completed", error=None) -> None:
        with self.connect() as con:
            con.execute(
                """
                UPDATE runs
                SET status=?, finished_at_utc=?, error=?
                WHERE run_id=?
                """,
                (status, utcnow(), error, run_id),
            )

    def get_run(self, run_id: str):
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError("Unknown run_id %s" % run_id)
        return dict(row)

    def best_checkpoint(self, run_id: str) -> str:
        with self.connect() as con:
            row = con.execute(
                """
                SELECT path FROM checkpoints
                WHERE run_id=?
                ORDER BY is_best DESC, epoch DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError("No checkpoint found for %s" % run_id)
        return row["path"]

    def best_validation(self, run_id: str) -> Dict[str, Any]:
        """Return the best recorded validation epoch for one physical run."""
        self.get_run(run_id)
        with self.connect() as con:
            row = con.execute(
                """
                SELECT epoch,val_corr_mean,val_corr_median,phase,lr
                FROM epochs
                WHERE run_id=? AND val_corr_mean IS NOT NULL
                ORDER BY val_corr_mean DESC, epoch ASC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError("No validation metrics found for %s" % run_id)
        return dict(row)

    def list_runs(
        self,
        status: Optional[str] = None,
        member_index: Optional[int] = None,
        candidate_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM runs WHERE 1=1"
        args: List[Any] = []
        if status is not None:
            query += " AND status=?"
            args.append(status)
        if member_index is not None:
            query += " AND member_index=?"
            args.append(int(member_index))
        if candidate_id is not None:
            query += " AND candidate_id=?"
            args.append(candidate_id)
        query += " ORDER BY created_at_utc DESC"
        with self.connect() as con:
            return [dict(row) for row in con.execute(query, args).fetchall()]

    # ------------------------------------------------------------------
    # Flat exports for quick inspection.  SQLite remains authoritative.

    def export_results(self, output_path: str) -> None:
        query = """
        SELECT
            w.workflow_id,
            c.candidate_id,
            c.stage,
            c.config_sha256 AS candidate_config_sha256,
            c.is_selected,
            r.run_id,
            r.member_index,
            r.status,
            r.created_at_utc,
            r.finished_at_utc,
            MAX(CASE WHEN m.metric='corr_to_average' AND m.scope='mean'
                     THEN m.value END) AS corr_to_average_mean,
            MAX(CASE WHEN m.metric='corr_to_average' AND m.scope='median'
                     THEN m.value END) AS corr_to_average_median,
            MAX(CASE WHEN m.metric='single_trial_corr' AND m.scope='mean'
                     THEN m.value END) AS single_trial_corr_mean
        FROM runs r
        JOIN candidates c ON c.candidate_id=r.candidate_id
        JOIN workflows w ON w.workflow_id=c.workflow_id
        LEFT JOIN metrics m ON m.run_id=r.run_id
        GROUP BY r.run_id
        ORDER BY r.created_at_utc DESC
        """
        with self.connect() as con:
            rows = [dict(x) for x in con.execute(query).fetchall()]
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "workflow_id",
            "candidate_id",
            "stage",
            "candidate_config_sha256",
            "is_selected",
            "run_id",
            "member_index",
            "status",
            "created_at_utc",
            "finished_at_utc",
            "corr_to_average_mean",
            "corr_to_average_median",
            "single_trial_corr_mean",
        ]
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def export_candidates(
        self,
        output_path: str,
        workflow_id: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> None:
        """Export the candidate-level decision table.

        Stored per-neuron ensemble ``corr_to_average`` values are the source of
        truth.  Quantiles and the published r>0.75 inclusion summary are derived
        at export time rather than stored as irreversible aggregate metrics.
        """
        where = ["1=1"]
        base_args: List[Any] = []
        if workflow_id is not None:
            where.append("c.workflow_id=?")
            base_args.append(workflow_id)
        if stage is not None:
            where.append("c.stage=?")
            base_args.append(str(stage).lower())
        base_query = """
        SELECT
            c.workflow_id,
            c.candidate_id,
            c.parent_candidate_id,
            c.stage,
            c.status,
            c.config_sha256,
            c.is_promoted,
            c.is_selected,
            c.created_at_utc,
            c.finished_at_utc,
            c.overrides_json,
            COUNT(DISTINCT r.run_id) AS n_runs,
            COUNT(DISTINCT CASE WHEN r.status='completed' THEN r.run_id END)
                AS n_completed_runs
        FROM candidates c
        LEFT JOIN runs r ON r.candidate_id=c.candidate_id
        WHERE %s
        GROUP BY c.candidate_id
        ORDER BY c.created_at_utc ASC
        """ % " AND ".join(where)

        with self.connect() as con:
            rows = [dict(x) for x in con.execute(base_query, base_args).fetchall()]

            # Candidate-level scalar metrics remain useful for screening and for
            # aggregate ensemble metrics other than corr_to_average.
            scalar_rows = [
                dict(x)
                for x in con.execute(
                    """
                    SELECT candidate_id,evaluation_kind,split,metric,scope,value
                    FROM candidate_metrics
                    WHERE neuron_id IS NULL
                    ORDER BY metric_id ASC
                    """
                ).fetchall()
            ]

            per_neuron_rows = [
                dict(x)
                for x in con.execute(
                    """
                    SELECT candidate_id,neuron_id,value
                    FROM candidate_metrics
                    WHERE evaluation_kind='ensemble'
                      AND split='test'
                      AND metric='corr_to_average'
                      AND neuron_id IS NOT NULL
                    ORDER BY candidate_id, neuron_id
                    """
                ).fetchall()
            ]

        # Keep only the latest scalar value for any repeated intentional
        # evaluation of the same metric key.
        scalars: Dict[tuple, float] = {}
        for m in scalar_rows:
            key = (
                m["candidate_id"],
                m["evaluation_kind"],
                m["split"],
                m["metric"],
                m["scope"],
            )
            scalars[key] = m["value"]

        per_neuron: Dict[str, List[Optional[float]]] = {}
        for m in per_neuron_rows:
            value = None if m["value"] is None else float(m["value"])
            per_neuron.setdefault(m["candidate_id"], []).append(value)

        def scalar(candidate_id, evaluation_kind, split, metric, scope):
            return scalars.get(
                (candidate_id, evaluation_kind, split, metric, scope)
            )

        def quantile(values: List[float], q: float) -> Optional[float]:
            if not values:
                return None
            ordered = sorted(values)
            if len(ordered) == 1:
                return ordered[0]
            pos = (len(ordered) - 1) * q
            lo = int(pos)
            hi = min(lo + 1, len(ordered) - 1)
            frac = pos - lo
            return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

        for row in rows:
            cid = row["candidate_id"]
            raw_values = per_neuron.get(cid, [])
            # Match the evaluator's historical population summaries: undefined
            # neuron correlations contribute 0.0 to the scored population.
            values = [0.0 if v is None else float(v) for v in raw_values]
            n_valid = sum(v is not None for v in raw_values)

            # Single-run screening score: selection into the promoted set is
            # normally based on validation, not test.
            row["screen_val_corr_mean"] = scalar(
                cid, "screening", "val", "corr", "mean"
            )
            if row["screen_val_corr_mean"] is None:
                row["screen_val_corr_mean"] = scalar(
                    cid, "screening", "val", "corr_to_average", "mean"
                )

            row["replication_val_corr_mean"] = scalar(
                cid, "replication", "val", "corr", "mean"
            )
            row["replication_val_corr_median"] = scalar(
                cid, "replication", "val", "corr", "median"
            )

            # Final promoted-candidate comparison.  The raw neuron-level values
            # are stored in candidate_metrics; everything below is derived.
            if values:
                row["ensemble_test_corr_mean"] = scalar(
                    cid, "ensemble", "test", "corr_to_average", "mean"
                )
                if row["ensemble_test_corr_mean"] is None:
                    row["ensemble_test_corr_mean"] = sum(values) / len(values)
                row["ensemble_test_corr_q10"] = quantile(values, 0.10)
                row["ensemble_test_corr_q25"] = quantile(values, 0.25)
                row["ensemble_test_corr_q50"] = quantile(values, 0.50)
                row["ensemble_test_corr_q75"] = quantile(values, 0.75)
                row["ensemble_test_corr_q90"] = quantile(values, 0.90)
                row["ensemble_test_n_neurons"] = len(values)
                row["ensemble_test_n_valid"] = n_valid
                row["ensemble_test_n_gt_0_75"] = sum(v > 0.75 for v in values)
                row["ensemble_test_frac_gt_0_75"] = (
                    row["ensemble_test_n_gt_0_75"] / len(values)
                )
            else:
                row["ensemble_test_corr_mean"] = None
                row["ensemble_test_corr_q10"] = None
                row["ensemble_test_corr_q25"] = None
                row["ensemble_test_corr_q50"] = None
                row["ensemble_test_corr_q75"] = None
                row["ensemble_test_corr_q90"] = None
                row["ensemble_test_n_neurons"] = 0
                row["ensemble_test_n_valid"] = 0
                row["ensemble_test_n_gt_0_75"] = None
                row["ensemble_test_frac_gt_0_75"] = None

            row["ensemble_test_single_trial_corr_mean"] = scalar(
                cid, "ensemble", "test", "single_trial_corr", "mean"
            )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "workflow_id",
            "candidate_id",
            "parent_candidate_id",
            "stage",
            "status",
            "config_sha256",
            "is_promoted",
            "is_selected",
            "n_runs",
            "n_completed_runs",
            "screen_val_corr_mean",
            "replication_val_corr_mean",
            "replication_val_corr_median",
            "ensemble_test_corr_mean",
            "ensemble_test_corr_q10",
            "ensemble_test_corr_q25",
            "ensemble_test_corr_q50",
            "ensemble_test_corr_q75",
            "ensemble_test_corr_q90",
            "ensemble_test_n_neurons",
            "ensemble_test_n_valid",
            "ensemble_test_n_gt_0_75",
            "ensemble_test_frac_gt_0_75",
            "ensemble_test_single_trial_corr_mean",
            "overrides_json",
            "created_at_utc",
            "finished_at_utc",
        ]
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
