"""Single-writer collection of standalone PBS runs into the project registry.

Training workers never write to the Brno2 SQLite file. Each worker writes one
scratch-local database and returns a handoff. A separate collection command
imports only the selected run and atomically publishes the updated registry.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .orchestrator import RunRegistry, candidate_config_sha256
from .owner_lock import break_owner_lock, owner_lock
from .sqlite_snapshot import atomic_snapshot, sqlite_snapshot


CHILDREN = ("params", "seeds", "epochs", "metrics", "checkpoints", "artifacts", "progress")
SKIP_IDS = {"metric_id", "artifact_id", "checkpoint_id"}
def _lock_path(project_db: Path) -> Path:
    return Path(str(Path(project_db).resolve()) + ".coordinator.lock")


@contextmanager
def project_writer_lock(project_db: Path):
    """Shared lock for the single authoritative project SQLite file."""
    with owner_lock(_lock_path(project_db)):
        yield


def break_lock(project_db: Path) -> Path:
    """Recover the project database lock; refuse unverifiable owners."""
    return break_owner_lock(_lock_path(project_db))


def _copy_rows(con, table, selector, key):
    columns = [row[1] for row in con.execute(f'PRAGMA worker.table_info("{table}")')
               if row[1] not in SKIP_IDS]
    if not columns:
        raise RuntimeError(f"Missing worker table: {table}")
    quoted = ", ".join(f'"{col}"' for col in columns)
    con.execute(f'INSERT INTO main."{table}" ({quoted}) '
                f'SELECT {quoted} FROM worker."{table}" WHERE {selector}', (key,))


def _copy_run(con, run_id, *, replace):
    if replace:
        for table in CHILDREN:
            con.execute(f'DELETE FROM main."{table}" WHERE run_id=?', (run_id,))
        con.execute("DELETE FROM main.runs WHERE run_id=?", (run_id,))
    _copy_rows(con, "runs", "run_id=?", run_id)
    for table in CHILDREN:
        _copy_rows(con, table, "run_id=?", run_id)


def _merge_local(project_copy: Path, handoff: Path, candidate_id: str):
    registry = RunRegistry(str(project_copy))
    registry.initialize()
    central = registry.get_candidate(candidate_id)
    with registry.connect() as con:
        con.execute("ATTACH DATABASE ? AS worker", (str(handoff),))
        try:
            worker_candidate = con.execute(
                "SELECT config_sha256 FROM worker.candidates WHERE candidate_id=?",
                (candidate_id,),
            ).fetchone()
            if not worker_candidate or worker_candidate["config_sha256"] != central["config_sha256"]:
                raise ValueError("Worker candidate identity differs from central registry")
            rows = con.execute(
                "SELECT * FROM worker.runs WHERE candidate_id=? AND member_index=0",
                (candidate_id,),
            ).fetchall()
            if len(rows) != 1:
                raise ValueError("Handoff must contain exactly one candidate/member run")
            source = rows[0]
            run_id = source["run_id"]
            if candidate_config_sha256(json.loads(source["config_json"])) != central["config_sha256"]:
                raise ValueError("Worker run changed candidate parameters")
            if source["status"] not in ("completed", "failed", "running"):
                raise ValueError(f"Unsupported worker run status: {source['status']}")
            existing = con.execute("SELECT run_id,status,created_at_utc FROM main.runs "
                                   "WHERE candidate_id=? AND member_index=0",
                                   (candidate_id,)).fetchone()
            if existing and existing["run_id"] != run_id:
                raise RuntimeError("Central candidate already has a different run ID")
            if existing and existing["created_at_utc"] != source["created_at_utc"]:
                raise RuntimeError("Existing run provenance changed")
            if existing and existing["status"] == "completed":
                if source["status"] == "completed":
                    return run_id, False
                raise RuntimeError("Refusing to downgrade completed central run")
            status = "completed" if source["status"] == "completed" else "failed"
            with con:
                _copy_run(con, run_id, replace=bool(existing))
                if source["status"] == "running":
                    con.execute("UPDATE main.runs SET status='failed', "
                                "error='Worker terminated before completion' WHERE run_id=?",
                                (run_id,))
                con.execute("UPDATE main.candidates SET status=?, "
                            "finished_at_utc=CURRENT_TIMESTAMP WHERE candidate_id=?",
                            (status, candidate_id))
            return run_id, True
        finally:
            con.execute("DETACH DATABASE worker")


def collect(project_db: Path, handoff: Path, candidate_id: str):
    """Import one worker run, committing the whole central DB atomically."""
    project_db = Path(project_db).resolve()
    handoff = Path(handoff).resolve()
    if project_db == handoff or not project_db.is_file() or not handoff.is_file():
        raise ValueError("Expected two existing, distinct registry files")
    with project_writer_lock(project_db):
        fingerprint = (project_db.stat().st_size, project_db.stat().st_mtime_ns)
        with tempfile.TemporaryDirectory(prefix="tvsd-collect-") as directory:
            local = Path(directory) / "project.sqlite"
            sqlite_snapshot(project_db, local)
            run_id, changed = _merge_local(local, handoff, candidate_id)
            if changed:
                current = (project_db.stat().st_size, project_db.stat().st_mtime_ns)
                if current != fingerprint:
                    raise RuntimeError("Central registry changed during collection; refusing publish")
                atomic_snapshot(local, project_db)
    return run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    merge = sub.add_parser("collect")
    merge.add_argument("--project-db", required=True)
    merge.add_argument("--worker-db", required=True)
    merge.add_argument("--candidate-id", required=True)
    recovery = sub.add_parser("break-lock")
    recovery.add_argument("--project-db", required=True)
    args = parser.parse_args()
    if args.operation == "collect":
        print("COLLECTED_RUN_ID=" + collect(
            Path(args.project_db), Path(args.worker_db), args.candidate_id))
    else:
        print("REMOVED_STALE_LOCK=" + str(break_lock(Path(args.project_db))))


if __name__ == "__main__":
    main()
