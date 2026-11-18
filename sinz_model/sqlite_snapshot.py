"""SQLite-consistent, atomic snapshot helper (stdlib only; safe on PBS host)."""
from __future__ import annotations

import argparse
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path


def sqlite_snapshot(source: Path, target: Path):
    source, target = Path(source), Path(target)
    if not source.is_file():
        raise FileNotFoundError(f"Missing SQLite source: {source}")
    if target.exists():
        raise FileExistsError(f"Snapshot destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro",
                                 uri=True, timeout=10)) as reader:
        with closing(sqlite3.connect(str(target), timeout=10)) as writer:
            reader.backup(writer)
            if writer.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RuntimeError(f"Snapshot failed SQLite integrity check: {source}")


def atomic_snapshot(source: Path, destination: Path):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tvsd-handoff-", dir=destination.parent) as folder:
        pending = Path(folder) / "worker.sqlite"
        sqlite_snapshot(Path(source), pending)
        os.replace(pending, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    print(atomic_snapshot(Path(args.source), Path(args.destination)))


if __name__ == "__main__":
    main()
