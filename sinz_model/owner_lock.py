"""Recoverable single-host ownership checks for filesystem mkdir locks.

Creation is atomic on a shared filesystem. Recovery is conservative: a lock
from another host, an unknown PID, or an unrecognized record is never stolen.
"""
from __future__ import annotations

import os
import re
import socket
from contextlib import contextmanager
from pathlib import Path


OWNER_PATTERN = re.compile(r"host=([^\s]+) pid=([1-9][0-9]*)\n?\Z")


@contextmanager
def owner_lock(lock: Path):
    """Atomic mkdir lock with a host/PID owner record."""
    lock = Path(lock)
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise RuntimeError(f"Coordinator already active: {lock}") from exc
    try:
        (lock / "owner").write_text(f"host={socket.gethostname()} pid={os.getpid()}\n")
        yield
    finally:
        (lock / "owner").unlink(missing_ok=True)
        lock.rmdir()


def break_owner_lock(lock: Path) -> Path:
    """Remove a stale lock only when the recorded PID is dead on this host."""
    lock = Path(lock)
    owner = lock / "owner"
    if lock.is_symlink() or not lock.is_dir() or owner.is_symlink() or not owner.is_file():
        raise RuntimeError(f"Missing or unverifiable coordinator lock owner: {lock}")

    def signature(path: Path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns

    initial_lock_stat = signature(lock)
    initial_owner_stat = signature(owner)
    contents = owner.read_text(encoding="utf-8")
    match = OWNER_PATTERN.fullmatch(contents)
    if match is None:
        raise RuntimeError(f"Invalid coordinator lock owner record: {owner}")
    host, pid = match.group(1), int(match.group(2))
    if host != socket.gethostname():
        raise RuntimeError(f"Lock belongs to {host}; check it on that host")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    except OSError as exc:
        raise RuntimeError(f"Cannot verify lock owner PID {pid} is dead") from exc
    else:
        raise RuntimeError(f"Coordinator lock owner PID {pid} is still alive")
    if (signature(lock) != initial_lock_stat or signature(owner) != initial_owner_stat or
            owner.read_text(encoding="utf-8") != contents):
        raise RuntimeError("Coordinator lock changed during recovery")
    if list(lock.iterdir()) != [owner]:
        raise RuntimeError(f"Unexpected files inside coordinator lock: {lock}")
    owner.unlink()
    lock.rmdir()
    return lock
