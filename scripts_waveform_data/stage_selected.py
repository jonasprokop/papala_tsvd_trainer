"""Stage explicitly selected files from read-only source to host disk.

No copy occurs without --execute. Never overwrites a conflicting destination.
A manifest is a UTF-8 text file with one relative FILE path per line. Directories
are intentionally not accepted: this prevents accidentally copying whole archives.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _parse_list(manifest: Path):
    for line_no, raw in enumerate(manifest.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"Unsafe relative path in manifest at line {line_no}: {line}")
        yield candidate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("/source/tvsd"))
    p.add_argument("--destination", type=Path, default=Path("/project/data/raw"))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--max-gb", type=float, default=50.0,
                   help="Maximum new data to stage; raise after examining disk space")
    args = p.parse_args()
    source = args.source.resolve(strict=True)
    destination = args.destination.resolve()
    if not source.is_dir():
        p.error("Source is not a directory")
    if args.max_gb <= 0:
        p.error("--max-gb must be positive")

    candidates = []
    count_skipped = 0
    seen = set()
    for relative in _parse_list(args.manifest):
        if relative in seen:
            continue
        seen.add(relative)
        src = (source / relative).resolve(strict=True)
        if not src.is_relative_to(source) or not src.is_file():
            raise ValueError(f"Not a safe existing source file: {relative}")
        dest = destination / relative
        if dest.exists():
            if not dest.is_file() or dest.stat().st_size != src.stat().st_size:
                raise FileExistsError(f"Conflicting destination (never overwritten): {dest}")
            count_skipped += 1
            print("EXISTS size matches (skipping):", relative)
            continue
        candidates.append((src, dest, src.stat().st_size, relative))

    total = sum(item[2] for item in candidates)
    print("Selected new files:", len(candidates))
    print("Already present, size matches:", count_skipped)
    print("Estimated new bytes:", total, f"({total / (1024**3):.3f} GiB)")
    print("Destination:", destination)
    if total > args.max_gb * (1024**3):
        raise ValueError(f"Exceeds --max-gb={args.max_gb}: review and explicitly increase limit")
    if not args.execute:
        print("DRY RUN — add --execute after reviewing paths and disk usage")
        for _, _, size, relative in candidates[:80]:
            print(f"  {size:>14}  {relative}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination).free
    if free < total:
        raise OSError(f"Insufficient free space: {free} bytes available, {total} needed")
    for src, dest, size, relative in candidates:
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive-create to avoid accidental overwrite in concurrent runs.
        print("COPY", relative, size, "bytes", flush=True)
        with src.open("rb") as fin, dest.open("xb") as fout:
            shutil.copyfileobj(fin, fout, length=8 * 1024 * 1024)
        shutil.copystat(src, dest)
    print("STAGING COMPLETE:", len(candidates), "new files")


if __name__ == "__main__":
    main()
