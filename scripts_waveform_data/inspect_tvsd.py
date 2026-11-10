"""Bounded TVSD tree inventory with *metadata-only* scientific format inspection.

Safe on large electrophysiology archives: no spike arrays or images are loaded into
RAM. Produces a JSON summary for planning the selective data copy.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

SKIP_NAMES = {".git", "__pycache__", ".ipynb_checkpoints", ".cache"}
HDF5_EXT = {".h5", ".hdf5", ".nix", ".hdf", ".hdf4", ".mat"}
OTHER_EXT = {".npz", ".npy", ".csv", ".tsv", ".json", ".txt"}


def _walk(root: Path, max_depth: int, max_files: int, max_entries_per_dir: int, max_dirs: int):
    """Visit directories before files, limit huge image collections."""
    directories = []
    files = []
    truncated = []
    remaining = [(root, 0)]
    while remaining:
        if len(directories) >= max_dirs:
            truncated.append(f"global directory limit reached: {max_dirs}")
            break
        folder, depth = remaining.pop(0)
        if folder.name in SKIP_NAMES:
            continue
        directories.append({"path": str(folder.relative_to(root)) or ".", "depth": depth})
        try:
            with os.scandir(folder) as scan:
                entries = sorted(scan, key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
        except (PermissionError, OSError) as exc:
            truncated.append(f"{folder}: {exc}")
            continue
        if len(entries) > max_entries_per_dir:
            truncated.append(f"{folder.relative_to(root)}: {len(entries)} entries, limited to {max_entries_per_dir}")
            entries = entries[:max_entries_per_dir]
        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                if depth < max_depth:
                    remaining.append((Path(entry.path), depth + 1))
            elif entry.is_file(follow_symlinks=False):
                if len(files) < max_files:
                    try:
                        files.append({"path": str(Path(entry.path).relative_to(root)),
                                      "bytes": entry.stat(follow_symlinks=False).st_size,
                                      "suffix": Path(entry.name).suffix.lower()})
                    except OSError as exc:
                        truncated.append(f"{entry.path}: {exc}")
                else:
                    if "global file limit reached" not in truncated:
                        truncated.append("global file limit reached")
    return directories, files, truncated


def _probe_metadata(path: Path, max_nodes: int = 24):
    ext = path.suffix.lower()
    try:
        if ext in HDF5_EXT:
            import h5py
            if not h5py.is_hdf5(str(path)):
                return {"format": "not HDF5 (possibly older MATLAB or another format)"}
            nodes = []
            with h5py.File(str(path), "r") as handle:
                def visitor(name, obj):
                    if len(nodes) >= max_nodes:
                        return True  # Stop h5py tree walk; NIX trees can be enormous.
                    if isinstance(obj, h5py.Dataset):
                        nodes.append({"name": name, "kind": "dataset", "shape": obj.shape,
                                      "dtype": str(obj.dtype)})
                    elif isinstance(obj, h5py.Group):
                        nodes.append({"name": name, "kind": "group"})
                handle.visititems(visitor)
            return {"format": "HDF5/NIX", "first_nodes": nodes,
                    "nodes_capped_at": max_nodes}
        if ext == ".npz":
            import zipfile
            with zipfile.ZipFile(str(path), "r") as zf:
                names = zf.namelist()
            return {"format": "NPZ", "members": names[:max_nodes], "total_members": len(names)}
        if ext == ".npy":
            import numpy as np
            obj = np.load(str(path), mmap_mode="r", allow_pickle=False)
            return {"format": "NPY", "shape": obj.shape, "dtype": str(obj.dtype)}
        if ext in OTHER_EXT:
            # Text preview is intentionally capped and disabled for unknown binaries.
            if ext == ".txt" or ext in {".csv", ".tsv", ".json"}:
                with path.open("rb") as f:
                    head = f.read(512)
                return {"format": "text (preview)", "first_bytes": head.decode("utf-8", "replace")}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"format": "metadata not inspected"}


def inspect(root: Path, max_depth: int, max_files: int,
            max_entries_per_dir: int, metadata_files: int, focus: str, max_dirs: int):
    directories, files, truncated = _walk(root, max_depth, max_files, max_entries_per_dir, max_dirs)
    extension_counts = collections.Counter(f["suffix"] or "<no extension>" for f in files)
    focus_items = [d["path"] for d in directories if focus.lower() in d["path"].lower()]
    focus_items += [f["path"] for f in files if focus.lower() in f["path"].lower()]
    probe_candidates = [f for f in files if f["suffix"] in HDF5_EXT | OTHER_EXT]
    # Probe focus first; never repeatedly decompress every NPZ.
    probe_candidates.sort(key=lambda f: (focus.lower() not in f["path"].lower(), f["path"]))
    probes = {}
    for item in probe_candidates[:metadata_files]:
        probes[item["path"]] = _probe_metadata(root / item["path"])
    return {
        "root": str(root), "limits": {
            "max_depth": max_depth, "max_files": max_files,
            "max_entries_per_dir": max_entries_per_dir, "metadata_files": metadata_files,
            "max_dirs": max_dirs,
        },
        "directories": directories,
        "files": files,
        "extension_counts_for_listed_files": dict(extension_counts),
        "matches_for_focus": focus_items,
        "file_metadata": probes,
        "warnings": truncated,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/source/tvsd"))
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--max-files", type=int, default=200)
    p.add_argument("--max-dirs", type=int, default=400)
    p.add_argument("--max-entries-per-dir", type=int, default=120)
    p.add_argument("--metadata-files", type=int, default=12)
    p.add_argument("--focus", default="spikes_filtered")
    p.add_argument("--report", type=Path, default=Path("/project/data/reports/tvsd_inventory.json"))
    args = p.parse_args()
    root = args.root.resolve(strict=True)
    if not root.is_dir():
        p.error(f"Not a directory: {root}")
    info = inspect(root, args.max_depth, args.max_files,
                   args.max_entries_per_dir, args.metadata_files, args.focus, args.max_dirs)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str) + "\n")
    print(f"Root: {root}")
    print(f"Visited {len(info['directories'])} directories, listed {len(info['files'])} files")
    print("Extensions:", info["extension_counts_for_listed_files"])
    print("Matches for", repr(args.focus) + ":")
    for hit in info["matches_for_focus"][:50]:
        print("  ", hit)
    print("Format probes:")
    for name, details in info["file_metadata"].items():
        summary = str(details)[:300]
        print("  ", name, summary)
    print("Warnings:", len(info["warnings"]))
    for warning in info["warnings"][:10]:
        print("  ", warning)
    print("JSON report:", args.report)


if __name__ == "__main__":
    main()
