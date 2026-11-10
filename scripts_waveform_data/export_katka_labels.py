#!/usr/bin/env python3

import gc
gc.disable()  # One pickle per process; avoids the observed GC crash.

import argparse
import re
from pathlib import Path

import pickle

import pandas as pd

# CHANGED: renamed to DEFAULT_SOURCE; can be overridden with --source
DEFAULT_SOURCE = Path(
    "/CSNG/studekat/ripple_paper_clean_submit/"
    "dataframes/sua_prop_all_NATIM"
)

OUTPUT = Path("data/reports/katka_labels")

V1_ARRAYS = {
    "monkeyN": {3, 4, 6, 7, 8},
    "monkeyF": {1, 2, 3, 4, 5, 6, 7, 8},
}

REQUIRED = [
    "cell_name",
    "array",
    "area",
    "channel_order",
    "train_order",
    "final_class",
]

OPTIONAL = [
    "width_wf_class",
    "wf_direction",
    "is_RB_phase_selective",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)  # NEW
    args = parser.parse_args()

    path = args.source / Path(args.filename).name  # CHANGED

    match = re.fullmatch(
        r"(monkey[FN])_all_arrays_date_(\d{8}_B\d+)\.pkl",
        path.name,
    )
    if match is None:
        raise ValueError(f"Unexpected filename: {path.name}")

    animal, recording = match.groups()
    session = f"{animal.replace('monkey', 'macaque')}_TVSD_{recording}"

    print(f"Loading {path}", fluspython scripts/propose_ks4_tracks.pyh=True)  # prints full path, so you see which dir
    with path.open("rb") as f:
        df = pickle.load(f)
    print(f"Loaded {len(df)} units", flush=True)

    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    selected = df.loc[
        df["area"].eq("V1")
        & df["array"].isin(V1_ARRAYS[animal]),
        REQUIRED + [c for c in OPTIONAL if c in df.columns],
    ].copy()

    selected.insert(0, "source_file", str(path))  # NEW: provenance
    selected.insert(0, "session", session)
    selected.insert(0, "animal", animal)

    if selected.empty:
        raise ValueError(f"No V1 units found: {session}")

    if selected[["cell_name", "array", "final_class"]].isna().any().any():
        raise ValueError(f"Missing identifiers/classes: {session}")

    if selected.duplicated(["array", "cell_name"]).any():
        raise ValueError(f"Duplicate unit identifiers: {session}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / f"{path.stem}_v1_labels.csv"
    temporary = destination.with_suffix(".csv.tmp")

    selected.to_csv(temporary, index=False)
    temporary.replace(destination)

    print(f"V1 units: {len(selected)}")
    print(selected["final_class"].value_counts().to_string())
    print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()