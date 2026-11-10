
from pathlib import Path
import re

import h5py
import pandas as pd

REPORTS = Path("/project/data/reports")
MANIFEST = REPORTS / "tvsd_manifest.csv"
LABEL_DIR = REPORTS / "katka_labels"
OUTPUT = REPORTS / "katka_ks4_v1_join.csv"


def decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


# Load Katka's per-recording labels.
label_tables = []

for path in sorted(LABEL_DIR.glob("*_v1_labels.csv")):
    match = re.fullmatch(
        r"monkey([FN])_all_arrays_date_(\d{8}_B\d+)_v1_labels",
        path.stem,
    )
    if match is None:
        raise ValueError(f"Unexpected label filename: {path.name}")

    session = f"macaque{match.group(1)}_TVSD_{match.group(2)}"

    df = pd.read_csv(path)
    df = df.loc[df["area"] == "V1",
                ["array", "cell_name", "final_class"]].copy()
    df.insert(0, "session", session)
    label_tables.append(df)

if not label_tables:
    raise RuntimeError("No Katka label CSVs found")

labels = pd.concat(label_tables, ignore_index=True)

if labels.duplicated(["session", "array", "cell_name"]).any():
    raise ValueError("Duplicate Katka unit identifiers")


# Check that every session has a label export.
manifest = pd.read_csv(MANIFEST)
manifest = manifest.loc[manifest["area"] == "V1"].copy()

expected = set(manifest["session"])
available = set(labels["session"])
missing = expected - available

if missing:
    raise RuntimeError(
        "Missing label exports:\n" + "\n".join(sorted(missing))
    )


# Read all strict-V1 KS4 unit identities.
records = []

for row in manifest.itertuples(index=False):
    with h5py.File(row.source_path, "r") as f:
        block = next(
            obj for key, obj in f["metadata"].items()
            if key.startswith("neo.block.")
        )
        segment = next(
            obj for key, obj in block["sections"].items()
            if key.startswith("neo.segment.")
        )

        for name, unit in segment["sections"].items():
            if not name.startswith("neo.spiketrain."):
                continue

            p = unit["properties"]

            records.append({
                "animal": row.animal,
                "session": row.session,
                "array": int(row.array),
                "electrode_id": int(p["Electrode_ID"][()][0]),
                "nix_name": decode(p["nix_name"][()][0]),
                "firing_rate": float(p["firing_rate"][()][0]),
            })

ks4 = pd.DataFrame(records)

if ks4.duplicated(["session", "array", "nix_name"]).any():
    raise ValueError("Duplicate KS4 unit identifiers")


# Exact join: never substitute an approximate match here.
joined = ks4.merge(
    labels,
    how="left",
    left_on=["session", "array", "nix_name"],
    right_on=["session", "array", "cell_name"],
    validate="one_to_one",
    indicator=True,
)

joined["has_katka_class"] = joined["_merge"].eq("both")
joined = joined.drop(columns=["cell_name", "_merge"])

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
joined.to_csv(OUTPUT, index=False)


print("\n=== ALL STRICT V1 UNITS ===")
print("Sessions:", joined["session"].nunique())
print("KS4 units:", len(joined))
print("Exact Katka matches:", joined["has_katka_class"].sum())
print("Coverage: {:.1%}".format(joined["has_katka_class"].mean()))

print("\n=== BY ANIMAL ===")
for animal, group in joined.groupby("animal"):
    print(
        "{}: {}/{} ({:.1%})".format(
            animal,
            group["has_katka_class"].sum(),
            len(group),
            group["has_katka_class"].mean(),
        )
    )

print("\n=== WAVEFORM CLASSES ===")
print(joined["final_class"].value_counts(dropna=False).to_string())

print("\nSaved:", OUTPUT)
