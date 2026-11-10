
from pathlib import Path
import pandas as pd
import h5py

ROOT = Path("/source/tvsd")
LABELS = Path("/project/data/reports/katka_labels")

CASES = [
    (
        "macaqueN_TVSD_20220111_B10",
        "monkeyN_all_arrays_date_20220111_B10_v1_labels.csv",
    ),
    (
        "macaqueF_TVSD_20240112_B4",
        "monkeyF_all_arrays_date_20240112_B4_v1_labels.csv",
    ),
]


def decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


for session, csv_name in CASES:
    labels = pd.read_csv(LABELS / csv_name)
    print(f"\n{'=' * 65}")
    print(session)
    print(f"{'Array':>5} {'Katka':>8} {'KS4':>8} {'Matched':>9}")

    total_katka = total_ks4 = total_matched = 0

    for array in sorted(labels["array"].unique()):
        file = (
            ROOT / session / "spikes_filtered"
            / f"{session}_Array{array}_spikes_KS4_filtered.nix"
        )

        ks4_ids = set()

        with h5py.File(file, "r") as f:
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
                ks4_ids.add(decode(p["nix_name"][()][0]))

        katka_ids = set(
            labels.loc[labels["array"] == array, "cell_name"]
        )

        matched = len(katka_ids & ks4_ids)

        print(
            f"{array:5d} {len(katka_ids):8d} "
            f"{len(ks4_ids):8d} {matched:9d}"
        )

        total_katka += len(katka_ids)
        total_ks4 += len(ks4_ids)
        total_matched += matched

    print(
        f"TOTAL {total_katka} Katka, "
        f"{total_ks4} KS4, {total_matched} exact matches"
    )
