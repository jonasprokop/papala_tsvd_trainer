from pathlib import Path
import h5py
import numpy as np

ROOT = Path("/source/tvsd")


def load_units(session, array, electrode):
    path = (ROOT / session / "spikes_filtered" /
            f"{session}_Array{array}_spikes_KS4_filtered.nix")

    units = []

    with h5py.File(path, "r") as f:
        block = next(
            v for k, v in f["metadata"].items()
            if k.startswith("neo.block.")
        )
        segment = next(
            v for k, v in block["sections"].items()
            if k.startswith("neo.segment.")
        )

        for unit in segment["sections"].values():
            p = unit["properties"]

            if int(p["Electrode_ID"][()][0]) != electrode:
                continue

            units.append({
                "name": p["neo_name"][()][0].decode(),
                "rate": float(p["firing_rate"][()][0]),
                "waveform": np.asarray(
                    p["avg_wf_zscored"][()], dtype=float
                ).ravel(),
            })

    return units


comparisons = [
    ("macaqueN_TVSD_20220111_B1",
     "macaqueN_TVSD_20220111_B2", 3, 161),
    ("macaqueN_TVSD_20220111_B1",
     "macaqueN_TVSD_20220112_B1", 3, 161),
    ("macaqueN_TVSD_20220111_B1",
     "macaqueN_TVSD_20220111_B2", 3, 162),
    ("macaqueF_TVSD_20240112_B1",
     "macaqueF_TVSD_20240112_B2", 1, 34),
]

for first, second, array, electrode in comparisons:
    a = load_units(first, array, electrode)
    b = load_units(second, array, electrode)

    print(f"\n{first} -> {second}")
    print(f"Array {array}, electrode {electrode}")

    for x in a:
        for y in b:
            corr = np.corrcoef(
                x["waveform"], y["waveform"]
            )[0, 1]

            print(
                f"  {x['name']} ({x['rate']:.1f} Hz)"
                f" -> {y['name']} ({y['rate']:.1f} Hz)"
                f"  waveform r={corr:.4f}"
            )
