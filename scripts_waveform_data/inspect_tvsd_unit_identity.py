from pathlib import Path
import h5py
import numpy as np

ROOT = Path("/source/tvsd")

# Same electrode arrays, across blocks and recording days.
SESSIONS = [
    ("macaqueN_TVSD_20220111_B1", 3),
    ("macaqueN_TVSD_20220111_B2", 3),
    ("macaqueN_TVSD_20220112_B1", 3),
    ("macaqueF_TVSD_20240112_B1", 1),
    ("macaqueF_TVSD_20240112_B2", 1),
    ("macaqueF_TVSD_20240115_B1", 1),
]


def preview(dataset):
    """Read only small metadata properties, never full waveforms."""
    if dataset.size > 8 or dataset.nbytes > 2048:
        return f"<shape={dataset.shape}, dtype={dataset.dtype}>"

    value = np.asarray(dataset[()]).tolist()

    def decode(v):
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="replace")
        if isinstance(v, list):
            return [decode(x) for x in v]
        return v

    return repr(decode(value))[:200]


for session_name, array in SESSIONS:
    filename = f"{session_name}_Array{array}_spikes_KS4_filtered.nix"
    path = ROOT / session_name / "spikes_filtered" / filename

    print(f"\n{'=' * 70}")
    print(f"SESSION: {session_name} / ARRAY {array}")

    if not path.exists():
        print("MISSING:", path)
        continue

    with h5py.File(path, "r") as f:
        metadata = f["metadata"]

        block = next(
            v for k, v in metadata.items()
            if k.startswith("neo.block.")
        )
        segment = next(
            v for k, v in block["sections"].items()
            if k.startswith("neo.segment.")
        )

        units = [
            (name, obj)
            for name, obj in segment["sections"].items()
            if name.startswith("neo.spiketrain.")
        ]

        print("Number of units:", len(units))

        all_fields = sorted({
            field
            for _, unit in units
            for field in unit["properties"].keys()
        })

        print("AVAILABLE ANNOTATIONS:")
        print(", ".join(all_fields))

        for name, unit in units[:3]:
            print(f"\nUNIT: {name}")

            for field, dataset in unit["properties"].items():
                print(f"  {field}: {preview(dataset)}")

            print("  NIX attributes:",
                  {k: str(v)[:100] for k, v in unit.attrs.items()})
