#!/usr/bin/env python3
"""
Rebuild Katka's NATIM unit labels (code_new_filter version) straight from the
filtered NIX files - no pickles, a few hundred MB of RAM.

Replicates SUA_NATIM_prop_pkl_create.py (new filter) + functions_analysis.py:
  unit list      spikes_filtered/*_spikes_KS4_filtered.nix   (her filter's output)
  avg waveform   the unit's 'avg_wf' annotation
  channel_order  row of Electrode_ID in the RS tMUA file of dates[M]['RS'][0]
                 (-1 = unmapped; Katka deletes those -> katka_kept=False)
  classes        aux_add_waveform_prop / width / up_down / final_classes,
                 peak = max before trough (first_peak_height_from_waveform)

NOT produced: is_RB_phase_selective, ch_is_noisy_* (need the ripple-band
spike-phase computation).

Every unit in the filtered files gets a row, all areas. Filter on
`in_v1_selection` / `katka_kept` downstream.

Usage:
    python rebuild_katka_labels.py                 # all 45 recordings
    python rebuild_katka_labels.py N 20220112_B1   # one recording
"""

import argparse
from pathlib import Path

import nixio
import numpy as np
import pandas as pd
import yaml
from scipy.stats import zscore

PARAMS = "/CSNG/studekat/ripple_paper_clean_copy/code_new_filter/params_analysis.yml"
OUT = "data/reports/katka_labels_rebuilt"

# your previous exporter's selection; Katka's params mark N arrays 1-8 as V1
V1_ARRAYS = {"N": {3, 4, 6, 7, 8}, "F": {1, 2, 3, 4, 5, 6, 7, 8}}


# ---------------------------------------------------------------- NIX metadata

def _value(prop):
    v = list(prop.values)
    if len(v) == 0:
        return None
    v = v[0] if len(v) == 1 else v
    if isinstance(v, str) and v.strip().lower() == "nan":
        return np.nan
    return v


def read_units(path):
    """Spiketrain annotations in neo's order (= Katka's train_order)."""
    f = nixio.File.open(str(path), nixio.FileMode.ReadOnly)
    try:
        seg = next(g for g in f.blocks[0].groups if g.type == "neo.segment")
        units = []
        for mt in seg.multi_tags:
            if mt.type != "neo.spiketrain":
                continue
            a = ({p.name: _value(p) for p in mt.metadata.inherited_properties()}
                 if mt.metadata is not None else {})
            a.setdefault("nix_name", mt.name)
            units.append(a)
        return units
    finally:
        f.close()


# ---------------------------------------------------------------- classifier

def width_class(width, intervals):
    for name, (lo, hi) in zip(["narrow", "medium", "wide"], intervals):
        if lo <= width <= hi:
            return name
    raise ValueError(f"width_wf={width} outside width_intervals")


def classify(wf, intervals, peak_th):
    wf = np.asarray(wf, dtype=float)
    if wf.ndim != 1 or wf.size < 3 or not np.isfinite(wf).all():
        raise ValueError("no usable avg_wf")
    trough = int(np.argmin(wf))
    width = int(np.argmax(wf[trough:]))            # trough -> following max
    wf_z = zscore(wf)
    direction = "UP" if abs(np.max(wf)) > abs(np.min(wf)) else "DOWN"
    wclass = width_class(width, intervals)
    if direction == "UP":
        final = "UP"
    elif wclass == "wide":
        final = "DOWN_wide"
    else:
        tz = int(np.argmin(wf_z))
        if tz == 0:
            raise ValueError("trough at sample 0")
        peak = wf_z[np.argmax(wf_z[:tz])]
        final = (f"DOWN_{wclass}_shallow" if peak < peak_th else
                 f"DOWN_{wclass}_sharp" if peak >= peak_th else "NO_CLASS")
    return {"width_wf": width, "width_wf_class": wclass,
            "wf_direction": direction, "final_class": final}


# ---------------------------------------------------------------- one recording

def rebuild(monkey, date, params):
    natim, rs = params["natim_data_folder"], params["data_folder"]
    rs_date = params["dates"][monkey]["RS"][0]
    stem = f"macaque{monkey}_TVSD_{date}"
    rows, notes = [], []

    for array in range(1, 17):
        path = Path(f"{natim}/{stem}/spikes_filtered/{stem}_Array{array}_spikes_KS4_filtered.nix")
        if not path.is_file():
            continue
        try:
            units = read_units(path)
        except Exception as e:
            notes.append(f"array {array}: spike file unreadable ({e!r}) - skipped, as Katka's code would")
            continue
        if not units:
            continue

        tpath = f"{rs}/macaque{monkey}_RS_{rs_date}/tMUA/macaque{monkey}_RS_{rs_date}_Array{array}_tMUA.nix"
        try:
            tmua = read_units(tpath)
            ch_map = {tmua[i].get("Electrode_ID"): i for i in range(min(64, len(tmua)))}
        except Exception as e:
            notes.append(f"array {array}: RS tMUA unreadable ({e!r}) - Katka's code drops this array")
            continue

        for order, u in enumerate(units):
            eid = u.get("Electrode_ID")
            ch = ch_map.get(eid, -1)
            area = u.get("Area")
            r = {
                "animal": f"monkey{monkey}", "session": stem, "date": date,
                "array": array, "area": area, "cell_name": u["nix_name"],
                "Electrode_ID": eid, "channel_order": ch,
                "train_order": order,
                "train_order_original": u.get("train_order_original"),
                "n_spikes": u.get("flt_n_spikes"), "FR": u.get("flt_FR_computed"),
                "katka_kept": ch >= 0,
                "in_v1_selection": area == "V1" and array in V1_ARRAYS[monkey],
                "source_file": str(path),
            }
            try:
                r.update(classify(u.get("avg_wf"), params["width_intervals"],
                                  params["first_peak_height"]))
                r["classify_error"] = ""
            except Exception as e:
                r.update(width_wf=np.nan, width_wf_class=np.nan,
                         wf_direction=np.nan, final_class=np.nan)
                r["classify_error"] = str(e)
            rows.append(r)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no units found")
    if df.duplicated(["array", "cell_name"]).any():
        raise ValueError("duplicate (array, cell_name)")
    n_err = int((df["classify_error"] != "").sum())
    n_err_kept = int(((df["classify_error"] != "") & df["katka_kept"]).sum())
    if n_err_kept:
        notes.append(f"{n_err_kept} kept units could not be classified "
                     "(Katka's run would have failed this recording) - see classify_error")
    elif n_err:
        notes.append(f"{n_err} unmapped units unclassifiable (irrelevant, Katka drops them)")
    return df, notes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("monkey", nargs="?", choices=["N", "F"])
    p.add_argument("date", nargs="?")
    p.add_argument("--params", default=PARAMS)
    p.add_argument("--out", default=OUT)
    a = p.parse_args()

    params = yaml.safe_load(Path(a.params).read_text())
    jobs = ([(a.monkey, a.date)] if a.monkey else
            [(m, d) for m in ["N", "F"] for d in params["dates"][m]["NATIM"]])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    frames, failed = [], []
    for monkey, date in jobs:
        name = f"monkey{monkey}_all_arrays_date_{date}"
        try:
            df, notes = rebuild(monkey, date, params)
        except Exception as e:
            print(f"FAILED {name}: {e!r}", flush=True)
            failed.append(name)
            continue
        dest = out / f"{name}_labels.csv"
        tmp = dest.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(dest)
        frames.append(df)
        sel = df[df.katka_kept & df.in_v1_selection]
        print(f"OK {name}: {len(df)} units, {int(df.katka_kept.sum())} kept by Katka, "
              f"{len(sel)} in your V1 selection", flush=True)
        for n in notes:
            print("   NOTE:", n)

    if frames:
        allu = pd.concat(frames, ignore_index=True)
        allu.to_csv(out / "all_units.csv", index=False)
        sel = allu[allu.katka_kept & allu.in_v1_selection]
        print(f"\n{len(jobs) - len(failed)}/{len(jobs)} recordings -> {out}/")
        print(f"{len(allu)} units total, {len(sel)} kept + in your V1 selection")
        print(sel["final_class"].value_counts(dropna=False).to_string())
    if failed:
        print("Failed:", *failed, sep="\n  ")


if __name__ == "__main__":
    main()
