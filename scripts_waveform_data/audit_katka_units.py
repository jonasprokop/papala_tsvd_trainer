#!/usr/bin/env python3
"""
Per-unit audit of Katka's NATIM pipeline: for every sorted unit, which
electrode it sits on, and where (and why) it was dropped.

The pipeline, as written in her code:

  spikes_KS4/*_spikes_KS4_unfiltered.nix   every sorted unit (Electrode_ID set by sorting)
     |  SUA_filter_metrics_pkl_create.py   -> dataframes/filter_metrics_NATIM/*.pkl
     |                                        (a unit whose metrics crash gets NO row)
     |  SUA_write_filtered_nix.py          thresholds, NaN = fail, no row = dropped
  spikes_filtered/*_spikes_KS4_filtered.nix
     |  SUA_NATIM_prop_pkl_create.py       channel_order from the RS tMUA file
     |                                        (-1 = dropped); a crash on unit k drops
     |                                        units k..end of that array
  sua_prop_all_NATIM/*.pkl                 -> your exported label CSVs

Only NIX metadata is read (via nixio), never spike times or waveforms, so
this is fast even though the unfiltered files hold ~1.5 GB of waveforms each.

Outputs (in --out):
  unit_audit.csv           one row per sorted unit, with status + reason
  array_summary.csv        counts per recording x array x status
  channel_order_check.csv  RS-tMUA electrode order vs the NATIM RB file's order
"""

import argparse
import ast
import pickle
from pathlib import Path

import nixio
import numpy as np
import pandas as pd
import yaml

PARAMS = "/CSNG/studekat/ripple_paper_clean_copy/code_new_filter/params_analysis.yml"
DF_FOLDER = "/CSNG/studekat/ripple_paper_clean_copy/dataframes"
LABELS = "data/reports/katka_labels"
OUT = "data/reports/katka_unit_audit"

V1_ARRAYS = {"N": {3, 4, 6, 7, 8}, "F": {1, 2, 3, 4, 5, 6, 7, 8}}

# copied into the audit so the "no obvious reason" drops can be explained
METRIC_CONTEXT = ["n_spikes", "FR_computed", "enough_spikes", "has_waveforms",
                  "coinc_ratio_worst", "coinc_partner_name", "coinc_mutual"]


# ---------------------------------------------------------------- NIX metadata

def _value(prop):
    v = list(prop.values)
    if len(v) == 0:
        return None
    v = v[0] if len(v) == 1 else v
    if isinstance(v, str) and v.strip().lower() == "nan":
        return np.nan
    return v


def _props(obj):
    if obj.metadata is None:
        return {}
    return {p.name: _value(p) for p in obj.metadata.inherited_properties()}


def read_units(path):
    """Spiketrain annotations in the order neo returns them (= train_order),
    plus the block annotations. None if the file doesn't exist."""
    if path is None or not Path(path).is_file():
        return None, None
    f = nixio.File.open(str(path), nixio.FileMode.ReadOnly)
    try:
        blk = f.blocks[0]
        seg = next(g for g in blk.groups if g.type == "neo.segment")
        units = []
        for mt in seg.multi_tags:
            if mt.type != "neo.spiketrain":
                continue
            a = _props(mt)
            a.setdefault("nix_name", mt.name)  # neo: metadata wins over object name
            units.append(a)
        return units, _props(blk)
    finally:
        f.close()


def read_signal_array_annotations(path):
    """Array annotations of the first analogsignal (per-channel metadata)."""
    if not Path(path).is_file():
        return None
    f = nixio.File.open(str(path), nixio.FileMode.ReadOnly)
    try:
        seg = next(g for g in f.blocks[0].groups if g.type == "neo.segment")
        da = next(d for d in seg.data_arrays if d.type == "neo.analogsignal")
        if da.metadata is None:
            return {}
        return {p.name: list(p.values) for p in da.metadata.inherited_properties()
                if p.type == "ARRAYANNOTATION"}
    finally:
        f.close()


# ---------------------------------------------------------------- paths

def natim_path(folder, monkey, date, array, kind):
    stem = f"macaque{monkey}_TVSD_{date}"
    sub, suf = {"all": ("spikes_KS4", "spikes_KS4_unfiltered"),
                "good": ("spikes_KS4", "spikes_KS4"),
                "filtered": ("spikes_filtered", "spikes_KS4_filtered"),
                "RB": ("RB", "RB")}[kind]
    return f"{folder}/{stem}/{sub}/{stem}_Array{array}_{suf}.nix"


def tmua_path(folder, monkey, rs_date, array):
    stem = f"macaque{monkey}_RS_{rs_date}"
    return f"{folder}/{stem}/tMUA/{stem}_Array{array}_tMUA.nix"


# ---------------------------------------------------------------- filter rule

def parse_criteria(block_ann, fallback):
    """The rule actually used is stored in each filtered file's block
    annotation 'filter_criteria' ({col: 'below 0.1'}); thresholds.pkl may
    have changed since."""
    s = (block_ann or {}).get("filter_criteria")
    if isinstance(s, str):
        try:
            d = ast.literal_eval(s)
            return {c: (r.split()[0], float(r.split()[1])) for c, r in d.items()}, "filtered_nix"
        except Exception:
            pass
    return fallback, "thresholds.pkl"


def evaluate(row, criteria, nan_fails=True):
    failed = []
    for col, (direc, th) in criteria.items():
        v = row.get(col, np.nan)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = np.nan
        if not np.isfinite(v):
            if nan_fails:
                failed.append(f"{col}=NaN")
            continue
        ok = v <= th if direc == "below" else v >= th
        if not ok:
            failed.append(f"{col}={v:.3g} (keep {direc} {th:g})")
    return failed


# ---------------------------------------------------------------- one recording

def audit_recording(monkey, date, params, df_folder, labels_dir, crit_fallback, nan_fails):
    natim = params["natim_data_folder"]
    rs_date = params["dates"][monkey]["RS"][0]
    name = f"monkey{monkey}_all_arrays_date_{date}"
    notes = []

    met_path = Path(df_folder) / "filter_metrics_NATIM" / f"{name}.pkl"
    try:
        metrics = pd.read_pickle(met_path)
        met_by_key = {(int(r["array"]), r["cell_name"]): r for _, r in metrics.iterrows()}
    except Exception as e:
        notes.append(f"metrics pickle unreadable: {e!r}")
        met_by_key = None

    label_csv = Path(labels_dir) / f"{name}_v1_labels.csv"
    labels = pd.read_csv(label_csv) if label_csv.is_file() else None
    lab_by_key = ({(int(r["array"]), r["cell_name"]): r for _, r in labels.iterrows()}
                  if labels is not None else None)

    rows, chk = [], []
    for array in range(1, 17):
        units, _ = read_units(natim_path(natim, monkey, date, array, "all"))
        if units is None:
            continue
        good, _ = read_units(natim_path(natim, monkey, date, array, "good"))
        good_names = {u["nix_name"] for u in good} if good else set()
        filt, filt_ann = read_units(natim_path(natim, monkey, date, array, "filtered"))
        filt_by_orig = {}
        if filt:
            for k, u in enumerate(filt):
                filt_by_orig[int(u.get("train_order_original", -1))] = (k, u)
        criteria, crit_src = parse_criteria(filt_ann, crit_fallback)
        if (filt_ann or {}).get("filter_resolve_mutual_pairs"):
            notes.append(f"array {array}: written with resolve_mutual_pairs=True; "
                         "recomputed pass/fail ignores that step")

        # channel_order exactly as aux_electrodeID_to_ch_order: last match wins
        tmua, _ = read_units(tmua_path(params["data_folder"], monkey, rs_date, array))
        ch_map = ({tmua[i].get("Electrode_ID"): i for i in range(min(64, len(tmua)))}
                  if tmua else {})
        tmua_ids = [tmua[i].get("Electrode_ID") for i in range(min(64, len(tmua)))] if tmua else []

        rb = read_signal_array_annotations(natim_path(natim, monkey, date, array, "RB"))
        key = next((k for k in (rb or {}) if "electrode" in k.lower()), None)
        chk.append({
            "monkey": monkey, "date": date, "array": array, "rs_date_for_mapping": rs_date,
            "tmua_found": bool(tmua), "rb_found": rb is not None,
            "rb_array_annotation_keys": ",".join(rb) if rb else "",
            "rb_electrode_key": key or "",
            "order_identical": (None if not (key and tmua_ids)
                                else [str(x) for x in rb[key]] == [str(x) for x in tmua_ids]),
        })

        for order, u in enumerate(units):
            cell = u["nix_name"]
            eid = u.get("Electrode_ID")
            r = {
                "monkey": f"monkey{monkey}", "date": date, "array": array,
                "train_order_original": order, "cell_name": cell,
                "Electrode_ID": eid, "Area": u.get("Area"),
                "KSLabel": u.get("KSLabel"), "Label": u.get("Label"),
                "channel_order": ch_map.get(eid, -1) if ch_map else np.nan,
                "in_good_units_file": cell in good_names,
            }
            m = met_by_key.get((array, cell)) if met_by_key is not None else None
            r["in_metrics"] = (m is not None) if met_by_key is not None else np.nan
            for c in METRIC_CONTEXT:
                r[c] = m.get(c, np.nan) if m is not None else np.nan
            failed = evaluate(m, criteria, nan_fails) if m is not None else []
            r["pass_filter_recomputed"] = (not failed) if m is not None else np.nan
            r["failed_criteria"] = "; ".join(failed)
            r["criteria_source"] = crit_src
            fk = filt_by_orig.get(order)
            if fk and fk[1].get("nix_name") != cell:
                notes.append(f"array {array}: train_order_original {order} name mismatch")
            r["in_filtered_nix"] = fk is not None if filt is not None else np.nan
            r["train_order"] = fk[0] if fk else np.nan
            lab = lab_by_key.get((array, cell)) if lab_by_key is not None else None
            r["in_label_csv"] = (lab is not None) if lab_by_key is not None else np.nan
            r["final_class"] = lab["final_class"] if lab is not None else np.nan
            r["status"], r["reason"] = classify(r, monkey, filt is not None,
                                                met_by_key is not None, lab_by_key is not None)
            rows.append(r)

    df = pd.DataFrame(rows)
    if not df.empty and lab_by_key is not None:
        mark_tail_drops(df)
    if labels is not None and not df.empty:
        known = set(zip(df["array"], df["cell_name"]))
        orphans = [k for k in lab_by_key if k not in known]
        if orphans:
            notes.append(f"{len(orphans)} label-CSV units not in the unfiltered files "
                         f"(different unit set?), e.g. {orphans[:3]}")
    return df, chk, notes


def classify(r, monkey, have_filt, have_met, have_lab):
    if have_met and not r["in_metrics"]:
        return "dropped_no_metrics", "no filter-metrics row: metric computation crashed for this unit/array"
    if have_met and not r["pass_filter_recomputed"]:
        return "dropped_filter", r["failed_criteria"]
    if not have_filt:
        return "unknown_no_filtered_file", "filtered NIX for this array missing"
    if not r["in_filtered_nix"]:
        if have_met:
            return "dropped_not_written", "passes the rule but absent from filtered NIX (rule changed after writing?)"
        return "dropped_filter_unknown_reason", "absent from filtered NIX; metrics pickle unreadable"
    if r["channel_order"] == -1:
        return "dropped_unmapped_electrode", "Electrode_ID not in the RS tMUA file -> channel_order -1"
    if r["Area"] != "V1" or r["array"] not in V1_ARRAYS[monkey]:
        return "outside_V1_selection", ""
    if not have_lab:
        return "reached_prop_stage_unknown", "no label CSV for this recording"
    if r["in_label_csv"]:
        return "kept", ""
    return "dropped_prop_stage", "in filtered NIX and mapped, but missing from sua_prop_all_NATIM"


def mark_tail_drops(df):
    """Her prop loop has one try/except per array: an error on unit k silently
    drops units k..end. Flag drops that form exactly such a tail."""
    for _, g in df[df["in_filtered_nix"] == True].groupby("array"):
        g = g.sort_values("train_order")
        miss = g["status"].eq("dropped_prop_stage").values
        if miss.any():
            first = int(np.argmax(miss))
            in_v1 = g["status"].isin(["kept", "dropped_prop_stage"]).values
            if miss[first:][in_v1[first:]].all():
                idx = g.index[first:][g["status"].values[first:] == "dropped_prop_stage"]
                df.loc[idx, "reason"] = (f"array loop crashed at filtered unit "
                                         f"{int(g['train_order'].iloc[first])}; "
                                         "every later unit on this array lost")


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("monkey", nargs="?", choices=["N", "F"])
    p.add_argument("date", nargs="?")
    p.add_argument("--params", default=PARAMS)
    p.add_argument("--df-folder", default=DF_FOLDER,
                   help="folder holding filter_metrics_NATIM/ and filter_metrics_chosen/")
    p.add_argument("--labels", default=LABELS)
    p.add_argument("--out", default=OUT)
    a = p.parse_args()

    params = yaml.safe_load(Path(a.params).read_text())
    try:
        with open(Path(a.df_folder) / "filter_metrics_chosen" / "thresholds.pkl", "rb") as f:
            t = pickle.load(f)
        crit_fallback, nan_fails = t["criteria"], t.get("nan_fails", True)
        print("thresholds.pkl:", crit_fallback, "nan_fails:", nan_fails)
    except Exception as e:
        print(f"thresholds.pkl unreadable ({e!r}); relying on filtered-NIX annotations")
        crit_fallback, nan_fails = {}, True

    jobs = ([(a.monkey, a.date)] if a.monkey else
            [(m, d) for m in ["N", "F"] for d in params["dates"][m]["NATIM"]])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames, checks = [], []
    for monkey, date in jobs:
        try:
            df, chk, notes = audit_recording(monkey, date, params, a.df_folder,
                                             a.labels, crit_fallback, nan_fails)
        except Exception as e:
            print(f"FAILED {monkey} {date}: {e!r}")
            continue
        frames.append(df)
        checks += chk
        counts = df["status"].value_counts().to_dict() if not df.empty else {}
        print(f"{monkey} {date}: {len(df)} sorted units  {counts}")
        for n in dict.fromkeys(notes):
            print("   NOTE:", n)

    audit = pd.concat(frames, ignore_index=True)
    audit.to_csv(out / "unit_audit.csv", index=False)
    (audit.groupby(["monkey", "date", "array", "status"]).size()
          .unstack(fill_value=0).reset_index()
          .to_csv(out / "array_summary.csv", index=False))
    pd.DataFrame(checks).to_csv(out / "channel_order_check.csv", index=False)

    print("\nOverall:")
    print(audit["status"].value_counts().to_string())
    fc = audit.loc[audit.status == "dropped_filter", "failed_criteria"].str.split("; ").explode()
    if not fc.empty:
        print("\nFailed criteria (a unit can fail several):")
        print(fc.str.split(r"[=(]").str[0].value_counts().to_string())
        print("  of which NaN:", int(fc.str.endswith("=NaN").sum()))
    print(f"\nWritten to {out}/")


if __name__ == "__main__":
    main()
