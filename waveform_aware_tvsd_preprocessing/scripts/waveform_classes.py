#!/usr/bin/env python3
"""Waveform-class annotations for extracted TVSD KS4 unit instances.

Classifications are computed from per-session raw average waveforms using
width_intervals and first_peak_height from a YAML parameter file.
Annotations never affect neuron matching, quality filtering, or selection.

Usage:
  python scripts/waveform_classes.py classify --root ROOT --params params_analysis.yml
  python scripts/waveform_classes.py summarize --root ROOT
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import zscore

ROOT = Path('/project/data/processed/tvsd_waveform_aware_v1')
TRACK_ID = 'candidate_joint_track_id'
INVENTORY_KEYS = ['session', 'array', 'nix_name']

# The classification rules are intentionally unchanged; thresholds come from YAML.

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



def _atomic_csv(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + '.tmp')
    try:
        df.to_csv(temp, index=False)
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)



def classify_inventory(root: Path, params_path: Path) -> dict:
    """Annotate all extracted units without modifying their source artifacts."""
    root, params_path = Path(root), Path(params_path)
    inv_path = root / 'unit_instances.csv'
    inv = pd.read_csv(inv_path, low_memory=False)
    required = {'unit_instance_id', 'session', 'array', 'nix_name'}
    if missing := required - set(inv):
        raise ValueError(f'Unit inventory missing {sorted(missing)}')
    if inv.unit_instance_id.duplicated().any() or inv.duplicated(INVENTORY_KEYS).any():
        raise ValueError('Duplicate UUID / keys in unit inventory')
    params = yaml.safe_load(params_path.read_text())
    if not isinstance(params, dict) or not {'width_intervals', 'first_peak_height'} <= set(params):
        raise ValueError('Waveform parameters require width_intervals and first_peak_height')
    intervals = params['width_intervals']
    peak = params['first_peak_height']
    if len(intervals) != 3:
        raise ValueError('Expected three waveform-width intervals')

    computed = []
    for session in sorted(inv.session.unique()):
        path = root / 'sessions' / f'{session}.npz'
        with np.load(path) as z:
            ids = z['unit_instance_ids'].astype(str)
            raw = z['waveforms_raw']
        if len(ids) != len(raw) or len(set(ids)) != len(ids):
            raise ValueError(f'{session}: malformed/duplicate waveform IDs')
        for uid, wf in zip(ids, raw):
            entry = {'unit_instance_id': uid}
            try:
                entry.update(classify(wf, intervals, peak))
                entry['classify_error'] = ''
            except Exception as exc:
                entry.update(width_wf=np.nan, width_wf_class=np.nan,
                             wf_direction=np.nan, final_class=np.nan)
                entry['classify_error'] = str(exc)
            computed.append(entry)
    annotations = pd.DataFrame(computed)
    if annotations.unit_instance_id.duplicated().any():
        raise ValueError('Duplicate unit across session NPZ files')
    if set(annotations.unit_instance_id) != set(inv.unit_instance_id):
        raise ValueError('Session NPZ UUID set does not exactly match inventory')
    joined = inv.merge(annotations, on='unit_instance_id', validate='one_to_one')
    joined['label_provenance'] = np.where(joined.final_class.notna(),
                                          'computed_from_raw_waveform', 'unclassified')
    output = root / 'unit_instances_labeled.csv'
    _atomic_csv(joined, output)
    metadata = {'params_sha256': hashlib.sha256(params_path.read_bytes()).hexdigest(),
                'source_inventory_sha256': hashlib.sha256(inv_path.read_bytes()).hexdigest(),
                'n_units': len(joined),
                'n_classified': int(joined.final_class.notna().sum()),
                'n_classification_failures': int(joined.final_class.isna().sum()),
                'output': str(output)}
    (root / 'waveform_class_config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata

def summarize_tracks(root: Path) -> dict:
    """Post-selection class summary. A label mismatch NEVER changes inclusion."""
    root = Path(root)
    labels = pd.read_csv(root / 'unit_instances_labeled.csv', low_memory=False)
    selection = root / 'identity_diagnostics/joint/selection'
    membership = pd.read_csv(root / 'identity_diagnostics/joint/candidate_joint_unit_membership.csv')
    selected = pd.read_csv(selection / 'unit_to_neuron.csv')
    if membership.unit_instance_id.duplicated().any() or labels.unit_instance_id.duplicated().any():
        raise ValueError('Duplicate labels or track membership')
    if set(labels.unit_instance_id) != set(membership.unit_instance_id):
        raise ValueError('Labels and track membership cover different unit sets')
    grouped = membership.merge(labels[['unit_instance_id', 'final_class']],
                            on='unit_instance_id', validate='one_to_one')
    if selected.unit_instance_id.duplicated().any():
        raise ValueError('Duplicate selected unit membership')
    selected_ids = selected[['unit_instance_id', 'neuron_index', 'neuron_id']]
    grouped = grouped.merge(selected_ids, on='unit_instance_id', how='left', validate='one_to_one')
    if len(grouped) != len(labels):
        raise AssertionError('Summarizing labels unexpectedly changed population size')
    # Polarity and class disagreements are audit fields, never selection criteria.
    grouped['label_polarity'] = grouped.final_class.map(
        lambda label: label.split('_', 1)[0] if isinstance(label, str) else pd.NA)

    def unique_values(s):
        return '|'.join(sorted(set(s.dropna().astype(str))))

    summary = grouped.groupby(TRACK_ID, sort=True).agg(
        waveform_class_set=('final_class', unique_values),
        polarity_set=('label_polarity', unique_values),
        n_unit_instances=('unit_instance_id', 'size'),
        n_sessions=('session', 'nunique'),
        neuron_index=('neuron_index', 'first'),
        neuron_id=('neuron_id', 'first'),
    ).reset_index()
    summary['n_distinct_classes'] = summary.waveform_class_set.map(
        lambda s: len(s.split('|')) if s else 0)
    summary['n_distinct_polarities'] = summary.polarity_set.map(
        lambda s: len(s.split('|')) if s else 0)
    summary['mixed_waveform_classes'] = summary.n_distinct_classes.gt(1)
    summary['mixed_polarity'] = summary.n_distinct_polarities.gt(1)
    summary['selected'] = summary.neuron_id.notna()
    _atomic_csv(summary, selection / 'track_waveform_class_audit.csv')
    _atomic_csv(grouped.loc[grouped.neuron_id.notna()],
                selection / 'selected_unit_waveform_classes.csv')
    result = {'n_tracks': len(summary), 'n_selected_tracks': int(summary.selected.sum()),
              'n_selected_mixed_class': int((summary.selected & summary.mixed_waveform_classes).sum()),
              'n_selected_mixed_polarity': int((summary.selected & summary.mixed_polarity).sum())}
    print(json.dumps(result, indent=2), flush=True)
    return result



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('classify', help='Classify extracted unit waveforms')
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--params', type=Path, required=True,
                   help='YAML containing width_intervals and first_peak_height')
    p = sub.add_parser('summarize', help='Summarize track-level waveform class disagreements')
    p.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == 'classify':
        classify_inventory(args.root, args.params)
    else:
        summarize_tracks(args.root)


if __name__ == '__main__':
    main()
