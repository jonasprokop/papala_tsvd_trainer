"""Prediction-free reliability export for the prepared tracked-neuron TVSD data.

Reconstructs each track's test responses from the frozen prepared dataset (no
KS4 or raw-spike access, no sorting or matching redone):
  sample_index.csv      one row per presentation: animal, session, trial_index,
                        source_row, stimulus_id, partition
  sessions/<s>.npz      raw spike counts [trials, local tracks] + observed mask
  unit_to_neuron.csv    track -> KS4 unit instance per session (provenance)
  neuron_index.csv      selected tracks (merged into the per-neuron table)
Only the frozen `test` partition (100 images, original repeats) is used.

Neuron identity: column order and `neuron_id` are taken from neuron_index.csv
exactly as model readouts take them, and checked against unit_to_neuron.csv and
the column placement used by TVSDTable; any disagreement aborts. Consumers
join on `neuron_id`, never on column position.

Compute once per frozen dataset: outputs carry the dataset version, split hash
and frozen-input hashes, so every model consumes the same estimates.

Writes (never overwrites):
  reliability_per_neuron.csv   all reliability definitions + session diagnostics
  reliability_per_session.csv  per (track, session): trials, images, offset
  reliability_summary.json     medians by animal / waveform class / n_sessions,
                               threshold counts, test-image session balance
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sinz_model.tvsd_data import TVSDTable
from .evaluation_protocols import METRIC_LEVELS, score_protocols

RELIABILITY_KEYS = ('papale_reliab_balanced_adapted', 'papale_oracle_balanced_adapted',
                    'papale_oracle_30rep_model_estimate',
                    'single_trial_pairwise_reliability_estimate',
                    'split_half_sb_reliability', 'papale_oracle_adapted',
                    'cadena_explainable_fraction',
                    'pairwise_reliability_within_session',
                    'pairwise_reliability_between_sessions',
                    'median_repeats_per_session_cell',
                    'session_offset_sd_over_signal_sd')
COUNT_KEYS = ('n_sessions', 'n_repeated_images', 'n_observations',
              'papale_balanced_k', 'papale_balanced_n_images')
THRESHOLDS = (0.15, 0.3, 0.4)


def load_test(prepared_root):
    root = Path(prepared_root)
    info = json.loads((root / 'dataset_info.json').read_text())
    widths = {a: int(info['n_neurons'][a]) for a in ('F', 'N')}
    rows = pd.read_csv(root / 'sample_index.csv', low_memory=False)
    test = TVSDTable(rows.loc[rows.partition.eq('test')], root, widths)
    return test, widths, info


def column_neuron_ids(root, widths):
    """Column order exactly as model readouts build it (transformer_model.train.
    _neuron_order): neuron_index.csv, F then N, sorted by neuron_index. Fails
    unless every identity source agrees with the column the data was placed in."""
    root = Path(root)
    neurons = pd.read_csv(root / 'neuron_index.csv', low_memory=False)
    units = pd.read_csv(root / 'unit_to_neuron.csv', low_memory=False)
    ids = []
    for animal in ('F', 'N'):
        rows = neurons.loc[neurons.animal.eq(animal)].sort_values('neuron_index')
        if rows.neuron_index.astype(int).tolist() != list(range(widths[animal])):
            raise ValueError(f'neuron_index not contiguous 0..{widths[animal]-1} for {animal}')
        ids.extend(rows.neuron_id.astype(str).tolist())
    if len(set(ids)) != len(ids) or any(not v or v == 'nan' for v in ids):
        raise ValueError('neuron_id missing or not unique')
    # TVSDTable places session data at column = offset(animal) + neuron_index.
    by_column = [f'{a}/{i}' for a in ('F', 'N') for i in range(widths[a])]
    if ids != by_column:
        raise ValueError('neuron_id does not equal animal/neuron_index for every column')
    unit_ids = units.assign(expected=units.animal + '/' + units.neuron_index.astype(str))
    if not (unit_ids.neuron_id == unit_ids.expected).all():
        raise ValueError('unit_to_neuron.csv neuron_id disagrees with animal/neuron_index')
    if set(unit_ids.neuron_id) != set(ids):
        raise ValueError('unit_to_neuron.csv and neuron_index.csv cover different neurons')
    return ids


def per_session_table(test, column_meta):
    """Per (track, session) counts and response offset vs the track's image means."""
    y, m = test.responses.astype(np.float64), test.mask
    sess = test.rows.session.to_numpy()
    records = []
    for col in range(y.shape[1]):
        obs = m[:, col]
        if not obs.any():
            continue
        ids, vals, ses = test.image_ids[obs], y[obs, col], sess[obs]
        image_mean = pd.Series(vals).groupby(ids).transform('mean').to_numpy()
        frame = pd.DataFrame(dict(session=ses, image=ids, value=vals, resid=vals - image_mean))
        for s, g in frame.groupby('session', sort=True):
            per_image = g.groupby('image').size()
            records.append(dict(**column_meta[col], session=s,
                                n_trials=len(g), n_images=int(per_image.size),
                                n_images_ge2_in_session=int((per_image >= 2).sum()),
                                mean_count=float(g.value.mean()),
                                offset_vs_image_means=float(g.resid.mean())))
    return pd.DataFrame.from_records(records)


def track_attributes(root, unit_labels):
    """Track-level waveform class / area when the labelled unit table is supplied."""
    units = pd.read_csv(Path(root) / 'unit_to_neuron.csv', low_memory=False)
    attrs = pd.DataFrame({'neuron_id': units.neuron_id.unique()})
    if unit_labels:
        labels = pd.read_csv(unit_labels, low_memory=False)
        keep = [c for c in ('final_class', 'area') if c in labels.columns]
        merged = units.merge(labels[['unit_instance_id'] + keep], on='unit_instance_id', how='left')
        for c in keep:
            # A track keeps a class only if all its unit instances agree.
            agg = merged.groupby('neuron_id')[c].agg(
                lambda v: v.dropna().iloc[0] if v.dropna().nunique() == 1 else 'mixed')
            attrs = attrs.merge(agg.rename(c).reset_index(), on='neuron_id', how='left')
    return attrs


def summarize(table, per_session):
    def med(frame):
        out = {k: (float(frame[k].median()) if frame[k].notna().any() else None)
               for k in RELIABILITY_KEYS}
        rel1 = frame['papale_reliab_balanced_adapted']   # Papale's formula, per neuron
        out['n'] = int(len(frame))
        out['n_scored'] = int(rel1.notna().sum())
        out['n_above_papale_reliab'] = {str(t): int((rel1 > t).sum()) for t in THRESHOLDS}
        out['median_balanced_k'] = (float(frame['papale_balanced_k'].median())
                                    if len(frame) else None)
        out['frac_between_sessions_undefined'] = float(
            frame['pairwise_reliability_between_sessions'].isna().mean())
        return out
    summary = dict(all=med(table),
                   by_animal={a: med(g) for a, g in table.groupby('animal')},
                   by_n_sessions={str(k): med(g) for k, g in table.groupby('n_sessions')})
    for c in ('final_class', 'area'):
        if c in table.columns:
            summary[f'by_{c}'] = {str(k): med(g) for k, g in table.groupby(c, dropna=False)}
    # Test-image balance: in how many of a track's sessions does each image appear?
    if len(per_session):
        summary['per_session'] = dict(
            median_trials_per_session=float(per_session.n_trials.median()),
            median_images_per_session=float(per_session.n_images.median()),
            frac_sessions_with_all_100_images=float((per_session.n_images == 100).mean()),
            median_images_ge2_in_session=float(per_session.n_images_ge2_in_session.median()))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prepared-root', required=True, type=Path)
    ap.add_argument('--output-dir', required=True, type=Path, help='new directory; never overwritten')
    ap.add_argument('--unit-labels', type=Path,
                    help='optional unit_instances_labeled.csv (final_class, area per unit instance)')
    ap.add_argument('--min-repeated-images', type=int, default=20)
    ap.add_argument('--splits', type=int, default=50)
    args = ap.parse_args()
    if args.output_dir.exists():
        ap.error('Output directory exists; use a distinct versioned name')
    test, widths, info = load_test(args.prepared_root)
    ids = column_neuron_ids(args.prepared_root, widths)
    animals = np.concatenate([[a] * widths[a] for a in ('F', 'N')])
    local = np.concatenate([np.arange(widths[a]) for a in ('F', 'N')])
    meta = [dict(column=int(c), animal=str(animals[c]), neuron_index=int(local[c]),
                 neuron_id=ids[c]) for c in range(len(animals))]
    scores = score_protocols(None, test.responses, test.mask, test.image_ids,
                             splits=args.splits, min_repeated_images=args.min_repeated_images,
                             neuron_ids=np.asarray(ids),
                             sessions=test.rows.session.to_numpy())
    table = pd.DataFrame(meta)
    for k in RELIABILITY_KEYS + COUNT_KEYS:
        table[k] = scores[k]
    neurons = pd.read_csv(args.prepared_root / 'neuron_index.csv', low_memory=False)
    table = table.merge(neurons.drop(columns=['animal', 'neuron_index'], errors='ignore'),
                        on='neuron_id', how='left')
    table = table.merge(track_attributes(args.prepared_root, args.unit_labels),
                        on='neuron_id', how='left')
    per_session = per_session_table(test, meta)
    summary = summarize(table, per_session)
    summary['metric_levels'] = {k: METRIC_LEVELS.get(k, 'count') for k in RELIABILITY_KEYS + COUNT_KEYS}
    summary['source'] = dict(prepared_root=str(args.prepared_root),
                             dataset_version=info.get('version'),
                             split_sha256=info.get('split_sha256'),
                             frozen_input_sha256=info.get('frozen_input_sha256'),
                             neuron_ids_in_column_order=ids,
                             response_window_ms=info.get('response_window_ms'),
                             splits=args.splits, min_repeated_images=args.min_repeated_images)
    args.output_dir.mkdir(parents=True)
    table.to_csv(args.output_dir / 'reliability_per_neuron.csv', index=False)
    per_session.to_csv(args.output_dir / 'reliability_per_session.csv', index=False)
    (args.output_dir / 'reliability_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['all'], indent=2))


if __name__ == '__main__':
    main()
