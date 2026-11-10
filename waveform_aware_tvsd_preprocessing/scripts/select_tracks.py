#!/usr/bin/env python3
"""Model-agnostic, auditable selection of *provisional* tracked KS4 neurons.

Default thresholds mirror the previous conservative V1 working cohort where
possible, without classification-dependent vetoes or trimming.
Outputs neuron_index.csv and unit_to_neuron.csv, never Gaussian readout indices.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from track_units import parse_session

ROOT = Path('/project/data/processed/tvsd_waveform_aware_v1')
TRACK = 'candidate_joint_track_id'


def _need(df, names, name):
    missing = set(names) - set(df)
    if missing:
        raise ValueError(f'{name}: missing columns {sorted(missing)}')


def select_tracks(*, root: Path = ROOT, output: Path | None = None,
                  min_recordings: int = 3, min_train: int = 2000,
                  min_pairwise_r: float = 0.93, max_rate_ratio: float = 3.0,
                  max_distinct_r: float = 0.98,
                  reappearance: str = 'merge', keep_all_tracks: bool = False,
                  site_audit: Path | None = None) -> dict:
    """Select longitudinal tracks (not readout weights); preserve full audit.

    reappearance='merge': one column spanning all selected recordings.
    reappearance='first-only': one column, only earliest recording retained.
    reappearance='exclude-repeated': exclude all tracks found in >1 session.
    """
    root = Path(root)
    output = Path(output) if output else root / 'identity_diagnostics/joint/selection'
    if (min_recordings < 1 or min_train < 0 or not -1 <= min_pairwise_r <= 1 or
            (max_rate_ratio != 0 and max_rate_ratio < 1) or
            not -1 <= max_distinct_r <= 1):
        raise ValueError('Invalid selection thresholds')
    if reappearance not in ('merge', 'first-only', 'exclude-repeated'):
        raise ValueError('Invalid reappearance policy')
    if reappearance == 'exclude-repeated' and min_recordings > 1:
        raise ValueError(
            "--reappearance exclude-repeated requires --min-recordings 1: "
            "repeated tracks are excluded, so no track can have multiple recordings"
        )
    membership_path = root / 'identity_diagnostics/joint/candidate_joint_unit_membership.csv'
    quality_path = root / 'identity_diagnostics/joint/candidate_joint_track_quality.csv'
    membership = pd.read_csv(membership_path)
    quality = pd.read_csv(quality_path)
    inventory = pd.read_csv(root / 'unit_instances.csv')
    presentations = pd.read_csv(root / 'images/presentation_to_stimulus.csv')
    _need(membership, ['unit_instance_id', TRACK], 'identity membership')
    _need(quality, [TRACK, 'min_pairwise_waveform_r'], 'track quality')
    _need(inventory, ['unit_instance_id', 'animal', 'session', 'array',
                       'electrode_id', 'firing_rate_hz'], 'KS4 inventory')
    _need(presentations, ['session', 'split'], 'presentations')
    if (not membership.unit_instance_id.is_unique or not inventory.unit_instance_id.is_unique
            or not quality[TRACK].is_unique or set(membership.unit_instance_id) != set(inventory.unit_instance_id)):
        raise ValueError('Identity membership must cover each original KS4 unit exactly once')
    units = inventory.merge(membership[['unit_instance_id', TRACK]],
                            on='unit_instance_id', validate='one_to_one')
    if units.groupby([TRACK, 'session']).size().max() > 1:
        raise ValueError('A proposed neuron contains two simultaneously recorded units')
    if units.groupby(TRACK).animal.nunique().max() > 1:
        raise ValueError('A proposed neuron crosses monkeys')
    if units.groupby(TRACK)[['array', 'electrode_id']].nunique().max().max() > 1:
        raise ValueError('A proposed neuron crosses physical electrodes')
    p = presentations.groupby(['session', 'split']).size().unstack(fill_value=0)
    for name in ('train', 'test'):
        if name not in p:
            p[name] = 0
    if not set(units.session).issubset(p.index):
        raise ValueError('Presentation counts missing for some KS4 sessions')
    units['train_presentations'] = units.session.map(p['train']).astype(int)
    units['test_presentations'] = units.session.map(p['test']).astype(int)
    # Determine the observations that would actually be retained BEFORE applying
    # the minimum-data gate. In first-only mode, a track's aggregate count can
    # otherwise exceed min_train even when its kept session falls below it.
    kept_units = units
    if reappearance == 'first-only':
        kept_units = (units.assign(_session_key=units.session.map(parse_session))
                           .sort_values(['_session_key', 'unit_instance_id'], kind='stable')
                           .drop_duplicates(TRACK, keep='first')
                           .drop(columns='_session_key'))
    kept_counts = kept_units.groupby(TRACK, sort=True).agg(
        n_kept_recordings=('session', 'nunique'),
        n_kept_train_presentations=('train_presentations', 'sum'),
        n_kept_test_presentations=('test_presentations', 'sum'),
    ).reset_index()
    table = units.groupby(TRACK, sort=True).agg(
        animal=('animal', 'first'), array=('array','first'), electrode_id=('electrode_id','first'),
        n_recordings=('session','nunique'), n_unit_instances=('unit_instance_id','size'),
        n_train_presentations=('train_presentations','sum'),
        n_test_presentations=('test_presentations','sum'),
        min_firing_rate=('firing_rate_hz','min'), max_firing_rate=('firing_rate_hz','max')
    ).reset_index()
    table = table.merge(kept_counts, on=TRACK, validate='one_to_one')
    with np.errstate(divide='ignore', invalid='ignore'):
        table['firing_rate_ratio'] = table.max_firing_rate / table.min_firing_rate
    invalid_rate = (table.min_firing_rate <= 0) | (~np.isfinite(table.firing_rate_ratio))
    table = table.merge(quality[[TRACK, 'min_pairwise_waveform_r']], on=TRACK,
                        how='left', validate='one_to_one')
    # Optional old electrode-site ambiguity veto. The simple unlabelled extractor
    # does not produce electrode_sites.csv, so report that gate as unavailable.
    if site_audit is None:
        candidate = root / 'identity_diagnostics/electrode_sites.csv'
        site_audit = candidate if candidate.is_file() else None
    has_site_audit = site_audit is not None
    table['ambiguous_multiunit_site'] = False
    if has_site_audit:
        sites = pd.read_csv(site_audit)
        keys = ['animal', 'array', 'electrode_id']
        _need(sites, keys + ['max_simultaneous_units','max_correlated_distinct_units'], 'site audit')
        if sites.duplicated(keys).any():
            raise ValueError('Duplicate electrode site in site audit')
        table = table.merge(sites[keys + ['max_simultaneous_units', 'max_correlated_distinct_units']],
                            on=keys, how='left', validate='many_to_one')
        if table.max_simultaneous_units.isna().any():
            raise ValueError('Site audit omits electrode sites')
        table['ambiguous_multiunit_site'] = ((table.max_simultaneous_units >= 2) &
                                               (table.max_correlated_distinct_units >= max_distinct_r))
    checks = {
        'insufficient_recordings': table.n_recordings < min_recordings,
        'insufficient_training_images': table.n_kept_train_presentations < min_train,
        # A single recording has no pairwise r: undefined is NOT inconsistent.
        'unstable_waveforms': (table.n_recordings > 1) & ~(table.min_pairwise_waveform_r >= min_pairwise_r),
        # --max-rate-ratio 0 opts out of this *separate selection* filter.
        'unstable_firing_rate': ((invalid_rate | (table.firing_rate_ratio > max_rate_ratio))
                                 if max_rate_ratio != 0 else pd.Series(False, index=table.index)),
        'ambiguous_multiunit_site': table.ambiguous_multiunit_site,
        'reappearing_neuron_excluded': ((table.n_recordings > 1) & (reappearance == 'exclude-repeated')),
    }
    table['exclusion_reasons'] = [
        ';'.join(name for name, bad in checks.items() if bool(bad.iloc[i]))
        for i in range(len(table))]
    table['include'] = (True if keep_all_tracks else table.exclusion_reasons.eq(''))
    # An explicit exclusion policy is always respected, including when --keep-all-tracks.
    if reappearance == 'exclude-repeated':
        table.loc[table.n_recordings > 1, 'include'] = False
    table['selection'] = np.where(table.include, 'included', 'excluded')
    included = table.loc[table.include].copy().sort_values(['animal', 'array', 'electrode_id', TRACK])
    if included.empty:
        raise ValueError('No tracks remain; adjust thresholds or reappearance policy')
    included['neuron_index'] = included.groupby('animal').cumcount().astype(np.int32)
    included['neuron_id'] = included.animal + '/' + included.neuron_index.astype(str)
    table = table.merge(included[[TRACK,'neuron_index','neuron_id']],
                        on=TRACK, how='left', validate='one_to_one')
    chosen = kept_units.merge(included[[TRACK,'neuron_index','neuron_id']], on=TRACK,
                         how='inner', validate='many_to_one')
    chosen = chosen.sort_values(['animal', 'neuron_index', 'session', 'unit_instance_id'],
                                kind='stable')
    if chosen.duplicated(['session','animal','neuron_index']).any():
        raise ValueError('Two simultaneous units mapped to the same neuron')
    if chosen.unit_instance_id.duplicated().any():
        raise ValueError('Unit appears in multiple neurons')
    outdir = output
    outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / 'track_selection_audit.csv', index=False)
    included.to_csv(outdir / 'neuron_index.csv', index=False)
    chosen[['unit_instance_id', 'session', 'animal', TRACK, 'neuron_index', 'neuron_id']].to_csv(
        outdir / 'unit_to_neuron.csv', index=False)
    settings = dict(min_recordings=min_recordings,min_train=min_train,
                    min_pairwise_r=min_pairwise_r,max_rate_ratio=max_rate_ratio,
                    max_distinct_r=max_distinct_r,reappearance=reappearance,
                    keep_all_tracks=keep_all_tracks,site_audit_applied=has_site_audit,
                    source_membership_sha256=hashlib.sha256(membership_path.read_bytes()).hexdigest(),
                    source_quality_sha256=hashlib.sha256(quality_path.read_bytes()).hexdigest())
    matching_file = root / 'identity_diagnostics/joint/matching_config.json'
    if matching_file.is_file():
        settings['matching_config_sha256'] = hashlib.sha256(matching_file.read_bytes()).hexdigest()
    (outdir / 'selection_config.json').write_text(json.dumps(settings, indent=2) + '\n')
    summary = dict(n_original_units=len(inventory), n_proposed_tracks=len(table),
                   n_selected_neurons=len(included), n_selected_unit_instances=len(chosen),
                   n_unselected_unit_instances=len(inventory)-len(chosen), **settings)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path)
    p.add_argument('--min-recordings',type=int,default=3)
    p.add_argument('--min-train',type=int,default=2000)
    p.add_argument('--min-pairwise-r',type=float,default=0.93)
    p.add_argument('--max-rate-ratio',type=float,default=3.0,
                   help='Maximum within-track firing-rate ratio (default: 3; 0 disables the rate filter)')
    p.add_argument('--max-distinct-r',type=float,default=0.98)
    p.add_argument('--site-audit',type=Path,help='Optional electrode_sites.csv for site-ambiguity gate')
    p.add_argument('--reappearance',choices=['merge','first-only','exclude-repeated'],
                   default='merge',help='Merge repeat recordings, retain first only, or drop all repeated tracks')
    p.add_argument('--keep-all-tracks',action='store_true',
                   help='Bypass quality thresholds (but still obey reappearance policy)')
    a=p.parse_args()
    select_tracks(root=a.root,output=a.output,min_recordings=a.min_recordings,
                  min_train=a.min_train,min_pairwise_r=a.min_pairwise_r,
                  max_rate_ratio=a.max_rate_ratio,max_distinct_r=a.max_distinct_r,
                  reappearance=a.reappearance,keep_all_tracks=a.keep_all_tracks,
                  site_audit=a.site_audit)

if __name__ == '__main__': main()
