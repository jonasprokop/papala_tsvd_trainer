#!/usr/bin/env python3
"""Attach rebuilt Katka waveform classes to the existing KS4 V1 inventory.

Strict exact (session, array, UUID) join. Never reprocesses spikes or changes
existing inventory/identity files; creates unit_instances_labeled.csv beside them.
Existing labels from the previous Katka exports must agree with rebuilt labels.

Example inside Docker:
    python scripts/attach_katka_rebuilt.py \
        --rebuilt /project/data/reports/katka_labels_rebuilt/all_units.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INVENTORY = Path('/project/data/processed/ks4_v1/unit_instances.csv')
DEFAULT_REBUILT = Path('/project/data/reports/katka_labels_rebuilt/all_units.csv')
DEFAULT_OUTPUT = Path('/project/data/processed/ks4_v1/unit_instances_labeled.csv')
KEY_LEFT = ['session', 'array', 'nix_name']
KEY_RIGHT = ['session', 'array', 'cell_name']
CLASSES = {'UP', 'DOWN_wide', 'DOWN_medium_shallow', 'DOWN_medium_sharp',
           'DOWN_narrow_shallow', 'DOWN_narrow_sharp'}


def check_numeric(left: pd.DataFrame, lc: str, rc: str, *, rtol: float=1e-5,
                  atol: float=1e-5) -> None:
    if lc not in left.columns or rc not in left.columns:
        return
    a = pd.to_numeric(left[lc], errors='coerce').to_numpy(float)
    b = pd.to_numeric(left[rc], errors='coerce').to_numpy(float)
    valid = np.isfinite(a) & np.isfinite(b)
    if not valid.all():
        raise ValueError(f'Nonfinite values in {lc}/{rc}: {(~valid).sum()} rows')
    different = ~np.isclose(a, b, rtol=rtol, atol=atol)
    if different.any():
        examples = left.loc[different, ['unit_instance_id', lc, rc]].head(8)
        raise ValueError(f'{lc}/{rc} differ for {different.sum():,} units:\n{examples.to_string(index=False)}')
    print(f'  PASS: {lc} matches rebuilt {rc} for {len(left):,} units')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inventory', type=Path, default=DEFAULT_INVENTORY)
    ap.add_argument('--rebuilt', type=Path, default=DEFAULT_REBUILT)
    ap.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    if args.output.resolve() == args.inventory.resolve():
        ap.error('Output must differ from original inventory (never overwrite it)')

    original = pd.read_csv(args.inventory, low_memory=False)
    rebuilt = pd.read_csv(args.rebuilt, low_memory=False)
    required_old = {'unit_instance_id', 'animal', 'session', 'array', 'nix_name', 'final_class'}
    required_new = {'session', 'array', 'area', 'cell_name', 'final_class',
                    'katka_kept', 'in_v1_selection'}
    if missing := required_old - set(original.columns):
        raise ValueError(f'Inventory missing columns: {sorted(missing)}')
    if missing := required_new - set(rebuilt.columns):
        raise ValueError(f'Rebuilt export missing columns: {sorted(missing)}')
    if original['unit_instance_id'].duplicated().any():
        raise ValueError('Duplicate inventory unit_instance_id')
    if original.duplicated(KEY_LEFT).any():
        raise ValueError('Duplicate keys in inventory')

    rebuilt = rebuilt.loc[rebuilt.area.eq('V1')].copy()
    if rebuilt.empty or rebuilt.duplicated(KEY_RIGHT).any():
        raise ValueError('Empty rebuilt V1 population or duplicate keys')
    if not rebuilt['katka_kept'].eq(True).all():
        raise ValueError('Some rebuilt V1 units have katka_kept=False')
    if not rebuilt['in_v1_selection'].eq(True).all():
        raise ValueError('Some rebuilt V1 units are excluded by in_v1_selection')
    if rebuilt['final_class'].isna().any() or not rebuilt['final_class'].isin(CLASSES).all():
        raise ValueError('Missing or unexpected rebuilt V1 waveform class')
    if 'classify_error' in rebuilt and rebuilt['classify_error'].notna().any():
        raise ValueError('Some rebuilt V1 classifications report errors')

    cols = KEY_RIGHT + ['final_class', 'katka_kept', 'in_v1_selection']
    extras = ['animal', 'Electrode_ID', 'train_order_original', 'n_spikes', 'FR',
              'width_wf', 'width_wf_class', 'wf_direction', 'source_file']
    cols += [col for col in extras if col in rebuilt]
    right = rebuilt[cols].rename(columns={
        'cell_name': 'nix_name', 'final_class': 'rebuilt_final_class',
        'animal': 'rebuilt_animal', 'Electrode_ID': 'rebuilt_electrode_id',
        'train_order_original': 'rebuilt_train_order_original',
        'n_spikes': 'rebuilt_n_spikes', 'FR': 'rebuilt_FR',
        'source_file': 'rebuilt_source_file',
    })
    out = original.merge(right, on=KEY_LEFT, how='outer',
                         indicator=True, validate='one_to_one')
    if not out['_merge'].eq('both').all():
        unmatched_old = out.loc[out['_merge'].eq('left_only'), KEY_LEFT].head(8)
        unmatched_new = out.loc[out['_merge'].eq('right_only'), KEY_LEFT].head(8)
        raise ValueError(
            f'Inventory/rebuilt V1 UUID sets differ: '
            f'{out._merge.eq("left_only").sum()} inventory-only, '
            f'{out._merge.eq("right_only").sum()} rebuilt-only.\n'
            f'Inventory-only:\n{unmatched_old.to_string(index=False)}\n'
            f'Rebuilt-only:\n{unmatched_new.to_string(index=False)}')
    out = out.drop(columns=['_merge'])
    out = out.set_index('unit_instance_id').loc[original.unit_instance_id].reset_index()

    # Confirm underlying unit identity against independently read KS4 metadata.
    expected_id = (out.session.astype(str) + '/A' + out['array'].astype(str)
                   + '/' + out.nix_name.astype(str))
    if not expected_id.eq(out.unit_instance_id).all():
        raise ValueError('Inventory unit_instance_id strings disagree with exact join key')
    if 'rebuilt_animal' in out:
        expected_animal = 'monkey' + out.animal.astype(str)
        if not out.rebuilt_animal.eq(expected_animal).all():
            raise ValueError('Animal mismatch between rebuilt records and inventory')
    print('Exact (session, array, UUID) matches:', f'{len(out):,}/{len(original):,}')
    check_numeric(out, 'electrode_id', 'rebuilt_electrode_id')
    check_numeric(out, 'train_order_original', 'rebuilt_train_order_original', rtol=0, atol=0)
    check_numeric(out, 'n_spikes_total', 'rebuilt_n_spikes', rtol=0, atol=0)
    check_numeric(out, 'firing_rate_hz', 'rebuilt_FR', rtol=1e-4, atol=1e-4)

    original_class = out.final_class.copy()
    previously_labeled = original_class.notna()
    conflict = previously_labeled & original_class.ne(out.rebuilt_final_class)
    if conflict.any():
        examples = out.loc[conflict, ['unit_instance_id','final_class','rebuilt_final_class']].head(15)
        raise ValueError(f'{conflict.sum()} prior Katka label conflicts. No output written:\n'
                         + examples.to_string(index=False))

    out['has_original_katka_label'] = previously_labeled
    out['original_final_class'] = original_class
    out['final_class'] = original_class.fillna(out.rebuilt_final_class)
    out['has_katka_label'] = out.final_class.notna()
    out['label_provenance'] = np.where(previously_labeled,
                                       'original_katka_export', 'rebuilt_filtered_ks4')
    # Keep original katka_source_csv as-is and preserve previous reference units CSV.
    if not out['has_katka_label'].all():
        raise ValueError('Some V1 units remain unlabeled after exact rebuild')
    if out['longitudinal_neuron_id'].notna().any() if 'longitudinal_neuron_id' in out else False:
        print('NOTE: existing longitudinal IDs are preserved unchanged.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + '.tmp')
    try:
        out.to_csv(temp, index=False)
        temp.replace(args.output)
    finally:
        temp.unlink(missing_ok=True)
    print(f'Previously labeled, verified agreeing: {previously_labeled.sum():,}')
    print(f'Newly labeled from rebuilt KS4: {(~previously_labeled).sum():,}')
    print(f'Total labeled V1 units: {len(out):,}')
    print('By animal:', out.groupby('animal').size().to_dict())
    print('Class counts:', out.final_class.value_counts().to_dict())
    print('Saved:', args.output)
    print('Unchanged: original unit_instances.csv, Katka reference, session NPZs, track proposals')


if __name__ == '__main__':
    main()
