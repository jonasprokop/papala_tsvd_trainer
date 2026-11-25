"""Bind saved tracked-unit reliability to model columns, with provenance checks.

This is an *adapted reconstructed* reliability, never Papale's measured MUA
``reliab`` coordinate. It is used only in reporting, not model selection.
"""

import csv
import json
from pathlib import Path

import numpy as np


def load_tracked_reliability(export_dir, neuron_ids, prepared_root):
    export_dir = Path(export_dir).expanduser()
    prepared_root = Path(prepared_root).expanduser()
    source = json.loads((export_dir / 'reliability_summary.json').read_text())['source']
    dataset = json.loads((prepared_root / 'dataset_info.json').read_text())
    ids = [str(v) for v in neuron_ids]
    if source.get('neuron_ids_in_column_order') != ids:
        raise ValueError('Reliability export neuron order/identities differ from model readout')
    for key in ('split_sha256', 'frozen_input_sha256', 'response_window_ms'):
        # A key missing on both sides would compare None == None and pass silently.
        if source.get(key) in (None, '', {}, []) or dataset.get(key) in (None, '', {}, []):
            raise ValueError(f'Provenance key missing, cannot verify binding: {key}')
        if source.get(key) != dataset.get(key):
            raise ValueError(f'Reliability export differs from training dataset: {key}')

    with (export_dir / 'reliability_per_neuron.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    mapping = {row['neuron_id']: row for row in rows}
    if len(mapping) != len(rows) or set(mapping) != set(ids):
        raise ValueError('Reliability CSV missing, duplicated or containing unexpected neuron IDs')
    if rows and 'papale_reliab_balanced_adapted' not in rows[0]:
        raise ValueError('Reliability CSV lacks papale_reliab_balanced_adapted')
    # pandas writes NaN as an empty field; float('') would raise.
    return np.asarray([float(mapping[i]['papale_reliab_balanced_adapted'] or 'nan')
                       for i in ids], dtype=np.float64)
