"""Read-only adapter between frozen TVSD dataset v1 and the Cadena trainer.

Architecture/search/evaluation remain in the original sinz_model package.
The two animal readouts occupy non-overlapping F|N columns in masks and metrics.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset

from .data import SplitArtifact, _payload, _hash_payload, dataset_manifest_fingerprint


def _module_from_path(filename):
    filename = Path(filename)
    if not filename.is_file():
        raise FileNotFoundError('TVSD dataset source not found: %s' % filename)
    spec = importlib.util.spec_from_file_location('tvsd_dataset_frozen_v1', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TVSDTable:
    def __init__(self, rows, prepared_root, widths):
        self.rows = rows.reset_index(drop=True)
        self.image_ids = self.rows.stimulus_id.str.split(':').str[-1].astype('int64').to_numpy()
        self.repeat_index = self.rows.groupby(['animal','stimulus_id']).cumcount().to_numpy(dtype='int64')
        self.session_info = sorted(self.rows.session.unique())
        full_width = widths['F'] + widths['N']
        n = len(self.rows)
        self.responses = np.zeros((n, full_width), dtype=np.float32)
        self.mask = np.zeros((n, full_width), dtype=bool)
        if not n:
            return
        for session, row_ids in self.rows.groupby('session', sort=False).indices.items():
            with np.load(Path(prepared_root) / 'sessions' / (session+'.npz'), allow_pickle=False) as data:
                counts = data['counts']
                observed = data['observed']
                indices = data['readout_indices'].astype('int64')
                source_rows = data['source_rows'].astype('int64')
            animal = str(self.rows.iloc[int(row_ids[0])].animal)
            if not ((self.rows.iloc[row_ids].animal == animal).all()):
                raise ValueError('Mixed animals in session %s' % session)
            offset = 0 if animal == 'F' else widths['F']
            global_idx = indices + offset
            if len(set(indices)) != len(indices) or (indices >= widths[animal]).any() or (indices < 0).any():
                raise ValueError('Invalid readout indices in %s' % session)
            trial_indices = self.rows.iloc[row_ids].trial_index.to_numpy(dtype='int64')
            if np.any(trial_indices < 0) or np.any(trial_indices >= len(counts)):
                raise ValueError('Invalid trial indices in %s' % session)
            if not np.array_equal(source_rows[trial_indices], self.rows.iloc[row_ids].source_row.to_numpy(dtype='int64')):
                raise ValueError('Presentation source-row mismatch for %s' % session)
            self.responses[np.asarray(row_ids)[:,None], global_idx[None,:]] = counts[trial_indices]
            self.mask[np.asarray(row_ids)[:,None], global_idx[None,:]] = observed[trial_indices]
        if np.any(self.responses[~self.mask] != 0):
            raise AssertionError('Unobserved responses must remain masked')

    def subset_by_image_ids(self, ids):
        take = self.rows.iloc[np.flatnonzero(np.isin(self.image_ids, ids))]
        # Return without re-reading session NPZ files.
        result = object.__new__(TVSDTable)
        selected = np.isin(self.image_ids, ids)
        result.rows = self.rows.loc[selected].reset_index(drop=True)
        result.image_ids = self.image_ids[selected]
        result.repeat_index = self.repeat_index[selected]
        result.session_info = self.session_info
        result.responses = self.responses[selected]
        result.mask = self.mask[selected]
        return result


class TVSDOriginalInterface(Dataset):
    """TVSDDataset -> Cadena batch keys; padded per-animal neuron masks."""
    def __init__(self, tvsd_ds, animal, widths, mean, std):
        self.inner = tvsd_ds
        self.animal = animal
        self.widths = widths
        self.mean = float(mean)
        self.std = float(std)
        self.n_readouts = widths['F'] + widths['N']
        self.repeat_index = tvsd_ds.samples.groupby('stimulus_id').cumcount().to_numpy(dtype='int64')
        if not (0 < self.std < 1.0):
            raise ValueError('Invalid grayscale normalization std')

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        sample = self.inner[i]
        rgb = sample['image']
        if tuple(rgb.shape[:1]) != (3,):
            raise ValueError('RGB channel count mismatch')
        gray = 0.2989*rgb[0] + 0.5870*rgb[1] + 0.1140*rgb[2]
        image = ((gray - self.mean) / self.std).unsqueeze(0).repeat(3,1,1)
        target = torch.zeros(self.n_readouts, dtype=torch.float32)
        mask = torch.zeros(self.n_readouts, dtype=torch.bool)
        offset = 0 if self.animal == 'F' else self.widths['F']
        end = offset + self.widths[self.animal]
        target[offset:end] = sample['responses']
        mask[offset:end] = sample['response_mask']
        if not torch.isfinite(image).all() or not torch.isfinite(target).all():
            raise ValueError('Nonfinite training batch')
        return dict(image=image,response=target,mask=mask,
                    image_id=torch.tensor(int(sample['stimulus_id'].split(':')[1]), dtype=torch.long),
                    repeat_index=torch.tensor(int(self.repeat_index[i]), dtype=torch.long))


class TVSDTrainingBundle:
    def __init__(self, cfg):
        dc = cfg['data']
        self.root = Path(dc['root'])
        self.info = json.loads((self.root/'dataset_info.json').read_text())
        self.widths = {a:int(self.info['n_readouts'][a]) for a in ('F','N')}
        if self.widths != {a:int(cfg['model']['readout']['outdims_by_animal'][a]) for a in ('F','N')}:
            raise ValueError('Frozen dataset and model readout counts disagree')
        self.df = pd.read_csv(self.root/'sample_index.csv', low_memory=False)
        for key in ('animal','partition','session','trial_index','source_row','stimulus_id','image_relative_path'):
            if key not in self.df.columns:
                raise ValueError('Missing TVSD presentation column: %s' % key)
        if not self.df.animal.isin(['F','N']).all() or not self.df.partition.isin(['train','val','test']).all():
            raise ValueError('Unexpected animal or image partition')
        if self.df.duplicated(['session','trial_index']).any():
            raise ValueError('Duplicate session/trial presentation')
        if self.df.groupby('stimulus_id').partition.nunique().max() > 1:
            raise ValueError('Stimulus leaks between partitions')
        if self.df.loc[self.df.partition.eq('test'),'stimulus_id'].nunique() != int(self.info['partition_images']['test']):
            raise ValueError('Frozen test identity mismatch')
        if int(self.info['n_selected_unit_instances']) != len(pd.read_csv(self.root/'unit_to_readout.csv')):
            raise ValueError('Unit mapping differs from frozen dataset')
        norm_path = Path(dc['preprocessing']['normalization_path'])
        self.normalization_stats = json.loads(norm_path.read_text())
        if self.normalization_stats.get('input_hw') != dc['preprocessing']['expected_output_hw']:
            raise ValueError('Normalization was measured for another image resolution')
        if self.normalization_stats.get('preprocessing') != 'rgb_resize_bicubic_then_luma':
            raise ValueError('Wrong image normalization recipe')
        self.mean = float(self.normalization_stats['mean'])
        self.std = float(self.normalization_stats['std'])
        if not (0 <= self.mean <= 1 and 0 < self.std < 1):
            raise ValueError('Invalid normalization statistics')
        if int(self.info['split_seed']) != int(dc['split']['seed']):
            raise ValueError('TVSD split seed changed')
        # Use prepared train/validation partition, never redraw it.
        train_ids = sorted(int(x.split(':')[1]) for x in self.df.loc[self.df.partition.eq('train'),'stimulus_id'].unique())
        val_ids = sorted(int(x.split(':')[1]) for x in self.df.loc[self.df.partition.eq('val'),'stimulus_id'].unique())
        if set(train_ids) & set(val_ids):
            raise ValueError('Train/validation image overlap')
        if len(train_ids) != int(self.info['partition_images']['train']) or len(val_ids) != int(self.info['partition_images']['val']):
            raise ValueError('Train/validation image counts differ from frozen manifest')
        fraction = len(train_ids)/(len(train_ids)+len(val_ids))
        if abs(fraction-float(dc['split']['train_fraction'])) > 1e-3:
            raise ValueError('Configured fraction disagrees with frozen split')
        algorithm='frozen_tvsd_dataset_v1'
        payload = _payload(train_ids,val_ids,int(dc['split']['seed']),fraction,algorithm)
        self.split = SplitArtifact(np.asarray(train_ids,dtype='int64'),np.asarray(val_ids,dtype='int64'),
                                   int(dc['split']['seed']),fraction,algorithm,_hash_payload(payload))
        self.full = TVSDTable(self.df.loc[self.df.partition.isin(['train','val'])],self.root,self.widths)
        self.test = TVSDTable(self.df.loc[self.df.partition.eq('test')],self.root,self.widths)
        self.mapping = [{'session_id':s,'animal':s[7]} for s in sorted(self.df.session.unique())]
        self.fingerprint = hashlib.sha256((dataset_manifest_fingerprint(str(self.root),False)+
            hashlib.sha256(norm_path.read_bytes()).hexdigest()).encode('ascii')).hexdigest()
        self.source = _module_from_path(dc['dataset_module_path'])
        self.cfg = cfg

    def dataset(self, partition):
        cfg = self.cfg['data']
        hw = cfg['preprocessing']['expected_output_hw']
        if hw[0] != hw[1]:
            raise ValueError('TVSDDataset currently expects square images')
        datasets=[]
        for animal in ('F','N'):
            inner=self.source.TVSDDataset(animal=animal,partition=partition,
                    prepared_root=self.root,image_root=Path(cfg['image_root']),image_size=int(hw[0]))
            datasets.append(TVSDOriginalInterface(inner,animal,self.widths,self.mean,self.std))
        return ConcatDataset(datasets)
