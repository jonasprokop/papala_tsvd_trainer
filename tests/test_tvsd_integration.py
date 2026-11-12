import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
import yaml

from sinz_model.tvsd_data import TVSDTrainingBundle, _module_from_path
from sinz_model.model import TVSDV1Model
from sinz_model.analysis import training_loss
from sinz_model.trainer import _initialize_readout_bias_from_targets


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / 'waveform_aware_tvsd_preprocessing' / 'scripts' / 'dataset.py'


class FakeCore(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 80, 3, 1, 1)
        self.output_shape = (80, 4, 4)

    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(self.conv(x), (4, 4))

    def set_trainable(self, trainable):
        for p in self.parameters():
            p.requires_grad = bool(trainable)

    def regularizer(self):
        return next(self.parameters()).new_zeros(())


def make_frozen_fixture(base):
    """Three miniature sessions with v3 NPZ keys, split SHA and train-only norm."""
    source = _module_from_path(SOURCE_PATH)
    from image_transform import image_hwc

    prepared = base / 'prepared'
    images = base / 'images'
    (prepared/'sessions').mkdir(parents=True)
    images.mkdir()

    ids = ['train:1', 'train:2', 'train:3', 'train:4', 'test:1', 'test:2']
    for i, stim in enumerate(ids):
        pic = np.zeros((12, 12, 3), dtype=np.uint8)
        pic[:, :, 0] = 25 + i*27 + np.arange(12)[:, None]
        pic[:, :, 1] = 40 + i*18 + np.arange(12)[None, :]
        pic[:, :, 2] = 65 + i*14
        Image.fromarray(pic).save(images/(stim.replace(':', '_')+'.png'))

    sessions = ['macaqueF_TVSD_20240112_B1',
                'macaqueF_TVSD_20240112_B2',
                'macaqueN_TVSD_20220111_B1']
    records = []
    unit_rows = []
    for session in sessions:
        animal = session[7]
        nunit = 2 if animal == 'F' else 1
        counts = np.arange(1, 6*nunit+1, dtype=np.uint16).reshape(6, nunit)
        observed = np.ones_like(counts, dtype=bool)
        if nunit == 2:
            counts[1, 1] = 0
            observed[1, 1] = False
        np.savez_compressed(prepared/'sessions'/(session+'.npz'),
                            counts=counts, observed=observed,
                            neuron_indices=np.arange(nunit, dtype=np.int32),
                            source_rows=np.arange(6, dtype=np.int64),
                            stimulus_ids=np.array(ids),
                            response_window_ms=np.array([40., 160.]))
        for i, st in enumerate(ids):
            records.append(dict(animal=animal, session=session, trial_index=i,
                                source_row=i, stimulus_id=st,
                                partition='train' if i < 3 else 'val' if i == 3 else 'test',
                                image_relative_path=st.replace(':', '_')+'.png'))
        for i in range(nunit):
            unit_rows.append(dict(session=session, animal=animal,
                                  unit_instance_id=f'{session}/A1/{i}', neuron_index=i))

    sample = pd.DataFrame.from_records(records)
    sample.to_csv(prepared/'sample_index.csv', index=False)
    pd.DataFrame(unit_rows).to_csv(prepared/'unit_to_neuron.csv', index=False)
    split_sha256 = source._split_sha256(sample)
    info = dict(version='tvsd_tracked_neurons_masked_v3', n_neurons={'F':2, 'N':1},
                n_selected_unit_instances=len(unit_rows),
                partition_images={'train':3, 'val':1, 'test':2},
                split_seed=8000, split_sha256=split_sha256,
                response_window_ms=[40.,160.], image_root=str(images))
    (prepared/'dataset_info.json').write_text(json.dumps(info))

    gray = np.concatenate([image_hwc(images/(f'train_{i}.png'), 12) for i in (1,2,3)], axis=0)
    mean, std = float(gray.mean()), float(gray.std())
    norm = dict(method='global_pixel_zscore_train_unique_images', color_mode='gray-bt601',
                image_size=12, mean=[mean], std=[std], partial=False,
                split_sha256=split_sha256)
    (prepared/'normalization.json').write_text(json.dumps(norm))

    cfg = yaml.safe_load((ROOT/'configs'/'tvsd_default.yaml').read_text())
    cfg['data']['root'] = str(prepared)
    cfg['data']['image_root'] = str(images)
    cfg['data']['dataset_module_path'] = str(SOURCE_PATH)
    cfg['data']['preprocessing']['normalization_path'] = str(prepared/'normalization.json')
    cfg['data']['preprocessing']['expected_output_hw'] = [12, 12]
    cfg['data']['split']['train_fraction'] = .75
    cfg['model']['readout']['outdims_by_animal'] = {'F':2,'N':1}
    cfg['model']['readout']['outdims'] = 3
    cfg['model']['core']['output_hw'] = [4,4]
    return cfg, source


def test_frozen_v3_adapter_end_to_end(tmp_path):
    cfg, source = make_frozen_fixture(tmp_path)
    bundle = TVSDTrainingBundle(cfg)
    assert bundle.widths == {'F':2,'N':1}
    assert bundle.normalization_stats['color_mode'] == 'gray-bt601'
    assert len(bundle.full.image_ids) == 12
    assert len(bundle.test.image_ids) == 6
    assert len(bundle.full.subset_by_image_ids(bundle.split.train_image_ids).image_ids) == 9
    assert len(bundle.full.subset_by_image_ids(bundle.split.val_image_ids).image_ids) == 3
    assert bundle.full.responses.shape == (12,3)
    assert bundle.full.mask.shape == (12,3)
    assert not np.any(bundle.full.mask[:,0] & bundle.full.mask[:,2])
    assert len(bundle.dataset('test')) == 6

    batch = next(iter(torch.utils.data.DataLoader(bundle.dataset('train'), batch_size=3)))
    assert batch['image'].shape == (3,3,12,12)
    assert batch['response'].shape == (3,3)
    assert batch['mask'].shape == (3,3)
    assert torch.equal(batch['image'][:,0], batch['image'][:,1])
    assert not bool(batch['mask'][:,2].any())  # F comes first
    # Production loader already applies normalization: adapter must NOT apply twice.
    original = source.TVSDDataset(animal='F', partition='train',
                                  prepared_root=cfg['data']['root'],
                                  image_root=cfg['data']['image_root'], image_size=12)[0]['image']
    assert torch.equal(batch['image'][0], original)
    with patch('sinz_model.model.V1ConvNextCore', FakeCore):
        model = TVSDV1Model(cfg)
        tr = bundle.full.subset_by_image_ids(bundle.split.train_image_ids)
        _initialize_readout_bias_from_targets(model, tr.responses, tr.mask)
        assert model.readout.outdims == 3
        pred = model(batch['image'])
        assert pred.shape == (3,3)
        assert bool((pred > 0).all())
        loss = training_loss(model, pred, batch['response'], batch['mask'],
                             gamma_readout=10, dataset_size=9, batch_size=3)
        assert torch.isfinite(loss['loss'])
        loss['loss'].backward()
        assert model.readout._features.grad is not None


def test_rejects_mismatched_normalization(tmp_path):
    cfg, _ = make_frozen_fixture(tmp_path)
    norm_path = Path(cfg['data']['preprocessing']['normalization_path'])
    norm = json.loads(norm_path.read_text())
    norm['split_sha256'] = 'altered'
    norm_path.write_text(json.dumps(norm))
    with pytest.raises(ValueError, match='Normalization and frozen split'):
        TVSDTrainingBundle(cfg)

