"""Papale/BBScore MUA adapter; entirely synthetic, no raw TVSD files needed."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from evaluation import mua_adapter as ma


class MUAAdapterTest(unittest.TestCase):
    def test_hdf5_axes_metadata_and_measured_reliability(self):
        class FakeFile:
            def __init__(self, data):
                self.data = data
            def __enter__(self):
                return self.data
            def __exit__(self, *_):
                return False

        def group(paths):
            refs = {f'path{i}': np.fromiter((ord(c) for c in p), dtype=np.uint16)
                    for i, p in enumerate(paths)}
            refs['things_path'] = np.array(list(refs), dtype=object)
            return refs

        train = np.zeros((3, 1024), np.float32)
        train[:, 0] = [-1, 0, 1]
        train[:, 511] = [1, 2, 3]
        train[:, 512] = [50, 50, 50]  # Must NOT leak into V1.
        reps = np.zeros((2, 2, 1024), np.float32)
        reps[:, :, 0] = [[1, 2], [3, 4]]
        test = reps.mean(axis=0)
        reliab = np.zeros((3, 1024), np.float32)
        reliab[:, 0] = [0.2, 0.4, 0.6]
        fake = {'meta.mat': {
            'train_imgs': group(['a.png','b.png','c.png']),
            'test_imgs': group(['d.png','e.png'])},
            'data.mat': {
                'train_MUA': train, 'test_MUA': test,
                'test_MUA_reps': reps, 'reliab': reliab}}
        opened = []
        def open_mat(path, mode, **kwargs):
            opened.append((str(path), mode, kwargs.get('driver')))
            return FakeFile(fake[str(path)])
        module = SimpleNamespace(File=open_mat)
        with patch.dict(sys.modules, {'h5py': module}):
            out = ma._read_monkey(Path('data.mat'), Path('meta.mat'),
                                  animal='F', region='V1')
        self.assertEqual(out[:2], (['a.png','b.png','c.png'], ['d.png','e.png']))
        self.assertEqual(opened, [('meta.mat', 'r', 'stdio'),
                                  ('data.mat', 'r', 'stdio')])
        self.assertEqual(out[2].shape, (3,512))
        self.assertEqual(out[4].shape, (2,2,512))
        self.assertEqual(float(out[2][0,0]), -1.0)
        self.assertEqual(float(out[2][0,-1]), 1.0)
        measured, oracle = out[-1]
        self.assertAlmostEqual(float(measured['mean'][0]), 0.4)
        self.assertEqual(measured['n_pairs'], 3)
        self.assertIsNone(oracle)

    def test_source_channel_ranges_match_bbscore(self):
        self.assertEqual(ma.CHANNELS['V1'], {'F': (0, 512), 'N': (0, 512)})
        self.assertEqual(ma.CHANNELS['V4'], {'F': (832, 1024), 'N': (512, 768)})
        self.assertEqual(ma.CHANNELS['IT'], {'F': (512, 832), 'N': (768, 1024)})

    def test_test_repetitions_keep_image_and_repeat_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            train = np.array([[0.5, -1.2], [1.5, 2.0], [3.0, 4.0]], np.float32)
            reps = np.array([[[10, 11], [20, 21]],
                             [[12, 13], [22, 23]],
                             [[14, 15], [24, 25]]], np.float32)  # [R,I,C]
            rows, observed, missing = ma._pack_monkey(
                Path(temporary), animal='F', region='V1',
                train_paths=['a.png', 'b.png', 'c.png'],
                test_paths=['t1.png', 't2.png'], train=train,
                reps=reps, partitions={1: 'train', 2: 'val', 3: 'train'})
            with np.load(Path(temporary)/'macaqueF_TVSD_normMUA_V1.npz') as packed:
                np.testing.assert_array_equal(packed['counts'][:3], train)
                np.testing.assert_array_equal(packed['counts'][3:],
                                              [[10,11], [12,13], [14,15],
                                               [20,21], [22,23], [24,25]])
                self.assertEqual(packed['counts'].dtype, np.float32)
                self.assertTrue(packed['observed'].all())
                self.assertTrue(np.array_equal(packed['response_window_ms'], [25,125]))
            self.assertEqual([r['stimulus_id'] for r in rows[3:]],
                             ['test:1']*3+['test:2']*3)
            self.assertEqual((observed, missing), (18, 0))

    def test_nan_cell_becomes_masked_zero_not_dropped_channel(self):
        with tempfile.TemporaryDirectory() as temporary:
            train = np.array([[0.5], [np.nan], [-0.1]], np.float32)
            reps = np.ones((2,2,1), np.float32)
            ma._pack_monkey(Path(temporary), animal='F', region='V1',
                            train_paths=['a','b','c'], test_paths=['d','e'],
                            train=train, reps=reps,
                            partitions={1:'train',2:'train',3:'val'})
            with np.load(Path(temporary)/'macaqueF_TVSD_normMUA_V1.npz') as dat:
                self.assertEqual(float(dat['counts'][1,0]), 0.0)
                self.assertFalse(dat['observed'][1,0])
                self.assertAlmostEqual(float(dat['counts'][2,0]), -0.1)

    def test_brainscore_mean_keeps_nan_and_oracle_is_read(self):
        class FakeFile:
            def __init__(self, data):
                self.data = data
            def __enter__(self):
                return self.data
            def __exit__(self, *_):
                return False

        def group(paths):
            refs = {f'path{i}': np.fromiter((ord(c) for c in p), dtype=np.uint16)
                    for i, p in enumerate(paths)}
            refs['things_path'] = np.array(list(refs), dtype=object)
            return refs
        reps = np.zeros((2, 2, 1024))
        reliab = np.full((3, 1024), .5)
        reliab[1, 7] = np.nan                     # one NaN pair on channel 7
        reliab[:, 8] = np.nan                     # no defined pairs on channel 8
        oracle = np.linspace(0, 1, 1024)[None, :]  # HDF5 [1, 1024]
        fake = {'meta.mat': {'train_imgs': group(['a.png']), 'test_imgs': group(['d.png', 'e.png'])},
                'data.mat': {'train_MUA': np.zeros((1, 1024)), 'test_MUA': reps.mean(0),
                             'test_MUA_reps': reps, 'reliab': reliab, 'oracle': oracle}}
        module = SimpleNamespace(File=lambda path, _, **kwargs: FakeFile(fake[str(path)]))
        with patch.dict(sys.modules, {'h5py': module}):
            out = ma._read_monkey(Path('data.mat'), Path('meta.mat'), animal='N', region='V4')
        measured, orc = out[-1]
        # V4 for N is channels 512:768; channel 7 is outside, so all finite there.
        self.assertTrue(np.isfinite(measured['mean']).all())
        with patch.dict(sys.modules, {'h5py': module}):
            out = ma._read_monkey(Path('data.mat'), Path('meta.mat'), animal='N', region='V1')
        measured, orc = out[-1]
        self.assertTrue(np.isnan(measured['mean'][7]))           # Brain-Score: NaN, filtered
        self.assertAlmostEqual(measured['nanmean'][7], .5)       # kept for inspection only
        self.assertTrue(np.isnan(measured['mean'][8]))
        self.assertTrue(np.isnan(measured['nanmean'][8]))        # no empty-slice warning
        np.testing.assert_allclose(orc, oracle.ravel()[:512])

    def test_complete_export_frozen_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            paths = [base/name for name in ('F.mat','N.mat','Fm.mat','Nm.mat')]
            for path in paths:
                path.write_bytes(b'synthetic')
            images = base/'images'
            images.mkdir()
            for name in ('a.png','b.png','c.png'):
                (images/name).write_bytes(b'img')
            # Real Papale metadata omits the THINGS object_images/ subfolder.
            (images/'object_images').mkdir()
            (images/'object_images'/'x.png').write_bytes(b'img')
            # 512 V1 channels per monkey; signed normalized values are allowed.
            train = np.tile(np.array([[-0.5],[0.5],[1.0]],np.float32),(1,512))
            reps = np.tile(np.array([[[0.4]], [[0.6]]],np.float32),(1,1,512))
            made = (['a.png','b.png','c.png'], ['x.png'], train,
                    reps.mean(axis=0), reps,
                    (dict(mean=np.zeros(512), nanmean=np.zeros(512), n_pairs=1), np.zeros(512)))
            dest = base/'new_mua_cohort'
            with patch.object(ma, '_read_monkey', return_value=made):
                info = ma.prepare(f_mua=paths[0], n_mua=paths[1],
                                  f_metadata=paths[2], n_metadata=paths[3],
                                  image_root=images, output_dir=dest,
                                  val_fraction=1/3)
                with self.assertRaises(FileExistsError):
                    ma.prepare(f_mua=paths[0], n_mua=paths[1],
                               f_metadata=paths[2], n_metadata=paths[3],
                               image_root=images, output_dir=dest)
            self.assertEqual(info['n_neurons'], {'F':512, 'N':512})
            self.assertEqual(info['partition_images']['test'], 1)
            self.assertFalse((dest/'normalization.json').exists())
            self.assertEqual(len(pd.read_csv(dest/'unit_to_neuron.csv')), 1024)
            index = pd.read_csv(dest/'sample_index.csv')
            self.assertEqual(len(index), 2*(3+2))
            self.assertEqual(set(index.loc[index.partition.eq('test'),
                                           'image_relative_path']),
                             {'object_images/x.png'})
            self.assertEqual(ma._split_sha256(index), info['split_sha256'])
            self.assertTrue(np.isfinite(pd.read_csv(dest/'neuron_index.csv')
                                        ['papale_reliab_measured']).all())
            self.assertEqual(json.loads((dest/'dataset_info.json').read_text())['version'],
                             'tvsd_papale_normmua_v1')


if __name__ == '__main__':
    unittest.main()
