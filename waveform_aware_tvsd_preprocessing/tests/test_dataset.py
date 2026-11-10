#!/usr/bin/env python3
"""Synthetic integrity tests: python -m unittest discover -s tests -v"""
import io
import json
import tempfile
import unittest
import zipfile
import sys
from unittest.mock import patch
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'scripts'))
from dataset import TVSDDataset, prepare_dataset
from select_tracks import select_tracks
from calibrate_normalization import calibrate
from image_transform import image_hwc, WEIGHTS
from audit_images import main as audit_images_main


class TinyPipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / 'ks4'
        self.images = self.base / 'images'
        self.out = self.base / 'prepared'
        for x in [self.root / 'sessions', self.root / 'images',self.images]:
            x.mkdir(parents=True)
        sessions = ['macaqueF_TVSD_20240112_B1','macaqueF_TVSD_20240112_B2',
                    'macaqueN_TVSD_20220111_B1']
        mappings = []
        records = []
        for session in sessions:
            animal = session[7]
            ids = [f'{session}/A1/u0'] if animal == 'N' else [f'{session}/A1/u0', f'{session}/A1/u1']
            for j,uid in enumerate(ids):
                mappings.append(dict(unit_instance_id=uid,session=session,animal=animal,
                                     array=1,electrode_id=j+1,firing_rate_hz=5.0+j))
            source_rows = np.arange(11,dtype=np.int64)
            split = np.array(['train']*10+['test'])
            image_index = np.array(list(range(1,11))+[1])
            unit_ids = list(reversed(ids))
            counts = np.ones((11,len(unit_ids)),dtype=np.uint16)
            for j in range(len(unit_ids)):
                counts[:,j] = j + 3
            mask = np.ones_like(counts,dtype=bool)
            mask[1, 0] = False
            counts[1, 0] = 0
            np.savez_compressed(self.root / 'sessions'/f'{session}.npz',
                                trial_source_rows=source_rows,unit_instance_ids=unit_ids,
                                split=split,image_index=image_index,
                                response_window_ms=np.array([40.,160.]),
                                counts=counts,observed=mask,
                                spike_times_s=np.array([1.,2.,3.]),spike_offsets=np.array([0,3]))
            for k in range(11):
                is_test = k == 10
                records.append(dict(session=session,source_row=k,
                                    stimulus_id=('test:1' if is_test else f'train:{k+1}'),
                                    split=('test' if is_test else 'train'),
                                    image_index=1 if is_test else k+1))
        pd.DataFrame(mappings).to_csv(self.root/'unit_instances.csv',index=False)
        present = pd.DataFrame(records)
        present.to_csv(self.root/'images/presentation_to_stimulus.csv',index=False)
        stim = present[['stimulus_id','split','image_index']].drop_duplicates().copy()
        stim['image_relative_path'] = stim.stimulus_id.str.replace(':','_',regex=False) + '.png'
        stim['resolution_status'] = 'resolved'
        stim.to_csv(self.root/'images/stimulus_manifest.csv',index=False)
        for k,path in enumerate(stim.image_relative_path):
            rgb=np.zeros((16,16,3),dtype=np.uint8)
            rgb[:,:,0] = 20 + k*17
            rgb[:,:,1] = 40 + (k%4)*23
            rgb[:,:,2] = 60 + (k%6)*13
            Image.fromarray(rgb).save(self.images / path)

    def tearDown(self):
        self.tmp.cleanup()

    def write_track_proposals(self):
        raw = pd.read_csv(self.root/'unit_instances.csv')
        # Two longitudinal F neurons with recordings in B1 and B2; one N singleton.
        raw['candidate_joint_track_id'] = raw.unit_instance_id.map(
            lambda uid: 'N_single' if 'macaqueN' in uid else 'F_' + uid.rsplit('/',1)[-1])
        diag = self.root/'identity_diagnostics/joint'
        diag.mkdir(parents=True,exist_ok=True)
        raw[['unit_instance_id','candidate_joint_track_id']].to_csv(
            diag/'candidate_joint_unit_membership.csv',index=False)
        pd.DataFrame([dict(candidate_joint_track_id=tid, min_pairwise_waveform_r=(0.99 if tid.startswith('F_') else float('nan')))
                      for tid in sorted(raw.candidate_joint_track_id.unique())]).to_csv(
            diag/'candidate_joint_track_quality.csv',index=False)

    def prep(self):
        self.write_track_proposals()
        select_tracks(root=self.root, min_recordings=1, min_train=0)
        return prepare_dataset(processed_root=self.root,
                               image_root=self.images,output_dir=self.out,
                               val_fraction=.2,seed=42,expected_sessions=3)

    def test_loader_alignment_and_splits(self):
        info = self.prep()
        self.assertEqual(info['n_neurons'],{'F':2,'N':1})
        self.assertEqual(info['partition_images'],{'train':8,'val':2,'test':1})
        samples=pd.read_csv(self.out/'sample_index.csv')
        self.assertEqual(len(samples),33)
        self.assertTrue(samples.loc[samples.stimulus_id.eq('test:1'),'partition'].eq('test').all())
        self.assertEqual(samples.groupby('stimulus_id').partition.nunique().max(),1)
        norm=calibrate(self.out,self.images,image_size=12)
        self.assertEqual(norm['color_mode'],'gray-bt601')
        ds=TVSDDataset(animal='F',partition='train',prepared_root=self.out,
                       image_root=self.images,image_size=12)
        self.assertEqual(len(ds),16)
        item=ds[0]
        self.assertEqual(tuple(item['image'].shape),(3,12,12))
        self.assertTrue(torch.equal(item['image'][0],item['image'][1]))
        self.assertTrue(torch.isfinite(item['image']).all())
        self.assertEqual(tuple(item['responses'].shape),(2,))
        self.assertTrue(item['response_mask'].any())
        self.assertTrue(torch.isfinite(item['responses']).all())
        self.assertTrue(torch.all(item['responses'][~item['response_mask']]==0))
        # Verify reverse source columns became correct unit indices.
        session=samples.loc[samples.animal.eq('F'),'session'].iloc[0]
        with np.load(self.out/'sessions'/f'{session}.npz') as packed:
            self.assertEqual(packed['neuron_indices'].tolist(),[0,1])
            self.assertEqual(packed['counts'][0].tolist(),[4,3])
            self.assertEqual(packed['observed'][1].tolist(),[True,False])
        test=TVSDDataset(animal='N',partition='test',prepared_root=self.out,
                         image_root=self.images,image_size=12)
        self.assertEqual(len(test),1)
        self.assertEqual(test[0]['stimulus_id'],'test:1')

    def test_subset_normalization_is_diagnostic_only(self):
        self.prep()
        complete = calibrate(self.out, self.images, image_size=12)
        before = (self.out / 'normalization.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'explicit --output'):
            calibrate(self.out, self.images, image_size=12, max_images=3)
        with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
            calibrate(self.out, self.images, image_size=12, max_images=3,
                      output=self.out / 'normalization.json')
        preview = self.out / 'diagnostic_3_images.json'
        subset = calibrate(self.out, self.images, image_size=12, max_images=3,
                           output=preview)
        self.assertTrue(subset['partial'])
        self.assertEqual(subset['n_images'], 3)
        self.assertEqual((self.out / 'normalization.json').read_bytes(), before)
        self.assertFalse(complete['partial'])
        with self.assertRaisesRegex(ValueError, 'Partial normalization'):
            TVSDDataset(animal='F', partition='train', prepared_root=self.out,
                        image_root=self.images, image_size=12,
                        normalization_path=preview)

    def test_overwrite_invalidates_normalization_and_different_split(self):
        self.prep()
        calibrate(self.out, self.images, image_size=12)
        old = (self.out / 'normalization.json').read_bytes()
        old_split = json.loads((self.out / 'dataset_info.json').read_text())['split_sha256']
        new = prepare_dataset(processed_root=self.root, image_root=self.images,
                              output_dir=self.out, val_fraction=.2, seed=43,
                              expected_sessions=3, overwrite=True)
        self.assertNotEqual(new['split_sha256'], old_split)
        self.assertFalse((self.out / 'normalization.json').exists())
        # Even restoring an old normalization file is detected, so stale stats
        # can never silently be reused after an image-level split change.
        (self.out / 'normalization.json').write_bytes(old)
        with self.assertRaisesRegex(ValueError, 'split fingerprint'):
            TVSDDataset(animal='F', partition='train', prepared_root=self.out,
                        image_root=self.images, image_size=12)
        norm = calibrate(self.out, self.images, image_size=12)
        self.assertEqual(norm['split_sha256'], new['split_sha256'])
        ds = TVSDDataset(animal='F', partition='train', prepared_root=self.out,
                         image_root=self.images, image_size=12)
        self.assertTrue(torch.isfinite(ds[0]['image']).all())

    def test_image_audit_rejects_non_square_sources(self):
        trials = self.root / 'trials'
        trials.mkdir()
        obs = pd.read_csv(self.root / 'images/presentation_to_stimulus.csv')
        stim = pd.read_csv(self.root / 'images/stimulus_manifest.csv')
        paths = stim.set_index('stimulus_id').image_relative_path
        obs['THINGS_image_path'] = obs.stimulus_id.map(paths)
        obs['onset_s'] = obs.groupby('session').cumcount().astype(float)
        for session, rows in obs.groupby('session'):
            rows.to_csv(trials / f'{session}.csv', index=False)
        command = ['audit_images.py', '--processed-root', str(self.root),
                   '--images-root', str(self.images), '--expected-sessions', '3']
        with patch.object(sys, 'argv', command), redirect_stdout(io.StringIO()):
            audit_images_main()
        nonsquare = pd.read_csv(self.root / 'images/nonsquare_images.csv')
        self.assertEqual(len(nonsquare), 0)
        bad = self.images / paths.iloc[0]
        Image.fromarray(np.zeros((16, 18, 3), dtype=np.uint8)).save(bad)
        with patch.object(sys, 'argv', command), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'non-square THINGS images'):
                audit_images_main()
        violations = pd.read_csv(self.root / 'images/nonsquare_images.csv')
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations.width.iloc[0], 18)
        self.assertEqual(violations.height.iloc[0], 16)

    def test_loader_detects_modified_sample_index(self):
        self.prep()
        calibrate(self.out, self.images, image_size=12)
        index_path = self.out/'sample_index.csv'
        table = pd.read_csv(index_path)
        table.loc[table.stimulus_id.eq('train:1'), 'partition'] = 'val'
        table.to_csv(index_path, index=False)
        with self.assertRaisesRegex(ValueError, 'frozen split fingerprint'):
            TVSDDataset(animal='F', partition='train', prepared_root=self.out,
                        image_root=self.images, image_size=12)

    def test_configurable_response_window_consistent_across_sessions(self):
        for path in (self.root / 'sessions').glob('*.npz'):
            with np.load(path, allow_pickle=False) as data:
                values = {key:data[key] for key in data.files}
            values['response_window_ms'] = np.array([60., 180.])
            np.savez_compressed(path, **values)
        summary = self.prep()
        self.assertEqual(summary['response_window_ms'], [60., 180.])
        calibrate(self.out, self.images, image_size=12)
        ds = TVSDDataset(animal='F', partition='train', prepared_root=self.out,
                         image_root=self.images, image_size=12)
        self.assertGreater(len(ds), 0)

    def test_reject_inconsistent_response_windows(self):
        path = self.root/'sessions/macaqueN_TVSD_20220111_B1.npz'
        with np.load(path, allow_pickle=False) as data:
            values = {key:data[key] for key in data.files}
        values['response_window_ms'] = np.array([60., 180.])
        np.savez_compressed(path, **values)
        with self.assertRaisesRegex(ValueError, 'inconsistent spike window'):
            self.prep()

    def test_reject_duplicate_unit_instance(self):
        p=self.root/'unit_instances.csv';m=pd.read_csv(p)
        m.loc[1,'unit_instance_id']=m.loc[0,'unit_instance_id']
        m.to_csv(p,index=False)
        with self.assertRaisesRegex(ValueError,'Identity membership must cover'):
            self.prep()

    def test_reject_unknown_unit_in_source_inventory(self):
        p=self.root/'unit_instances.csv';m=pd.read_csv(p)
        m.loc[0,'unit_instance_id']='unknown'
        m.to_csv(p,index=False)
        with self.assertRaisesRegex(ValueError,'indexed unit missing'):
            self.prep()

    def test_provisional_tracks_are_neuron_columns(self):
        info=self.prep()
        self.assertEqual(info['n_neurons'], {'F':2,'N':1})
        mapping=pd.read_csv(self.out/'unit_to_neuron.csv')
        self.assertEqual(len(mapping),5)
        self.assertEqual(mapping[mapping.animal.eq('F')].neuron_index.nunique(),2)
        self.assertEqual(mapping[mapping.animal.eq('F')].groupby('neuron_index').session.nunique().tolist(),[2,2])
        index=pd.read_csv(self.out/'neuron_index.csv')
        self.assertEqual(len(index),3)

    def test_first_only_discards_later_recordings_in_prepared_dataset(self):
        self.write_track_proposals()
        select_tracks(root=self.root, min_recordings=1,min_train=0,
                      reappearance='first-only')
        summary = prepare_dataset(processed_root=self.root,image_root=self.images,
                                  output_dir=self.out,expected_sessions=3,
                                  val_fraction=.2,seed=42)
        self.assertEqual(summary['n_neurons'],{'F':2,'N':1})
        self.assertEqual(summary['n_presentations'],22)
        self.assertNotIn('macaqueF_TVSD_20240112_B2', summary['session_counts'])
        calibrate(self.out,self.images,image_size=12)
        a=TVSDDataset(animal='F',partition='train',prepared_root=self.out,
                      image_root=self.images,image_size=12)
        self.assertEqual(len(a),8)
        self.assertEqual(a.n_neurons,2)

    def test_exclude_repeated_requires_min_recordings_one(self):
        # Validate the incompatible defaults BEFORE loading any files or
        # possibly leaving the user with a misleading empty selection.
        with self.assertRaisesRegex(ValueError, r'exclude-repeated requires --min-recordings 1'):
            select_tracks(root=self.root, reappearance='exclude-repeated')
        with self.assertRaisesRegex(ValueError, r'exclude-repeated requires --min-recordings 1'):
            select_tracks(root=self.root, reappearance='exclude-repeated',
                          keep_all_tracks=True)
        self.write_track_proposals()
        result = select_tracks(root=self.root, reappearance='exclude-repeated',
                               min_recordings=1, min_train=0)
        self.assertEqual(result['n_selected_neurons'], 1)
        index = pd.read_csv(self.root/'identity_diagnostics/joint/selection/neuron_index.csv')
        self.assertEqual(index.candidate_joint_track_id.tolist(), ['N_single'])
        self.assertEqual(index.n_kept_train_presentations.tolist(), [10])

    def test_first_only_applies_min_train_to_kept_session(self):
        self.write_track_proposals()
        selected = select_tracks(root=self.root, reappearance='merge',
                                 min_recordings=1, min_train=15)
        self.assertEqual(selected['n_selected_neurons'], 2)
        included = pd.read_csv(self.root/'identity_diagnostics/joint/selection/neuron_index.csv')
        self.assertEqual(sorted(included.n_train_presentations.tolist()), [20, 20])
        self.assertEqual(sorted(included.n_kept_train_presentations.tolist()), [20, 20])
        # Both F tracks have 20 train presentations across two sessions, but
        # only 10 in the first session; N's singleton also has 10.
        with self.assertRaisesRegex(ValueError, 'No tracks remain'):
            select_tracks(root=self.root, reappearance='first-only',
                          min_recordings=1, min_train=15)
        result = select_tracks(root=self.root, reappearance='first-only',
                               min_recordings=1, min_train=10)
        self.assertEqual(result['n_selected_neurons'], 3)
        index = pd.read_csv(self.root/'identity_diagnostics/joint/selection/neuron_index.csv')
        self.assertEqual(sorted(index.n_train_presentations.tolist()), [10, 20, 20])
        self.assertEqual(sorted(index.n_kept_train_presentations.tolist()), [10, 10, 10])
        self.assertEqual(sorted(index.n_kept_test_presentations.tolist()), [1, 1, 1])
        self.assertEqual(sorted(index.n_kept_recordings.tolist()), [1, 1, 1])
        audit = pd.read_csv(self.root/'identity_diagnostics/joint/selection/track_selection_audit.csv')
        self.assertTrue((audit.n_kept_train_presentations >= 10).all())
        # 10 counts training+validation images; only prepared dataset defines
        # the final training split. Identity selection never uses test images.

    def test_stale_identity_selection_is_rejected(self):
        self.write_track_proposals()
        select_tracks(root=self.root,min_recordings=1,min_train=0)
        membership_path=self.root/'identity_diagnostics/joint/candidate_joint_unit_membership.csv'
        mapping=pd.read_csv(membership_path)
        mapping.loc[0,'candidate_joint_track_id']='changed'
        mapping.to_csv(membership_path,index=False)
        with self.assertRaisesRegex(ValueError, 'Selected neurons are stale'):
            prepare_dataset(processed_root=self.root,image_root=self.images,
                            output_dir=self.out,expected_sessions=3)

    def test_missing_track_selection_is_error(self):
        with self.assertRaisesRegex(FileNotFoundError, 'run select_tracks.py'):
            prepare_dataset(processed_root=self.root,image_root=self.images,
                            output_dir=self.out,expected_sessions=3)

    def test_reappearance_policies_and_custom_quality_cutoffs(self):
        self.write_track_proposals()
        selected = select_tracks(root=self.root, min_recordings=1, min_train=0,
                                 reappearance='merge')
        self.assertEqual(selected['n_selected_neurons'],3)
        self.assertEqual(selected['n_selected_unit_instances'],5)
        selected = select_tracks(root=self.root, min_recordings=1, min_train=0,
                                 reappearance='first-only')
        self.assertEqual(selected['n_selected_neurons'],3)
        self.assertEqual(selected['n_selected_unit_instances'],3)
        selected = select_tracks(root=self.root, min_recordings=1, min_train=0,
                                 reappearance='exclude-repeated')
        self.assertEqual(selected['n_selected_neurons'],1)
        self.assertEqual(selected['n_selected_unit_instances'],1)
        # Singleton has undefined waveform correlation, but is kept if requested.
        selection=pd.read_csv(self.root/'identity_diagnostics/joint/selection/neuron_index.csv')
        self.assertEqual(selection.candidate_joint_track_id.tolist(),['N_single'])
        # The original conservative defaults remain enabled and configurable.
        with self.assertRaisesRegex(ValueError, 'No tracks remain'):
            select_tracks(root=self.root)



class NormalizationTest(unittest.TestCase):
    def test_explicit_color_weights_and_full_field(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'img.png'
            im=np.zeros((10,20,3),dtype=np.uint8)
            im[:,:10,:]=[255,0,0]
            im[:,10:,:]=[0,0,255]
            Image.fromarray(im).save(path)
            a=image_hwc(path,224)  # MATLAB values are the implicit default
            self.assertTrue(np.array_equal(a,image_hwc(path,224,'gray-bt601')))
            b=image_hwc(path,224,'gray-skimage')
            self.assertEqual(a.shape,(224,224,1))
            self.assertAlmostEqual(float(a[100,10,0]),0.2989,places=4)
            self.assertAlmostEqual(float(b[100,10,0]),0.2125,places=4)
            self.assertAlmostEqual(float(a[100,-10,0]),0.1140,places=4)
            self.assertNotAlmostEqual(float(a.mean()),float(b.mean()),places=3)
            self.assertEqual(image_hwc(path,224,'rgb').shape,(224,224,3))

    def test_rgb_normalization_mode(self):
        t=TinyPipelineTest()
        t.setUp()
        try:
            t.prep()
            norm=calibrate(t.out,t.images,image_size=224,mode='rgb')
            self.assertEqual(len(norm['mean']),3)
            ds=TVSDDataset(animal='F',partition='val',prepared_root=t.out,
                           image_root=t.images,image_size=224)
            self.assertEqual(tuple(ds[0]['image'].shape),(3,224,224))
            self.assertTrue(torch.isfinite(ds[0]['image']).all())
        finally:
            t.tearDown()

if __name__ == '__main__':
    unittest.main(verbosity=2)
