#!/usr/bin/env python3
"""Smoke/integrity tests for tvsd_dataset.py. Run: python scripts/test_tvsd_dataset.py"""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from tvsd_dataset import TVSDDataset, prepare_dataset


class TinyPipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / 'ks4'
        self.map_dir = self.base / 'mapping'
        self.images = self.base / 'images'
        self.out = self.base / 'prepared'
        for x in [self.root / 'sessions', self.root / 'images',self.map_dir,self.images]:
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
                                     readout_index=j,readout_id=f'{animal}/{j}',
                                     train_presentations=10,test_presentations=1))
            source_rows = np.arange(11,dtype=np.int64)
            split = np.array(['train']*10+['test'])
            image_index = np.array(list(range(1,11))+[1])
            unit_ids = list(reversed(ids)) + [f'{session}/A1/not_selected']
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
        pd.DataFrame(mappings).to_csv(self.map_dir/'unit_to_readout.csv',index=False)
        present = pd.DataFrame(records)
        present.to_csv(self.root/'images/presentation_to_stimulus.csv',index=False)
        stim = present[['stimulus_id','split','image_index']].drop_duplicates().copy()
        stim['image_relative_path'] = stim.stimulus_id.str.replace(':','_',regex=False) + '.png'
        stim['resolution_status'] = 'resolved'
        stim.to_csv(self.root/'images/stimulus_manifest.csv',index=False)
        for path in stim.image_relative_path:
            Image.new('RGB',(16,16),color=(127,80,60)).save(self.images / path)

    def tearDown(self):
        self.tmp.cleanup()

    def prep(self):
        return prepare_dataset(processed_root=self.root,map_dir=self.map_dir,
                               image_root=self.images,output_dir=self.out,
                               val_fraction=.2,seed=42,expected_sessions=3)

    def test_loader_alignment_and_splits(self):
        info = self.prep()
        self.assertEqual(info['n_readouts'],{'F':2,'N':1})
        self.assertEqual(info['partition_images'],{'train':8,'val':2,'test':1})
        samples=pd.read_csv(self.out/'sample_index.csv')
        self.assertEqual(len(samples),33)
        self.assertTrue(samples.loc[samples.stimulus_id.eq('test:1'),'partition'].eq('test').all())
        self.assertEqual(samples.groupby('stimulus_id').partition.nunique().max(),1)
        ds=TVSDDataset(animal='F',partition='train',prepared_root=self.out,
                       image_root=self.images,image_size=12)
        self.assertEqual(len(ds),16)
        item=ds[0]
        self.assertEqual(tuple(item['image'].shape),(3,12,12))
        self.assertEqual(tuple(item['responses'].shape),(2,))
        self.assertTrue(item['response_mask'].any())
        self.assertTrue(torch.isfinite(item['responses']).all())
        self.assertTrue(torch.all(item['responses'][~item['response_mask']]==0))
        # Verify reverse source columns became correct readout indices.
        session=samples.loc[samples.animal.eq('F'),'session'].iloc[0]
        with np.load(self.out/'sessions'/f'{session}.npz') as packed:
            self.assertEqual(packed['readout_indices'].tolist(),[0,1])
            self.assertEqual(packed['counts'][0].tolist(),[4,3])
            self.assertEqual(packed['observed'][1].tolist(),[True,False])
        test=TVSDDataset(animal='N',partition='test',prepared_root=self.out,
                         image_root=self.images,image_size=10)
        self.assertEqual(len(test),1)
        self.assertEqual(test[0]['stimulus_id'],'test:1')

    def test_reject_simultaneous_readout_collision(self):
        p=self.map_dir/'unit_to_readout.csv';m=pd.read_csv(p)
        q=m.loc[m.session.eq('macaqueF_TVSD_20240112_B1')].index
        m.loc[q,'readout_index']=0
        m.to_csv(p,index=False)
        with self.assertRaisesRegex(ValueError,'simultaneously'):
            self.prep()

    def test_reject_wrong_unit_count(self):
        p=self.map_dir/'unit_to_readout.csv';m=pd.read_csv(p)
        m.loc[0,'train_presentations']=999
        m.to_csv(p,index=False)
        with self.assertRaisesRegex(ValueError,'training presentation count'):
            self.prep()


class RealManifestStructuralTest(unittest.TestCase):
    """Checks against the actually uploaded mapping/images audit, if available."""
    def test_uploaded_mapping_and_image_audit(self):
        folder=Path('/mnt/data')
        if not (folder/'unit_to_readout.csv').is_file() or not (folder/'images.zip').is_file():
            self.skipTest('Real inputs are not mounted')
        selected=pd.read_csv(folder/'unit_to_readout.csv')
        with zipfile.ZipFile(folder/'images.zip') as z:
            present=pd.read_csv(io.BytesIO(z.read('images/presentation_to_stimulus.csv')))
        self.assertEqual(selected.session.nunique(),45)
        self.assertEqual(selected.groupby('animal').readout_index.nunique().to_dict(),{'F':451,'N':282})
        self.assertEqual(int(selected.train_presentations.sum()),6495007)
        count=present.groupby(['session','split']).size().unstack(fill_value=0)
        maps=selected.groupby('session')[['train_presentations','test_presentations']].first()
        self.assertTrue(np.array_equal(maps.train_presentations.to_numpy(),count.loc[maps.index,'train'].to_numpy()))
        self.assertTrue(np.array_equal(maps.test_presentations.to_numpy(),count.loc[maps.index,'test'].to_numpy()))
        self.assertEqual(len(present),50496)
        present['animal'] = present.session.str.extract(r'^macaque([FN])_')[0]
        per_animal = present.groupby(['stimulus_id','animal']).size()
        splits = per_animal.index.get_level_values('stimulus_id')
        self.assertTrue((per_animal[splits.str.startswith('train:')] == 1).all())
        self.assertTrue((per_animal[splits.str.startswith('test:')] == 30).all())
        self.assertEqual(present.stimulus_id.nunique(),22348)


if __name__ == '__main__':
    unittest.main(verbosity=2)
