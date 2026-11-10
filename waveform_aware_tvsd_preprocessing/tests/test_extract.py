"""Synthetic NIX + trial-metadata test; no actual experimental files included."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from extract_ks4 import process_session,rebuild_inventory,read_nix_units

class NixExtractionTest(unittest.TestCase):
    def test_nix_without_class_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            session='macaqueF_TVSD_20240112_B1'
            source=root/session/'spikes_filtered'
            source.mkdir(parents=True)
            nix=source/(session+'_Array1_spikes_KS4_filtered.nix')
            trial=pd.DataFrame({'Trial_start_s':[1.,2.,3.,4.,5.,6.],
                'Success':[1]*6,'train_or_test_dataset':['train']*5+['test'],
                'THINGS_Train_Image_Index':[1,2,3,4,5,0],
                'THINGS_Test_Image_Index':[0,0,0,0,0,1],
                'THINGS_image_path':[f'img{i}.png' for i in range(6)]})
            trial.to_csv(root/session/(session+'_trial_metadata.csv'),index=False)
            uid='neo.spiketrain.fake123'
            with h5py.File(nix,'w') as f:
                b=f.create_group('metadata/neo.block.0/sections/neo.segment.0/sections/'+uid+'/properties')
                props={'nix_name':np.bytes_(uid),'Electrode_ID':np.array([5]),
                       'Area':np.bytes_('V1'),'Array_ID':np.array([1]),
                       't_start':np.array([0.]),'t_stop':np.array([10.]),
                       'firing_rate':np.array([15.]),'train_order_original':np.array([0]),
                       'avg_wf_zscored':np.linspace(-1,1,90,dtype='f4'),
                       'avg_wf':np.linspace(-80,50,90,dtype='f4')}
                for k,v in props.items():
                    g=b.create_group(k);g.create_dataset('value',data=v)
                f.create_dataset('data/neo.block.0/data_arrays/'+uid+'.times/data',
                                 data=np.array([1.050,1.060,2.050,2.055,4.050,6.050]))
            # Set properties to the original NIX layout: datasets inside each property group
            # read_nix_units expects p[k][()] and therefore original mock must be datasets.
            with h5py.File(nix,'a') as f:
                b=f['metadata/neo.block.0/sections/neo.segment.0/sections/'+uid+'/properties']
                for k in list(b):
                    data=b[k]['value'][()];del b[k];b.create_dataset(k,data=data)
            u=list(read_nix_units(nix,1,'V1'))
            self.assertEqual(len(u),1)
            out=root/'processed'
            manifest=pd.DataFrame([{'session':session,'array':1,'area':'V1','units':1,
                                    'source_path':str(nix)}])
            process_session(session,manifest,out,np.array([40.,160.]),'V1')
            rebuild_inventory(manifest,out,np.array([40.,160.]))
            inv=pd.read_csv(out/'unit_instances.csv')
            self.assertEqual(len(inv),1)
            self.assertEqual(inv.waveform_valid.iloc[0],True)
            self.assertFalse(any('class' in k.lower() for k in inv.columns))
            with np.load(out/'sessions'/f'{session}.npz') as z:
                self.assertEqual(z['counts'].shape,(6,1))
                self.assertEqual(z['counts'].reshape(-1).tolist(),[2,2,0,1,0,1])
                self.assertTrue(z['observed'].all())

if __name__=='__main__':unittest.main()
