"""TVSD integration tests: tiny prepared data, frozen split, one Gaussian readout."""
import importlib.util
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
import yaml

from sinz_model.tvsd_data import TVSDTrainingBundle
from sinz_model.model import TVSDV1Model
from sinz_model.analysis import training_loss
from sinz_model.trainer import _initialize_readout_bias_from_targets


def load_source(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeCore(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv=torch.nn.Conv2d(3,80,3,1,1)
        self.output_shape=(80,4,4)
    def forward(self,x):
        return torch.nn.functional.adaptive_avg_pool2d(self.conv(x),(4,4))
    def set_trainable(self, trainable):
        for p in self.parameters(): p.requires_grad=trainable
    def regularizer(self):
        return next(self.parameters()).new_zeros(())


def test_tiny_pipeline():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    source=load_source(root/'scripts'/'tvsd_dataset.py','tvsd_test_dataset_module')
    with tempfile.TemporaryDirectory() as tmp:
        base=Path(tmp); raw=base/'raw'; images=base/'images'; mapping=base/'maps'; packed=base/'prepared'
        for d in (raw/'images',raw/'sessions',images,mapping): d.mkdir(parents=True,exist_ok=True)
        sessions=['macaqueF_TVSD_20240112_B1','macaqueF_TVSD_20240112_B2','macaqueN_TVSD_20220111_B1']
        # Two images in train/val and two repeated test images across animals.
        ids=['train:1','train:2','train:3','train:4','test:1','test:2']
        for idx,stim in enumerate(ids):
            x=np.zeros((12,12,3),dtype=np.uint8)
            x[:,:,0]=25+idx*27+np.arange(12)[:,None]
            x[:,:,1]=40+idx*18+np.arange(12)[None,:]
            x[:,:,2]=65+idx*14
            Image.fromarray(x).save(images/(stim.replace(':','_')+'.png'))
        sm=[]; pr=[]; um=[]
        for ss in sessions:
            ani=ss[7]; nunit=2 if ani=='F' else 1
            units=[f'{ss}/A1/{j}' for j in range(nunit)]
            for j,uid in enumerate(units):
                um.append(dict(unit_instance_id=uid,session=ss,animal=ani,readout_index=j,
                   readout_id=f'{ani}/{j}',train_presentations=4,test_presentations=2))
            counts=np.arange(1,nunit*6+1,dtype=np.uint16).reshape(6,nunit)
            observed=np.ones_like(counts,dtype=bool)
            if nunit>1: counts[1,1]=0; observed[1,1]=False
            np.savez_compressed(raw/'sessions'/(ss+'.npz'),trial_source_rows=np.arange(6),
                unit_instance_ids=units,split=np.array(['train']*4+['test']*2),
                image_index=np.array([1,2,3,4,1,2]),response_window_ms=np.array([40.,160.]),
                counts=counts,observed=observed)
            for row,stim in enumerate(ids):
                sp,index=stim.split(':')
                pr.append(dict(session=ss,source_row=row,stimulus_id=stim,
                               split=sp,image_index=int(index)))
        for stim in ids:
            split,index=stim.split(':')
            sm.append(dict(stimulus_id=stim,split=split,image_index=int(index),
                           resolution_status='resolved',image_relative_path=stim.replace(':','_')+'.png'))
        pd.DataFrame(pr).to_csv(raw/'images'/'presentation_to_stimulus.csv',index=False)
        pd.DataFrame(sm).to_csv(raw/'images'/'stimulus_manifest.csv',index=False)
        pd.DataFrame(um).to_csv(mapping/'unit_to_readout.csv',index=False)
        info=source.prepare_dataset(processed_root=raw,map_dir=mapping,image_root=images,
                            output_dir=packed,val_fraction=.25,seed=8000,expected_sessions=3)
        # Use pixel-streaming calibration matching the production script.
        cal=load_source(root/'scripts'/'calibrate_tvsd_normalization.py','tvsd_norm_test')
        df=pd.read_csv(packed/'sample_index.csv')
        pp=df.loc[df.partition.eq('train'),'image_relative_path'].unique()
        all_gray=[]
        for p in pp:
            with Image.open(images/p) as im:
                rgb=np.asarray(im.convert('RGB').resize((12,12),Image.Resampling.BICUBIC),dtype=np.float64)/255.
            all_gray.append(rgb[:,:,0]*.2989+rgb[:,:,1]*.5870+rgb[:,:,2]*.1140)
        mean=float(np.mean(all_gray)); std=float(np.std(all_gray))
        norm=packed/'tvsd_normalization.json'
        norm.write_text(json.dumps(dict(mean=mean,std=std,input_hw=[12,12],
             preprocessing='rgb_resize_bicubic_then_luma',source_partition='train')))
        cfg=yaml.safe_load((root/'configs'/'tvsd_default.yaml').read_text())
        cfg['data']['root']=str(packed)
        cfg['data']['image_root']=str(images)
        cfg['data']['dataset_module_path']=str(root/'scripts'/'tvsd_dataset.py')
        cfg['data']['preprocessing']['normalization_path']=str(norm)
        cfg['data']['preprocessing']['expected_output_hw']=[12,12]
        cfg['model']['readout']['outdims_by_animal']={'F':2,'N':1}
        cfg['model']['readout']['outdims']=3
        cfg['model']['core']['output_hw']=[4,4]
        cfg['data']['split']['train_fraction']=.75
        bundle=TVSDTrainingBundle(cfg)
        assert len(bundle.full.image_ids)==12
        assert len(bundle.test.image_ids)==6
        assert len(bundle.full.subset_by_image_ids(bundle.split.train_image_ids).image_ids)==9
        assert len(bundle.full.subset_by_image_ids(bundle.split.val_image_ids).image_ids)==3
        assert bundle.full.responses.shape==(12,3)
        assert bundle.full.mask.shape==(12,3)
        assert not np.any(bundle.full.mask[:,2] & bundle.full.mask[:,0])
        assert len(bundle.dataset('test'))==6
        b=next(iter(torch.utils.data.DataLoader(bundle.dataset('train'),batch_size=3)))
        assert b['image'].shape==(3,3,12,12)
        assert b['response'].shape==(3,3)
        assert b['mask'].shape==(3,3)
        assert torch.allclose(b['image'][:,0],b['image'][:,1])
        assert not bool(b['mask'][:,2].any())  # F comes first in ConcatDataset
        with patch('sinz_model.model.V1ConvNextCore',FakeCore):
            model=TVSDV1Model(cfg)
            _initialize_readout_bias_from_targets(model, bundle.full.responses, bundle.full.mask)
            assert model.readout.outdims == 3
            assert model.readout.bias.shape == (3,)
            assert model.readout._features.shape[-1] == 3
            pred=model(b['image'])
            assert pred.shape==(3,3)
            assert bool((pred>0).all())
            loss=training_loss(model,pred,b['response'],b['mask'],gamma_readout=10,
                               dataset_size=9,batch_size=3)
            assert torch.isfinite(loss['loss'])
            loss['loss'].backward()
            assert model.readout._features.grad is not None
            assert model.readout._features.grad.shape[-1] == 3
        print('TVSD INTEGRATION PASS: frozen split, single Gaussian readout, loss/backprop')

if __name__=='__main__':test_tiny_pipeline()
