"""Run in the repository root after preparation and normalization."""
from pathlib import Path
import sys
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dataset import TVSDDataset

ROOT=Path('/project/data/processed/jorge_v1/model_dataset')

for animal in ('F','N'):
    splits={part:TVSDDataset(animal=animal,partition=part,prepared_root=ROOT,image_size=224)
            for part in ('train','val','test')}
    print(animal,{k:len(v) for k,v in splits.items()},'provisional tracked neurons',splits['train'].n_neurons)
    batch=next(iter(DataLoader(splits['train'],batch_size=8,num_workers=0)))
    print('image',tuple(batch['image'].shape),
          'spike counts',tuple(batch['responses'].shape),
          'valid mask',tuple(batch['response_mask'].shape))
    # Always compute losses only where batch['response_mask'] is True.
    # An unobserved neuron is not a zero-spike observation.
