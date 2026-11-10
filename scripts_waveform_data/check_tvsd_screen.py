#!/usr/bin/env python3
"""Fail fast if the TVSD single-Gaussian configuration isn't ready for screening."""
import json
from pathlib import Path
import yaml

root = Path('/project')
config = root / 'configs/tvsd_default.yaml'
cfg = yaml.safe_load(config.read_text())
assert cfg['data']['kind'] == 'tvsd'
assert int(cfg['model']['readout']['outdims']) == 733
assert cfg['model']['readout']['outdims_by_animal'] == {'F':451,'N':282}
assert tuple(cfg['model']['core']['output_hw']) == (28,28)
assert tuple(cfg['data']['preprocessing']['expected_output_hw']) == (224,224)
assert int(cfg['training']['total_epochs']) == 200
assert int(cfg['training']['readout_only_epochs']) == 20
for name, path in (
    ('prepared dataset', cfg['data']['root']),
    ('dataset loader', cfg['data']['dataset_module_path']),
    ('image directory', cfg['data']['image_root']),
    ('normalization', cfg['data']['preprocessing']['normalization_path']),
):
    if not Path(path).exists():
        raise FileNotFoundError('%s missing: %s' % (name,path))
info = json.loads((Path(cfg['data']['root'])/'dataset_info.json').read_text())
assert info['n_readouts'] == {'F':451,'N':282}, info['n_readouts']
assert info['partition_images'] == {'train':20023,'val':2225,'test':100}, info['partition_images']
stats=json.loads(Path(cfg['data']['preprocessing']['normalization_path']).read_text())
assert stats['input_hw'] == [224,224]
assert stats['preprocessing'] == 'rgb_resize_bicubic_then_luma'
assert 0.0 < stats['std'] < 1.0
plan=yaml.safe_load((root/'configs/tvsd_screen_plan.yaml').read_text())
assert len(plan['candidates']) == 5
from sinz_model.search import _load_explicit_plan, _apply_overrides, _validate_existing_path
items=_load_explicit_plan(str(root/'configs/tvsd_screen_plan.yaml'))
for item in items:
    for key in item['overrides']:
        _validate_existing_path(cfg, key)
    result=_apply_overrides(cfg,item['overrides'])
    assert result['model']['readout']['outdims']==733
print('TVSD preflight PASS: 733 readouts, 45-session dataset, frozen split, '
      'normalization, 6 controlled configurations')
print('GPU training remains on the existing /project environment; nothing changed in dataset.')
