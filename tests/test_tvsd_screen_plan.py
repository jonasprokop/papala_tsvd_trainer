import unittest
from pathlib import Path
import yaml
from sinz_model.search import _load_explicit_plan, _apply_overrides, _validate_existing_path, _load_space

ROOT=Path(__file__).resolve().parents[1]

class TVSDScreenPlanTests(unittest.TestCase):
    def setUp(self):
        self.cfg=yaml.safe_load((ROOT/'configs/tvsd_default.yaml').read_text())
        self.plan=_load_explicit_plan(str(ROOT/'configs/tvsd_screen_plan.yaml'))

    def test_six_candidates_total(self):
        self.assertEqual(1+len(self.plan),6)
        self.assertEqual(len({p['name'] for p in self.plan}),5)

    def test_one_at_a_time_with_real_runtime_keys(self):
        self.assertEqual(self.cfg['training']['loss']['gamma_readout'],10.0)
        self.assertEqual(self.cfg['model']['readout']['init_sigma'],0.6)
        for p in self.plan:
            self.assertEqual(len(p['overrides']),1)
            key=next(iter(p['overrides']))
            _validate_existing_path(self.cfg,key)
            modified=_apply_overrides(self.cfg,p['overrides'])
            self.assertEqual(modified['model']['readout']['outdims'],733)
            self.assertEqual(modified['training']['total_epochs'],200)
            self.assertEqual(modified['data']['kind'],'tvsd')

    def test_normalization_paths_and_random_search_remain_available(self):
        self.assertEqual(self.cfg['data']['dataset_module_path'],'/project/scripts/tvsd_dataset.py')
        self.assertEqual(self.cfg['data']['preprocessing']['normalization_path'],'/project/configs/tvsd_normalization.json')
        original=_load_space(str(ROOT/'configs/tvsd_search_space.yaml'))
        self.assertIn('training.loss.gamma_readout', original)
        self.assertIn('model.readout.init_sigma', original)

if __name__=='__main__':
    unittest.main()
