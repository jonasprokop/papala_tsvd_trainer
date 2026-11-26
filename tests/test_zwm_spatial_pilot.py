"""Small synthetic checks for the separate spatial-token pilot.

Run on MetaCentrum's Python 3.10 environment with torch installed.
No model download, GPU, neural dataset or BBScore checkout is needed.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from transformer_model.benchmark_spatial_pilot import (
    FEATURE_DIM, _feature_chunks, _load_features, _load_original_torch_ridge)


class SpatialPilotTests(unittest.TestCase):
    def test_original_bbscore_class_loaded_without_unrelated_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'utils.py'
            source.write_text('import a_nonexistent_jepa_package\n'
                              'class TorchRidgeCV:\n'
                              '    def __init__(self):\n'
                              '        self.value = 7\n'
                              '    def fit(self, x, y):\n'
                              '        return self\n'
                              '    def predict(self, x):\n'
                              '        return np.asarray(x) + self.value\n')
            original = _load_original_torch_ridge(source)
            np.testing.assert_array_equal(original().predict([1, 2]), [8, 9])

    def test_layer_chunks_are_flattened_and_resumable(self):
        class FakeExtractor:
            calls = 0

            def __init__(self, model_ref, *, repo_root, layer, pooling):
                self.assert_pooling = pooling

            def extract(self, image):
                self.__class__.calls += 1
                value = np.asarray(image, dtype=np.float32).mean()
                return np.full((1, 2 * FEATURE_DIM), value, dtype=np.float32)

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids = ['train:a', 'train:b', 'test:c']
            for i, name in enumerate(ids):
                Image.new('RGB', (2, 2), (i * 20, i * 20, i * 20)).save(root / (name + '.png'))
            cfg = {'data': {'image_root': str(root)},
                   'model': {'backbone': {'model_ref': 'fake', 'repo_root': str(root)}}}
            protocol = {'train_ids': ids[:2], 'test_ids': ids[2:], 'n_train': 2,
                        'layer': 'h.6', 'chunk_size': 2,
                        'image_sources': {name: [name + '.png', 0, 0] for name in ids}}
            with patch('transformer_model.backbones.ZWMFeatures', FakeExtractor):
                chunks = list(_feature_chunks(root/'pilot', protocol, cfg, extract=True))
            self.assertEqual(len(chunks), 2)
            self.assertEqual(FakeExtractor.calls, 3)
            self.assertEqual(chunks[0][1].shape, (2, 1536))
            train, test = _load_features(root/'pilot', protocol, cfg)
            self.assertEqual(train.shape, (2, 1536))
            self.assertEqual(test.shape, (1, 1536))
            self.assertEqual(FakeExtractor.calls, 3)
            np.testing.assert_allclose(train[:, 0], [0, 20])
            np.testing.assert_allclose(test[:, 0], [40])


if __name__ == '__main__':
    unittest.main()
