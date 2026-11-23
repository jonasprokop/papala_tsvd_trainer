"""The non-finite training diagnostic is serializable and preserves masking."""
import json
import tempfile
import unittest

import torch

from sinz_model.trainer import _nonfinite_loss_report


class NonfiniteDiagnosticTest(unittest.TestCase):
    def test_report_records_sources_without_raw_responses(self):
        model = torch.nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(float("inf"))
        x = torch.tensor([[[[0.0, 1.0]]]])
        y = torch.tensor([[2.0, float("nan")]])
        mask = torch.tensor([[True, False]])
        pred = torch.tensor([[float("inf"), 1.0]])
        terms = {
            "loss": torch.tensor(float("nan")),
            "poisson": torch.tensor(float("nan")),
            "feature_l1": torch.tensor(1.0),
            "regularizer": torch.tensor(4.0),
        }

        with tempfile.TemporaryDirectory() as folder:
            path = _nonfinite_loss_report(
                folder, model, x, y, mask, pred, terms,
                0, 0, "readout_only", 4.0)
            report = json.loads(path.read_text())

        self.assertEqual(path.name, "nonfinite_loss_e000_b00000.json")
        self.assertEqual(report["observed_targets"], 1)
        self.assertEqual(report["target_observed"]["nan"], 0)
        self.assertEqual(report["target_unobserved"]["nan"], 1)
        self.assertEqual(report["prediction_observed"]["posinf"], 1)
        self.assertEqual(report["loss_terms"]["poisson"]["nan"], 1)
        self.assertEqual(report["loss_terms"]["regularizer"]["finite_max"], 4.0)
        self.assertEqual(report["nonfinite_parameters"]["weight"], 1)
