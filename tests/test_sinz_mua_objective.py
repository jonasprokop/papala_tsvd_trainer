"""Opt-in signed normalized-MUA training must not change legacy Poisson runs."""
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml

from sinz_model.analysis import historical_poisson_nll, training_loss
from sinz_model.model import SinzV1Model
from sinz_model.trainer import _response_objective


ROOT = Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load((ROOT/'configs'/'tvsd_default.yaml').read_text())


class DummyReadout(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.full((kwargs['outdims'],), -2.0))

    def forward(self, features, sample=None):
        return self.bias.unsqueeze(0).expand(features.shape[0], -1)

    def feature_l1(self, average):
        return self.bias.new_zeros(())


class DummyCore(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.output_shape = (80, 4, 4)

    def forward(self, x):
        return torch.zeros((len(x), *self.output_shape), dtype=x.dtype, device=x.device)


def test_signed_output_is_opt_in_and_default_is_bitwise_legacy():
    cfg = config()
    cfg['model']['core']['output_hw'] = [4, 4]
    inp = torch.zeros((2, 3, 12, 12))
    with patch('sinz_model.model.V1ConvNextCore', DummyCore), \
         patch('sinz_model.model.FullGaussian2d', DummyReadout):
        old = SinzV1Model(cfg)
        y_default = old(inp)
        expected = torch.nn.functional.elu(torch.full_like(y_default, -3.0)) + 1.0
        assert torch.equal(y_default, expected)
        cfg['model']['output']['activation'] = 'identity'
        signed = SinzV1Model(cfg)
        assert torch.equal(signed(inp), torch.full_like(y_default, -2.0))


def test_mse_mask_scaling_and_gradients():
    class Model:
        readout = DummyReadout(outdims=2)

    pred = torch.tensor([[0.0, 1.0], [-1.0, 2.0]], requires_grad=True)
    target = torch.tensor([[-2.0, 50.0], [1.0, -3.0]])
    mask = torch.tensor([[True, False], [True, True]])
    result = training_loss(Model(), pred, target, mask, gamma_readout=0.0,
                           dataset_size=10, batch_size=2, objective='mse')
    expected = math.sqrt(5) * (4 + 4 + 25)
    assert float(result['mse']) == pytest.approx(expected)
    assert 'poisson' not in result
    result['loss'].backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[0,1].item() == 0


def test_old_poisson_objective_and_keys_unchanged():
    class Model:
        readout = DummyReadout(outdims=2)

    pred = torch.tensor([[1.5, 2.5]])
    target = torch.tensor([[1.0, 3.0]])
    mask = torch.tensor([[True, True]])
    implicit = training_loss(Model(), pred, target, mask, gamma_readout=0.0,
                             dataset_size=100, batch_size=1)
    explicit = training_loss(Model(), pred, target, mask, gamma_readout=0.0,
                             dataset_size=100, batch_size=1, objective='poisson')
    expected = historical_poisson_nll(pred, target, mask, 100, 1)
    assert torch.equal(implicit['loss'], expected)
    assert torch.equal(implicit['poisson'], explicit['poisson'])


def test_normalized_mua_refuses_poisson_before_run_creation():
    cfg = config()
    assert _response_objective(cfg, {'version':'tvsd_tracked_neurons_masked_v3'}) == 'poisson'
    with pytest.raises(ValueError, match='normalized MUA'):
        _response_objective(cfg, {'signal_type': 'normalized_MUA'})
    cfg['training']['loss']['kind'] = 'mse'
    with pytest.raises(ValueError, match=r'MSE \+ identity'):
        _response_objective(cfg, {'signal_type': 'normalized_MUA'})
    cfg['model']['output']['activation'] = 'identity'
    assert _response_objective(cfg, {'signal_type':'normalized_MUA'}) == 'mse'


def test_mua_config_does_not_share_optuna_registry():
    ours = yaml.safe_load((ROOT/'configs'/'tvsd_sinz_mua_v1.yaml').read_text())
    existing = config()
    assert ours['training']['loss']['kind'] == 'mse'
    assert ours['model']['output']['activation'] == 'identity'
    assert ours['model']['readout']['outdims_by_animal'] == {'F':512,'N':512}
    assert ours['project']['sqlite_path'] != existing['project']['sqlite_path']
    assert existing['training']['loss'].get('kind','poisson') == 'poisson'
    assert existing['model']['output'].get('activation','elu') == 'elu'
