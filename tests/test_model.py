

# --- test_core_contract.py ---
import os

import pytest
import torch

from sinz_model.model import V1ConvNextCore


@pytest.mark.skipif(
    os.environ.get("RUN_HF_TESTS") != "1",
    reason="Requires HF model download/cache",
)
def test_core_shape():
    core = V1ConvNextCore(
        revision="2a62c34e37c1092d9a99cba587d5428837495d7c"
    )
    x = torch.randn(2, 1, 93, 93)
    y = core(x)
    assert y.shape == (2, 80, 11, 11)


# --- test_readout.py ---
import torch

from sinz_model.model import FullGaussian2d


def test_readout_shapes_and_sampling():
    readout = FullGaussian2d()
    assert tuple(readout._mu.shape) == (1, 458, 1, 2)
    assert tuple(readout.sigma.shape) == (1, 458, 1, 1)
    assert tuple(readout._features.shape) == (1, 80, 1, 458)
    assert tuple(readout.bias.shape) == (458,)

    x = torch.randn(4, 80, 11, 11)
    readout.train()
    y1 = readout(x)
    y2 = readout(x)
    assert y1.shape == (4, 458)
    assert not torch.allclose(y1, y2)

    readout.eval()
    with torch.no_grad():
        y3 = readout(x)
        y4 = readout(x)
    assert torch.allclose(y3, y4)


# --- test_model_contract.py ---
import os

import pytest
import torch
import yaml

from sinz_model.model import SinzV1Model


@pytest.mark.skipif(
    os.environ.get("RUN_HF_TESTS") != "1",
    reason="Requires HF model download/cache",
)
def test_end_to_end_positive():
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    model = SinzV1Model(cfg)
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(2, 1, 93, 93))
    assert y.shape == (2, 458)
    assert torch.all(y > 0)
