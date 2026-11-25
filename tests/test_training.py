

# --- test_losses.py ---
import torch

from sinz_model.analysis import masked_poisson_nll


def test_masked_poisson_ignores_missing_neurons():
    prediction = torch.tensor([[1.0, 100.0]])
    target = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, False]])
    loss = masked_poisson_nll(prediction, target, mask)
    expected = torch.tensor(1.0)
    assert torch.allclose(loss, expected, atol=1e-6)


# --- test_metrics.py ---
import numpy as np

from sinz_model.analysis import correlation_to_average


def test_corr_to_average():
    image_ids = np.array([1, 1, 2, 2, 3, 3])
    pred_by_image = {1: 1.0, 2: 2.0, 3: 3.0}
    prediction = np.array(
        [[pred_by_image[i]] for i in image_ids], dtype=float
    )
    response = np.array(
        [[1.0], [1.2], [2.0], [2.2], [3.0], [3.2]], dtype=float
    )
    mask = np.ones_like(response, dtype=bool)
    corr, payload = correlation_to_average(
        prediction, response, mask, image_ids
    )
    assert corr.shape == (1,)
    assert corr[0] > 0.999
    assert len(payload["image_ids"]) == 3


# --- test_orchestrator.py ---
from pathlib import Path

from sinz_model.orchestrator import RunRegistry
from sinz_model.cli import build_parser


def test_registry_init(tmp_path):
    db = tmp_path / "results.sqlite"
    reg = RunRegistry(str(db))
    reg.initialize()
    assert db.exists()
    assert reg.list_runs() == []


def test_cli_uses_config_argument():
    args = build_parser().parse_args(["status", "--config", "cfg.yaml", "run-id"])
    assert args.config == "cfg.yaml"


def test_progress_upsert_uses_progress_fraction(tmp_path):
    reg = RunRegistry(str(tmp_path / "results.sqlite"))
    reg.initialize()
    # Insert the minimum run row required by the progress foreign key.
    with reg.connect() as con:
        con.execute(
            "INSERT INTO runs(run_id, member_index, status, created_at_utc, config_sha256, config_json, run_dir) VALUES (?,?,?,?,?,?,?)",
            ("run", 0, "running", "now", "x", "{}", str(tmp_path)),
        )
    reg.update_progress("run", "readout_only", 0, 1, 2, 1, 0.5,
                        1.0, 2.0, 1e-3, 0.1, 0, 1.0, 2.0, "latest.pt")
    assert reg.get_progress("run")["progress_fraction"] == 0.5


# --- test_training_phase.py ---
import torch
from torch import nn

from sinz_model.trainer import _configure_model_for_phase


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        self.readout = nn.Linear(4, 2)

    def set_core_trainable(self, trainable):
        for p in self.core.parameters():
            p.requires_grad = bool(trainable)


CFG = {
    "training": {
        "phase1_freeze": {
            "freeze_core_parameters": True,
            "freeze_batchnorm_running_stats": True,
        }
    }
}


def test_phase1_full_core_freeze():
    model = DummyModel()
    _configure_model_for_phase(model, "readout_only", CFG)
    assert model.training is True
    assert model.core.training is False
    assert model.readout.training is True
    assert all(not p.requires_grad for p in model.core.parameters())
    assert all(p.requires_grad for p in model.readout.parameters())


def test_phase2_unfreezes_core_and_batchnorm():
    model = DummyModel()
    _configure_model_for_phase(model, "readout_only", CFG)
    _configure_model_for_phase(model, "joint_finetune", CFG)
    assert model.training is True
    assert model.core.training is True
    assert model.readout.training is True
    assert all(p.requires_grad for p in model.core.parameters())


# --- test_resume_phase_boundary.py ---
import torch
from torch import nn
from sinz_model.trainer import _optimizer_for_resume

class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Sequential(nn.Linear(4,4), nn.BatchNorm1d(4))
        self.readout = nn.Linear(4,2)
    def set_core_trainable(self, trainable):
        for p in self.core.parameters():
            p.requires_grad = bool(trainable)

CFG = {
    "training": {
        "readout_only_epochs": 20,
        "phase1_lr": 1e-3,
        "phase2_lr": 1e-4,
        "phase1_freeze": {"freeze_batchnorm_running_stats": True},
        "optimizer": {
            "name": "AdamW",
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "reset_at_phase2": False,
        },
    }
}

def test_resume_at_epoch20_builds_joint_optimizer():
    model = Dummy()
    opt = _optimizer_for_resume(model, CFG, next_epoch=20)
    # The checkpoint at next_epoch=20 was saved after epoch 19, so it still
    # has the phase-1 optimizer. The normal epoch-20 transition adds the core
    # parameter group afterwards.
    assert model.core.training is False
    assert all(not p.requires_grad for p in model.core.parameters())
    assert len(opt.param_groups) == 1

def test_resume_before_epoch20_is_readout_only():
    model = Dummy()
    opt = _optimizer_for_resume(model, CFG, next_epoch=19)
    assert model.core.training is False
    assert all(not p.requires_grad for p in model.core.parameters())
    assert len(opt.param_groups) == 1


def test_resume_after_boundary_builds_joint_optimizer():
    model = Dummy()
    opt = _optimizer_for_resume(model, CFG, next_epoch=21,
                                checkpoint_phase="joint_finetune")
    assert model.core.training is True
    assert all(p.requires_grad for p in model.core.parameters())
    assert len(opt.param_groups) == 2
    assert all(abs(g["lr"] - 1e-4) < 1e-12 for g in opt.param_groups)
