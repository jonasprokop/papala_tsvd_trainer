

# --- test_checkpoint_atomic.py ---
import torch
from sinz_model.runtime import atomic_torch_save
def test_atomic(tmp_path):
    p=tmp_path/"latest.pt"; atomic_torch_save({"x":torch.tensor([1])},str(p))
    assert p.exists() and not (tmp_path/"latest.pt.tmp").exists()


# --- test_progress.py ---
from sinz_model.runtime import EpochTiming,format_seconds
def test_format_seconds(): assert format_seconds(3661)=="01:01:01"
def test_eta():
    t=EpochTiming(3); t.add(10); t.add(20); t.add(30); assert t.eta(4)==80
