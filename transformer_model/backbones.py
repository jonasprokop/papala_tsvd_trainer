"""ZWM upstream inference adapter (no ZWM source copied into this project)."""
from __future__ import annotations
from pathlib import Path
import torch
from .features import FeatureHook, pool_tokens


class ZWMFeatures:
    """Capture representations while ZWM's own single-image inference runs.

    The upstream model (checkpoint, predictor, transforms) is provided by the
    separately installed ZWM package. We deliberately do not call model(image).
    """
    def __init__(self, model_ref: str, *, repo_root: str, layer: str,
                 pooling: str = 'mean'):
        import zwm.zwm_predictor as upstream
        ZWMPredictor = upstream.ZWMPredictor
        self.repo_root = Path(repo_root).resolve()
        if not (self.repo_root / 'out' / model_ref).is_file():
            raise FileNotFoundError(f'ZWM checkpoint missing: {self.repo_root / "out" / model_ref}')
        # samefile handles the two Singularity bind aliases /mnt/tvsd and
        # /storage/... for one physical checkout.
        module_file=Path(upstream.__file__).resolve()
        if not any(module_file.samefile(path) for path in
                   self.repo_root.rglob('zwm_predictor.py') if path.is_file()):
            raise RuntimeError('Loaded ZWM package is not from model.backbone.repo_root')
        # Upstream ModelFactory finds its own repository; require a matching tree.
        # Ensure that repo_root points to the installed upstream source checkout.
        self.predictor = ZWMPredictor(model_ref)
        self.model_ref = model_ref
        self.layer = layer
        self.pooling = pooling
        # Avoid assumptions about whether upstream calls the torch module
        # predictor.model, predictor.zwm_model, etc. Resolve by layer topology.
        owners = []
        for value in vars(self.predictor).values():
            if isinstance(value, torch.nn.Module):
                try:
                    value.get_submodule(layer)
                except AttributeError:
                    continue
                owners.append(value)
        if len(owners) != 1:
            raise RuntimeError(f'Cannot uniquely resolve ZWM layer {layer!r}: {len(owners)} matches')
        self.hook = FeatureHook(owners[0], layer)

    @torch.inference_mode()
    def extract(self, image):
        self.hook.activation = None
        self.hook.count = 0
        self.predictor.single_image_forward(image, frame_gap=-1)
        return pool_tokens(self.hook.take(), self.pooling).detach().float().cpu().numpy()

    def close(self):
        self.hook.close()
