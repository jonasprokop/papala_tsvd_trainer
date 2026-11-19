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


class ZWMLayerBank(ZWMFeatures):
    """Capture several ZWM blocks during ONE upstream image forward pass.

    All hooks must belong to the same predictor and fire exactly once. The
    returned feature axis is ordered like ``layers``; no layer is silently
    omitted or evaluated using a second model invocation.
    """

    def __init__(self, model_ref: str, *, repo_root: str, layers, pooling: str = 'mean'):
        layers = tuple(layers)
        if not layers or len(set(layers)) != len(layers):
            raise ValueError('Expected a nonempty list of unique ZWM layers')
        super().__init__(model_ref, repo_root=repo_root, layer=layers[0], pooling=pooling)
        self.layers = layers
        self.hooks = [self.hook]
        try:
            for name in layers[1:]:
                owners = []
                for value in vars(self.predictor).values():
                    if isinstance(value, torch.nn.Module):
                        try:
                            value.get_submodule(name)
                        except AttributeError:
                            continue
                        owners.append(value)
                if len(owners) != 1:
                    raise RuntimeError(f'Cannot uniquely resolve ZWM layer {name!r}: {len(owners)} matches')
                self.hooks.append(FeatureHook(owners[0], name))
        except BaseException:
            self.close()
            raise

    @torch.inference_mode()
    def extract(self, image):
        for hook in self.hooks:
            hook.activation = None
            hook.count = 0
        self.predictor.single_image_forward(image, frame_gap=-1)
        arrays = [pool_tokens(hook.take(), self.pooling).detach().float().cpu().numpy()
                  for hook in self.hooks]
        if any(array.shape[0] != 1 for array in arrays):
            raise ValueError('Expected one feature row per ZWM layer')
        return arrays

    def close(self):
        for hook in getattr(self, 'hooks', [self.hook]):
            hook.close()
