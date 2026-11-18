"""Read-only activation capture; pooling conventions are explicit experiment choices."""
from __future__ import annotations
import torch


def pool_tokens(t, mode='mean'):
    if isinstance(t, (list,tuple)):
        t = t[0]
    if not isinstance(t, torch.Tensor):
        raise TypeError(f'Expected tensor activation, got {type(t)}')
    if t.ndim == 2:
        return t
    if mode == 'flatten':
        return t.flatten(1)
    if t.ndim == 3:
        if mode == 'mean':
            return t.mean(dim=1)
        if mode == 'cls':
            return t[:,0]
    if t.ndim == 4 and mode == 'mean':
        return t.mean(dim=(-2,-1))
    raise ValueError(f'Unsupported activation {tuple(t.shape)} and pooling {mode}')


class FeatureHook:
    def __init__(self, module:torch.nn.Module, layer:str):
        self.layer=layer
        self.activation=None
        self.count=0
        self.handle=module.get_submodule(layer).register_forward_hook(self._capture)
    def _capture(self, module, args, output):
        self.count += 1
        self.activation=output[0] if isinstance(output,(tuple,list)) else output
    def take(self):
        if self.activation is None:
            raise RuntimeError(f'Layer {self.layer} did not execute')
        if self.count != 1:
            raise RuntimeError(f'Layer {self.layer} executed {self.count} times during one extraction; ambiguous activation')
        value=self.activation
        self.activation=None
        self.count=0
        return value
    def close(self):
        self.handle.remove()
