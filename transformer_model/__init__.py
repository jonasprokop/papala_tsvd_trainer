"""Optional pretrained transformer strategy; CPU benchmarking imports stay light."""
from importlib import import_module

__all__ = ['RidgeReadout','FittedRidge','fit_masked_ridge_cv',
           'TransformerNeuralModel','ZWMNeuralModel','load_zwm_bundle']


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    source = '.ridge' if name in __all__[:3] else '.model'
    return getattr(import_module(source,__name__),name)
