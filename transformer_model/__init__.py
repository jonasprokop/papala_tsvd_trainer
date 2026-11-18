"""Optional pretrained transformer strategy; no import-time upstream dependency."""
from .ridge import RidgeReadout, FittedRidge, fit_masked_ridge_cv
from .model import TransformerNeuralModel, ZWMNeuralModel, load_zwm_bundle
