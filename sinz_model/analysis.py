from __future__ import annotations

from typing import Dict

import torch


def masked_poisson_nll(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """
    Poisson NLL without the target-only log-factorial constant.

    For each sample, average over valid/observed neurons, then average samples.
    This makes session-coverage differences explicit and prevents missing
    neurons (stored as zeros) from contributing to the objective.

    The exact registered trainer reduction was not serialized; this reduction
    is therefore an explicit replication choice.
    """
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("prediction, target and mask must have identical shapes")

    term = prediction - target * torch.log(prediction + float(eps))
    weight = mask.to(dtype=term.dtype)
    per_sample_den = weight.sum(dim=1).clamp_min(1.0)
    per_sample = (term * weight).sum(dim=1) / per_sample_den
    return per_sample.mean()


def historical_poisson_nll(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    dataset_size: int,
    batch_size: int,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Poisson term used by the recovered nnvision trainer.

    The historical trainer sums the observed neuron terms over the batch and
    applies ``sqrt(N / B)``.  This is deliberately separate from
    ``masked_poisson_nll`` because the latter remains a useful normalized
    diagnostic metric.
    """
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("prediction, target and mask must have identical shapes")
    term = prediction - target * torch.log(prediction + float(eps))
    scale = (float(dataset_size) / float(batch_size)) ** 0.5
    return scale * (term * mask.to(dtype=term.dtype)).sum()


def training_loss(
    model,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    gamma_readout: float,
    dataset_size: int | None = None,
    batch_size: int | None = None,
    eps: float = 1.0e-8,
) -> Dict[str, torch.Tensor]:
    if dataset_size is None or batch_size is None:
        poisson = masked_poisson_nll(prediction, target, mask, eps=eps)
        feature_l1 = model.readout.feature_l1(average=True)
    else:
        poisson = historical_poisson_nll(
            prediction, target, mask, dataset_size, batch_size, eps=eps
        )
        feature_l1 = model.readout.feature_l1(average=False)
    regularizer = float(gamma_readout) * feature_l1
    total = poisson + regularizer
    return {
        "loss": total,
        "poisson": poisson,
        "feature_l1": feature_l1,
        "regularizer": regularizer,
    }


# ---------------------------------------------------------------------------

from typing import Dict, Tuple

import numpy as np


def _pearson_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def per_neuron_masked_pearson(
    prediction: np.ndarray,
    response: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    prediction = np.asarray(prediction)
    response = np.asarray(response)
    mask = np.asarray(mask, dtype=bool)
    if prediction.shape != response.shape or response.shape != mask.shape:
        raise ValueError("prediction, response and mask must have same shape")

    n_neurons = prediction.shape[1]
    out = np.full(n_neurons, np.nan, dtype=np.float64)
    for n in range(n_neurons):
        valid = mask[:, n]
        out[n] = _pearson_1d(
            prediction[valid, n],
            response[valid, n],
        )
    return out


def correlation_to_average(
    prediction: np.ndarray,
    response: np.ndarray,
    mask: np.ndarray,
    image_ids: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Primary paper-comparable metric.

    Group test rows by image ID. For each image/neuron:
      - average observed responses across repeats where that neuron was present;
      - average deterministic model predictions across the same rows.

    Then compute Pearson correlation across unique images for every neuron.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    image_ids = np.asarray(image_ids, dtype=np.int64)

    unique_ids = np.unique(image_ids)
    n_neurons = prediction.shape[1]

    pred_avg = np.full((len(unique_ids), n_neurons), np.nan, dtype=np.float64)
    resp_avg = np.full_like(pred_avg, np.nan)
    avg_mask = np.zeros_like(pred_avg, dtype=bool)

    for i, image_id in enumerate(unique_ids):
        rows = image_ids == image_id
        for n in range(n_neurons):
            valid = rows & mask[:, n]
            if valid.any():
                pred_avg[i, n] = prediction[valid, n].mean()
                resp_avg[i, n] = response[valid, n].mean()
                avg_mask[i, n] = True

    corr = per_neuron_masked_pearson(
        prediction=pred_avg,
        response=resp_avg,
        mask=avg_mask,
    )
    return corr, {
        "image_ids": unique_ids,
        "prediction_average": pred_avg,
        "response_average": resp_avg,
        "mask_average": avg_mask,
    }


def summarize_correlations(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "n_valid": 0,
        }
    # The historical evaluator replaces undefined neuron correlations by 0
    # before computing the population mean.
    scored = np.nan_to_num(values, nan=0.0)
    return {
        "mean": float(scored.mean()),
        "median": float(np.median(scored)),
        "n_valid": int(finite.size),
    }
