from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (
    CombinedNaturalImageDataset,
    ImageStore,
    build_combined_response_table,
)
from .analysis import (
    correlation_to_average,
    per_neuron_masked_pearson,
    summarize_correlations,
)
from .model import load_best_model
from .data import PreprocessConfig
from .runtime import seed_worker, canonical_json


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    prediction_parts = []
    response_parts = []
    mask_parts = []
    image_id_parts = []
    repeat_parts = []

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        prediction = model(image, sample=False).cpu().numpy()

        prediction_parts.append(prediction)
        response_parts.append(batch["response"].numpy())
        mask_parts.append(batch["mask"].numpy())
        image_id_parts.append(batch["image_id"].numpy())
        repeat_parts.append(batch["repeat_index"].numpy())

    return {
        "prediction": np.concatenate(prediction_parts, axis=0),
        "response": np.concatenate(response_parts, axis=0),
        "mask": np.concatenate(mask_parts, axis=0),
        "image_id": np.concatenate(image_id_parts, axis=0),
        "repeat_index": np.concatenate(repeat_parts, axis=0),
    }


def compute_natural_image_metrics(collected):
    single = per_neuron_masked_pearson(
        collected["prediction"],
        collected["response"],
        collected["mask"],
    )
    corr_avg, avg_payload = correlation_to_average(
        collected["prediction"],
        collected["response"],
        collected["mask"],
        collected["image_id"],
    )
    return {
        "single_trial_corr": single,
        "corr_to_average": corr_avg,
        "single_trial_summary": summarize_correlations(single),
        "corr_to_average_summary": summarize_correlations(corr_avg),
        "averaged_payload": avg_payload,
    }


def save_evaluation(
    output_dir: str,
    run_id: str,
    metrics,
    collected=None,
    save_predictions=False,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_id": run_id,
        "single_trial_corr": metrics["single_trial_summary"],
        "corr_to_average": metrics["corr_to_average_summary"],
        "n_unique_test_images": int(
            len(metrics["averaged_payload"]["image_ids"])
        ),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    neuron_ids = np.arange(len(metrics["corr_to_average"]))
    with open(out / "per_neuron.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["neuron_id", "corr_to_average", "single_trial_corr"]
        )
        for n in neuron_ids:
            writer.writerow(
                [
                    int(n),
                    float(metrics["corr_to_average"][n]),
                    float(metrics["single_trial_corr"][n]),
                ]
            )

    if save_predictions and collected is not None:
        np.savez_compressed(
            out / "predictions.npz",
            **collected
        )

    return summary


def build_test_loader(cfg):
    data_cfg = cfg["data"]
    if data_cfg.get("kind") == "tvsd":
        from .tvsd_data import TVSDTrainingBundle
        from .runtime import seed_worker
        bundle = TVSDTrainingBundle(cfg)
        loader_cfg = data_cfg["loader"]
        workers = int(loader_cfg["num_workers"])
        loader = DataLoader(bundle.dataset("test"),
                            batch_size=int(loader_cfg["batch_size"]),
                            shuffle=False, num_workers=workers,
                            pin_memory=bool(loader_cfg["pin_memory"]),
                            worker_init_fn=seed_worker,
                            persistent_workers=workers>0)
        return bundle.test, loader
    response_cfg = data_cfg["response"]
    preprocess_cfg = PreprocessConfig.from_dict(
        data_cfg["preprocessing"]
    )
    table = build_combined_response_table(
        data_root=data_cfg["root"],
        split="test",
        time_bins=response_cfg["time_bins"],
        average_bins=response_cfg["average_bins"],
        expected_total_neurons=response_cfg["expected_total_neurons"],
    )
    store = ImageStore(
        images_root=str(Path(data_cfg["root"]) / data_cfg["images_subdir"]),
        filename_pattern=data_cfg["image_filename_pattern"],
        image_id_offset=data_cfg["image_id_offset"],
    )
    preprocess_cfg, _ = store.prepare(
        preprocess_cfg, data_cfg["preprocessing"].get("normalization", {})
    )
    dataset = CombinedNaturalImageDataset(table, store, preprocess_cfg)

    loader_cfg = data_cfg["loader"]
    loader = DataLoader(
        dataset,
        batch_size=int(loader_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(loader_cfg["num_workers"]),
        pin_memory=bool(loader_cfg["pin_memory"]),
        worker_init_fn=seed_worker,
        persistent_workers=(
            int(loader_cfg["num_workers"]) > 0
        ),
    )
    return table, loader


def evaluate_checkpoint(cfg, checkpoint_path, device):
    model, saved_cfg = load_best_model(checkpoint_path, device)
    if canonical_json(cfg["data"]) != canonical_json(saved_cfg["data"]):
        raise ValueError("Evaluation data config differs from checkpoint")

    table, loader = build_test_loader(cfg)
    collected = collect_predictions(model, loader, device)
    metrics = compute_natural_image_metrics(collected)
    return metrics, collected


def evaluate_ensemble(cfg, checkpoint_paths, device):
    if not checkpoint_paths:
        raise ValueError("Ensemble requires at least one checkpoint")
    first, saved_cfg = load_best_model(checkpoint_paths[0], device)
    if canonical_json(cfg["data"]) != canonical_json(saved_cfg["data"]):
        raise ValueError("Evaluation data config differs from checkpoint")
    table, loader = build_test_loader(saved_cfg)

    all_predictions = []
    reference = None
    for i, path in enumerate(checkpoint_paths):
        model, member_cfg = (first, saved_cfg) if i == 0 else load_best_model(path, device)
        if canonical_json(member_cfg["data"]) != canonical_json(saved_cfg["data"]):
            raise ValueError("Ensemble checkpoints have different data preprocessing")
        collected = collect_predictions(model, loader, device)
        if reference is None:
            reference = collected
        else:
            if not np.array_equal(
                reference["image_id"],
                collected["image_id"],
            ) or not np.array_equal(reference["response"], collected["response"]) or not np.array_equal(reference["mask"], collected["mask"]):
                raise RuntimeError("Ensemble member test order mismatch")
        all_predictions.append(collected["prediction"])

    ensemble_prediction = np.mean(
        np.stack(all_predictions, axis=0),
        axis=0,
    )
    ensemble = dict(reference)
    ensemble["prediction"] = ensemble_prediction
    metrics = compute_natural_image_metrics(ensemble)
    return metrics, ensemble
