from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Sequence, Tuple
import re

import numpy as np
import torch


@dataclass(frozen=True)
class PreprocessConfig:
    crop: Tuple[Tuple[int, int], Tuple[int, int]]
    subsample: int
    scale: float
    expected_output_hw: Tuple[int, int]
    mean: float
    std: float

    @classmethod
    def from_dict(cls, cfg):
        crop = cfg["crop"]
        return cls(
            crop=(
                (int(crop[0][0]), int(crop[0][1])),
                (int(crop[1][0]), int(crop[1][1])),
            ),
            subsample=int(cfg["subsample"]),
            scale=float(cfg["scale"]),
            expected_output_hw=(
                int(cfg["expected_output_hw"][0]),
                int(cfg["expected_output_hw"][1]),
            ),
            mean=None if cfg.get("mean") is None else float(cfg["mean"]),
            std=None if cfg.get("std") is None else float(cfg["std"]),
        )


def _rescale_historical(image: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1:
        return image

    from skimage.transform import rescale

    dtype = image.dtype
    try:
        # Historical scikit-image 0.19.x signature.
        out = rescale(
            image,
            scale,
            mode="reflect",
            multichannel=False,
            anti_aliasing=False,
            preserve_range=True,
        )
    except TypeError:
        # Compatibility only for running tests in newer skimage.
        out = rescale(
            image,
            scale,
            mode="reflect",
            channel_axis=None,
            anti_aliasing=False,
            preserve_range=True,
        )
    return out.astype(dtype)


def transform_image_geometry_numpy(
    image: np.ndarray,
    cfg: PreprocessConfig,
) -> np.ndarray:
    """Apply the recovered historical geometry, but not z-scoring."""
    if image.ndim != 2:
        raise ValueError("Historical V1 path expects a 2D grayscale image")

    top, bottom = cfg.crop[0]
    left, right = cfg.crop[1]
    h, w = image.shape

    h_stop = h - bottom if bottom else h
    w_stop = w - right if right else w

    x = image[
        top:h_stop:cfg.subsample,
        left:w_stop:cfg.subsample,
    ]
    x = _rescale_historical(x, cfg.scale)

    if tuple(x.shape) != cfg.expected_output_hw:
        raise ValueError(
            "Transformed image has shape %r; expected %r. "
            "Refusing implicit resize/interpolation."
            % (tuple(x.shape), cfg.expected_output_hw)
        )

    # Historical ImageCache converted transformed arrays to torch.float before
    # computing global normalization statistics. Keep the same float32 domain.
    return x[None, ...].astype(np.float32, copy=False)


def transform_image_numpy(
    image: np.ndarray,
    cfg: PreprocessConfig,
) -> np.ndarray:
    x = transform_image_geometry_numpy(image, cfg)
    if cfg.mean is None or cfg.std is None:
        raise ValueError(
            "PreprocessConfig normalization is unresolved; prepare the image "
            "store first or provide fixed mean/std"
        )
    if not np.isfinite(cfg.std) or cfg.std <= 0:
        raise ValueError("Normalization std must be finite and positive")
    x = (x - float(cfg.mean)) / float(cfg.std)
    return x.astype(np.float32, copy=False)


def preprocess_image(
    image: np.ndarray,
    cfg: PreprocessConfig,
) -> torch.Tensor:
    return torch.from_numpy(transform_image_numpy(image, cfg)).float()


def audit_data_alignment(
    data_root: str,
    images_subdir: str,
    image_filename_pattern: str,
    preprocess_cfg: PreprocessConfig,
    image_id_offset: int = 0,
    offsets: Sequence[int] = (-1, 0, 1),
    max_images: Optional[int] = None,
) -> Dict[str, object]:
    """Report response/image-ID coverage and transformed crop statistics."""
    root = Path(data_root)
    image_root = root / images_subdir
    store = ImageStore(str(image_root), image_filename_pattern, image_id_offset)
    file_ids = store.list_file_ids()
    file_id_set = set(file_ids)

    sessions = discover_sessions(data_root)
    response_ids = set()
    train_ids = set()
    test_ids = set()
    for session in sessions:
        train_frame = pd.read_csv(session.train_csv, usecols=["image_id"])
        test_frame = pd.read_csv(session.test_csv, usecols=["image_id"])
        session_train = {int(x) for x in train_frame["image_id"].tolist()}
        session_test = {int(x) for x in test_frame["image_id"].tolist()}
        train_ids.update(session_train)
        test_ids.update(session_test)
        response_ids.update(session_train)
        response_ids.update(session_test)

    coverage = {}
    for offset in offsets:
        matched = [int(image_id) + int(offset) in file_id_set for image_id in response_ids]
        coverage[str(int(offset))] = {
            "offset": int(offset),
            "response_ids": len(response_ids),
            "matched": int(sum(matched)),
            "missing": int(len(matched) - sum(matched)),
            "coverage": float(np.mean(matched)) if matched else 0.0,
        }

    stats_ids = sorted(response_ids)
    if max_images is not None:
        stats_ids = stats_ids[: int(max_images)]
    count = 0
    total = 0.0
    total_sq = 0.0
    raw_min = None
    raw_max = None
    missing_stats = []
    for image_id in stats_ids:
        path = store.path_for_id(image_id)
        if not path.exists():
            missing_stats.append(int(image_id))
            continue
        x = transform_image_geometry_numpy(store.load(image_id), preprocess_cfg).astype(np.float64)
        count += int(x.size)
        total += float(x.sum())
        total_sq += float(np.square(x).sum())
        xmin, xmax = float(x.min()), float(x.max())
        raw_min = xmin if raw_min is None else min(raw_min, xmin)
        raw_max = xmax if raw_max is None else max(raw_max, xmax)
    if count:
        mean = total / count
        var = max(total_sq / count - mean * mean, 0.0)
        raw_stats = {
            "n_images": int(count // np.prod(preprocess_cfg.expected_output_hw)),
            "mean": float(mean), "std": float(np.sqrt(var)),
            "min": raw_min, "max": raw_max,
        }
    else:
        raw_stats = {"n_images": 0, "mean": None, "std": None, "min": None, "max": None}

    return {
        "data_root": str(root), "images_root": str(image_root),
        "n_sessions": len(sessions),
        "total_neurons": int(sessions[-1].global_end) if sessions else 0,
        "n_image_files": len(file_id_set),
        "image_file_id_min": min(file_id_set) if file_id_set else None,
        "image_file_id_max": max(file_id_set) if file_id_set else None,
        "n_response_ids": len(response_ids),
        "n_train_unique_ids": len(train_ids),
        "n_test_unique_ids": len(test_ids),
        "train_test_overlap": len(train_ids & test_ids),
        "offset_coverage": coverage,
        "configured_image_id_offset": int(image_id_offset),
        "raw_cropped_statistics": raw_stats,
        "statistics_missing_ids": missing_stats,
        "resolved_mean": None if preprocess_cfg.mean is None else float(preprocess_cfg.mean),
        "resolved_std": None if preprocess_cfg.std is None else float(preprocess_cfg.std),
    }


# ---------------------------------------------------------------------------

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


@dataclass(frozen=True)
class SplitArtifact:
    train_image_ids: np.ndarray
    val_image_ids: np.ndarray
    seed: int
    train_fraction: float
    algorithm: str
    sha256: str


def _payload(train_ids, val_ids, seed, train_fraction, algorithm):
    return {
        "train_image_ids": [int(x) for x in train_ids],
        "val_image_ids": [int(x) for x in val_ids],
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "algorithm": str(algorithm),
    }


def _hash_payload(payload) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def create_split(
    image_ids: Iterable[int],
    train_fraction: float,
    seed: int,
) -> SplitArtifact:
    ids = np.asarray(sorted(set(int(x) for x in image_ids)), dtype=np.int64)
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0,1)")

    rng = np.random.RandomState(int(seed))
    permutation = rng.permutation(ids)
    n_train = int(len(permutation) * float(train_fraction))

    train_ids = permutation[:n_train]
    val_ids = permutation[n_train:]
    algorithm = "numpy.RandomState.permutation"

    payload = _payload(
        train_ids, val_ids, seed, train_fraction, algorithm
    )
    sha = _hash_payload(payload)
    return SplitArtifact(
        train_image_ids=train_ids,
        val_image_ids=val_ids,
        seed=int(seed),
        train_fraction=float(train_fraction),
        algorithm=algorithm,
        sha256=sha,
    )


def save_split(split: SplitArtifact, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload(
        split.train_image_ids,
        split.val_image_ids,
        split.seed,
        split.train_fraction,
        split.algorithm,
    )
    payload["sha256"] = split.sha256
    with open(path_obj, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_split(path: str) -> SplitArtifact:
    with open(path, "r") as f:
        payload = json.load(f)
    expected = payload.pop("sha256")
    observed = _hash_payload(payload)
    if expected != observed:
        raise ValueError("Split artifact SHA mismatch")

    return SplitArtifact(
        train_image_ids=np.asarray(payload["train_image_ids"], dtype=np.int64),
        val_image_ids=np.asarray(payload["val_image_ids"], dtype=np.int64),
        seed=int(payload["seed"]),
        train_fraction=float(payload["train_fraction"]),
        algorithm=str(payload["algorithm"]),
        sha256=expected,
    )


# ---------------------------------------------------------------------------

import ast
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset



@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    num_units: int
    global_start: int
    global_end: int
    train_csv: str
    test_csv: str


@dataclass
class CombinedResponseTable:
    responses: np.ndarray
    mask: np.ndarray
    image_ids: np.ndarray
    repeat_index: np.ndarray
    session_info: List[SessionInfo]

    def subset_by_image_ids(self, ids: Iterable[int]) -> "CombinedResponseTable":
        allowed = set(int(x) for x in ids)
        keep = np.asarray([int(x) in allowed for x in self.image_ids])
        return CombinedResponseTable(
            responses=self.responses[keep],
            mask=self.mask[keep],
            image_ids=self.image_ids[keep],
            repeat_index=self.repeat_index[keep],
            session_info=self.session_info,
        )


def _json_or_literal(value):
    if isinstance(value, (list, tuple)):
        return value
    if not isinstance(value, str):
        return value
    value = value.strip()
    try:
        return json.loads(value)
    except Exception:
        try:
            return ast.literal_eval(value)
        except Exception:
            # Some pandas CSV exports contain NumPy-style arrays without
            # commas, e.g. "[[0. 1. ...]]". Fall back to a flat numeric parse;
            # parse_response_cell reshapes it using num_units below.
            flat = np.fromstring(
                value.replace("[", " ").replace("]", " ").replace(",", " "),
                sep=" ",
                dtype=np.float32,
            )
            return flat


def parse_response_cell(
    value,
    num_units: int,
    time_bins: Sequence[int],
    average_bins: bool = False,
) -> np.ndarray:
    arr = np.asarray(_json_or_literal(value), dtype=np.float32)

    if arr.ndim == 1:
        if arr.size == num_units:
            return arr.astype(np.float32, copy=False)
        if arr.size % num_units == 0:
            arr = arr.reshape(num_units, arr.size // num_units)
        else:
            raise ValueError(
                "1D response has %d values, incompatible with num_units=%d"
                % (arr.size, num_units)
            )

    if arr.ndim != 2:
        raise ValueError("Response cell must parse to 1D or 2D array")

    if arr.shape[0] == num_units:
        units_by_time = arr
    elif arr.shape[1] == num_units:
        units_by_time = arr.T
    else:
        raise ValueError(
            "Cannot align response shape %r with num_units=%d"
            % (arr.shape, num_units)
        )

    bins = np.asarray(list(time_bins), dtype=np.int64)
    if bins.ndim != 1 or bins.size == 0:
        raise ValueError("time_bins must contain at least one bin index")
    if np.any(bins < 0) or np.any(bins >= units_by_time.shape[1]):
        raise IndexError(
            "Requested time bin outside response array: bins=%s, available=%d"
            % (bins.tolist(), units_by_time.shape[1])
        )
    selected = units_by_time[:, bins]
    if average_bins:
        out = selected.mean(axis=1)
    else:
        out = selected.sum(axis=1)
    return out.astype(np.float32, copy=False)


def _read_num_units(meta_path: Path) -> int:
    with open(meta_path, "r") as f:
        meta = json.load(f)

    if isinstance(meta, dict):
        for key in ("num_units", "n_units", "num_neurons"):
            if key in meta:
                return int(meta[key])
        if "unit_ids" in meta:
            return int(len(meta["unit_ids"]))
    raise KeyError("Could not determine num_units from %s" % meta_path)


# Canonical global neuron order of the released macaque V1 twin.
#
# Source: enigma-brain/dualneuron, dualneuron/data/recordings.py,
# SESSION_ORDER["v1"], commit a9ea5f99637ab278d708c563d29b1c9accb5cf67.
# Documented there as the verbatim ``neuronal_data_files`` order of the
# nnvision ``CSRF19_V1`` dataset config used to train the ConvNeXt twin.
# It is deliberately NOT sorted by session id. Independently verified here:
# per-neuron test correlations of a retrained model vs the released
# correlations.npy go from Pearson -0.03 (numeric sort) to 0.94 (this order).
CANONICAL_V1_SESSION_ORDER: Tuple[str, ...] = tuple(str(x) for x in (
    3631896544452, 3632669014376, 3632932714885, 3633364677437, 3634055946316,
    3634142311627, 3634658447291, 3634744023164, 3635178040531, 3635949043110,
    3636034866307, 3636552742293, 3637161140869, 3637248451650, 3637333931598,
    3637760318484, 3637851724731, 3638367026975, 3638456653849, 3638885582960,
    3638373332053, 3638541006102, 3638802601378, 3638973674012, 3639060843972,
    3639406161189, 3640011636703, 3639664527524, 3639492658943, 3639749909659,
    3640095265572, 3631807112901,
))
CANONICAL_V1_TOTAL_NEURONS = 458


def session_order_sha256(session_ids: Sequence[str]) -> str:
    """Provenance hash of a session order (comma-joined ids)."""
    return hashlib.sha256(",".join(str(s) for s in session_ids).encode("utf-8")).hexdigest()


CANONICAL_V1_SESSION_ORDER_SHA256 = session_order_sha256(CANONICAL_V1_SESSION_ORDER)


def _resolve_session_order(train_ids, test_ids) -> List[str]:
    """Return the session order, enforcing the canonical V1 order on real data.

    Any overlap with the canonical list means this is the released V1 dataset:
    then the discovered train and test session sets must both equal the
    canonical set exactly (hard failure on missing/extra sessions), and the
    canonical order is used. With no overlap at all (synthetic test data) the
    legacy numeric sort is kept and a warning is emitted.
    """
    canonical = set(CANONICAL_V1_SESSION_ORDER)
    discovered = set(train_ids) | set(test_ids)
    if discovered & canonical:
        problems = []
        for label, ids in (("train", set(train_ids)), ("test", set(test_ids))):
            missing = sorted(canonical - ids)
            extra = sorted(ids - canonical)
            if missing:
                problems.append("%s/ missing sessions: %s" % (label, missing))
            if extra:
                problems.append("%s/ has sessions not in canonical order: %s" % (label, extra))
        if problems:
            raise ValueError(
                "Session set does not match the canonical V1 order "
                "(dualneuron SESSION_ORDER['v1']):\n  " + "\n  ".join(problems)
            )
        return list(CANONICAL_V1_SESSION_ORDER)

    warnings.warn(
        "No discovered session matches the canonical V1 session list; falling "
        "back to numeric session-id order. Neuron indices will NOT match the "
        "released model. This is only expected for synthetic test data.",
        RuntimeWarning,
    )
    return sorted(
        set(train_ids) & set(test_ids),
        key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
    )


def discover_sessions(data_root: str) -> List[SessionInfo]:
    root = Path(data_root)
    train_root = root / "train"
    test_root = root / "test"

    train_ids = {p.name for p in train_root.iterdir() if p.is_dir()}
    test_ids = {p.name for p in test_root.iterdir() if p.is_dir()}
    session_ids = _resolve_session_order(train_ids, test_ids)
    if not session_ids:
        raise FileNotFoundError("No common train/test session directories found")
    canonical_mode = tuple(session_ids) == CANONICAL_V1_SESSION_ORDER

    sessions = []
    cursor = 0
    for sid in session_ids:
        train_dir = train_root / sid
        test_dir = test_root / sid
        meta_path = train_dir / "meta_data.json"
        if not meta_path.exists():
            meta_path = test_dir / "meta_data.json"
        num_units = _read_num_units(meta_path)
        sessions.append(
            SessionInfo(
                session_id=sid,
                num_units=num_units,
                global_start=cursor,
                global_end=cursor + num_units,
                train_csv=str(train_dir / "responses.csv"),
                test_csv=str(test_dir / "responses.csv"),
            )
        )
        cursor += num_units
    if canonical_mode and cursor != CANONICAL_V1_TOTAL_NEURONS:
        raise ValueError(
            "Canonical V1 sessions sum to %d units; expected %d"
            % (cursor, CANONICAL_V1_TOTAL_NEURONS)
        )
    return sessions


def session_mapping_dict(session_info: Sequence[SessionInfo]) -> List[Dict[str, object]]:
    order_ids = [s.session_id for s in session_info]
    order_sha = session_order_sha256(order_ids)
    is_canonical = tuple(order_ids) == CANONICAL_V1_SESSION_ORDER
    return [
        {
            "order_index": i,
            "session_id": s.session_id,
            "num_units": s.num_units,
            "global_start": s.global_start,
            "global_end": s.global_end,
            "session_order_sha256": order_sha,
            "canonical_v1_order": is_canonical,
        }
        for i, s in enumerate(session_info)
    ]


def _load_session_rows(
    csv_path: str,
    session: SessionInfo,
    time_bins: Sequence[int],
    average_bins: bool,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"responses", "image_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("%s missing columns %r" % (csv_path, sorted(missing)))

    parsed = [
        parse_response_cell(
            value,
            num_units=session.num_units,
            time_bins=time_bins,
            average_bins=average_bins,
        )
        for value in df["responses"].tolist()
    ]
    df = df.copy()
    df["_response_vector"] = parsed
    if "trial_id" not in df:
        df["trial_id"] = np.arange(len(df), dtype=np.int64)
    return df


def build_combined_response_table(
    data_root: str,
    split: str,
    time_bins: Sequence[int],
    average_bins: bool = False,
    expected_total_neurons: Optional[int] = 458,
) -> CombinedResponseTable:
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")

    sessions = discover_sessions(data_root)
    total_units = sessions[-1].global_end
    if expected_total_neurons is not None and total_units != expected_total_neurons:
        raise ValueError(
            "Discovered %d neurons; expected %d"
            % (total_units, expected_total_neurons)
        )

    frames = {}
    for session in sessions:
        path = session.train_csv if split == "train" else session.test_csv
        frames[session.session_id] = _load_session_rows(
            path,
            session,
            time_bins=time_bins,
            average_bins=average_bins,
        )

    all_image_ids = sorted(
        set(
            int(x)
            for frame in frames.values()
            for x in frame["image_id"].tolist()
        )
    )

    if split == "train":
        responses = np.zeros((len(all_image_ids), total_units), dtype=np.float32)
        mask = np.zeros((len(all_image_ids), total_units), dtype=bool)
        image_to_row = {image_id: i for i, image_id in enumerate(all_image_ids)}

        for session in sessions:
            frame = frames[session.session_id]
            # Historical loader used first occurrence when a training image
            # appeared more than once in a long session.
            first_rows = frame.drop_duplicates(subset=["image_id"], keep="first")
            for image_id, response in zip(
                first_rows["image_id"].tolist(),
                first_rows["_response_vector"].tolist(),
            ):
                i = image_to_row[int(image_id)]
                responses[i, session.global_start:session.global_end] = response
                mask[i, session.global_start:session.global_end] = True

        keep = mask.any(axis=1)
        return CombinedResponseTable(
            responses=responses[keep],
            mask=mask[keep],
            image_ids=np.asarray(all_image_ids, dtype=np.int64)[keep],
            repeat_index=np.zeros(int(keep.sum()), dtype=np.int64),
            session_info=sessions,
        )

    # Test: union IDs x maximum repeat count, then combine matching repeat
    # ordinal across sessions, matching recovered historical logic.
    max_repeats = 0
    grouped = {}
    for session in sessions:
        frame = frames[session.session_id]
        per_image = {}
        for image_id, group in frame.groupby("image_id", sort=False):
            per_image[int(image_id)] = list(group["_response_vector"])
            max_repeats = max(max_repeats, len(per_image[int(image_id)]))
        grouped[session.session_id] = per_image

    n_rows = len(all_image_ids) * max_repeats
    responses = np.zeros((n_rows, total_units), dtype=np.float32)
    mask = np.zeros((n_rows, total_units), dtype=bool)
    image_ids = np.repeat(
        np.asarray(all_image_ids, dtype=np.int64),
        max_repeats,
    )
    repeat_index = np.tile(
        np.arange(max_repeats, dtype=np.int64),
        len(all_image_ids),
    )

    image_to_base = {
        image_id: i * max_repeats for i, image_id in enumerate(all_image_ids)
    }

    for session in sessions:
        per_image = grouped[session.session_id]
        for image_id, vectors in per_image.items():
            base = image_to_base[image_id]
            for repeat_i, response in enumerate(vectors):
                row_i = base + repeat_i
                responses[
                    row_i,
                    session.global_start:session.global_end,
                ] = response
                mask[
                    row_i,
                    session.global_start:session.global_end,
                ] = True

    keep = mask.any(axis=1)
    return CombinedResponseTable(
        responses=responses[keep],
        mask=mask[keep],
        image_ids=image_ids[keep],
        repeat_index=repeat_index[keep],
        session_info=sessions,
    )


class ImageStore:
    """Image store for the released PNG export (and legacy .npy files).

    `prepare()` mirrors the historical ImageCache behavior: all transformed
    images are materialized as float32, one global scalar mean/std is computed
    before the train/validation split, and the entire bank is z-scored in RAM.
    """

    def __init__(
        self,
        images_root: str,
        filename_pattern: str = "{image_id:06d}.png",
        image_id_offset: int = 0,
    ) -> None:
        self.images_root = Path(images_root)
        self.filename_pattern = filename_pattern
        self.image_id_offset = int(image_id_offset)
        self._prepared = None
        self._prepared_ids = None
        self._id_to_index = None

    def path_for_id(self, image_id: int) -> Path:
        file_id = int(image_id) + self.image_id_offset
        return self.images_root / self.filename_pattern.format(image_id=file_id)

    def _suffix(self) -> str:
        return Path(self.filename_pattern.format(image_id=0)).suffix.lower()

    def list_file_ids(self) -> List[int]:
        suffix = self._suffix()
        ids = []
        for path in sorted(self.images_root.glob("*" + suffix)):
            match = re.search(r"(-?\d+)$", path.stem)
            if match:
                ids.append(int(match.group(1)))
        return ids

    def load(self, image_id: int) -> np.ndarray:
        path = self.path_for_id(image_id)
        if not path.exists():
            raise FileNotFoundError(
                "Image ID %d resolved to missing file %s"
                % (int(image_id), path)
            )
        if path.suffix.lower() == ".npy":
            return np.load(str(path), allow_pickle=False)

        from PIL import Image
        with Image.open(path) as im:
            image = np.asarray(im)
        if image.ndim != 2:
            raise ValueError(
                "Expected grayscale 2D image at %s, got shape %r"
                % (path, image.shape)
            )
        return image

    def prepare(self, cfg: PreprocessConfig, normalization_cfg=None):
        normalization_cfg = normalization_cfg or {}
        mode = str(normalization_cfg.get("mode", "fixed"))
        file_ids = self.list_file_ids()
        if not file_ids:
            raise FileNotFoundError(
                "No %s images found under %s" % (self._suffix(), self.images_root)
            )

        # Response IDs map to file IDs through image_id_offset. The prepared bank
        # is indexed by response-space IDs so __getitem__ stays unambiguous.
        response_ids = [int(fid) - self.image_id_offset for fid in file_ids]
        h, w = cfg.expected_output_hw
        bank = np.empty((len(response_ids), 1, h, w), dtype=np.float32)
        for i, image_id in enumerate(response_ids):
            bank[i] = transform_image_geometry_numpy(self.load(image_id), cfg)

        tensor = torch.from_numpy(bank)
        if mode == "compute_from_all_images":
            # Historical ImageCache.zscore_images used torch.mean/std on the
            # stacked float32 transformed image cache. torch.std here therefore
            # intentionally uses its default Bessel correction.
            mean = float(tensor.mean().item())
            std = float(tensor.std().item())
        elif mode == "fixed":
            if cfg.mean is None or cfg.std is None:
                raise ValueError("fixed normalization requires preprocessing.mean/std")
            mean, std = float(cfg.mean), float(cfg.std)
        else:
            raise ValueError("Unknown normalization mode %r" % mode)

        if not np.isfinite(std) or std <= 0:
            raise ValueError("Computed normalization std is invalid: %r" % std)

        tensor.sub_(mean).div_(std)
        resolved = replace(cfg, mean=mean, std=std)
        self._prepared = bank
        self._prepared_ids = np.asarray(response_ids, dtype=np.int64)
        self._id_to_index = {int(v): i for i, v in enumerate(response_ids)}

        stats = {
            "mode": mode,
            "population": "all_image_files_before_train_validation_split",
            "n_images": len(response_ids),
            "n_pixels": int(bank.size),
            "mean": mean,
            "std": std,
            "dtype_for_statistics": "torch.float32",
            "torch_std_correction": 1,
            "image_id_offset": self.image_id_offset,
            "filename_pattern": self.filename_pattern,
            "crop": [list(cfg.crop[0]), list(cfg.crop[1])],
            "subsample": int(cfg.subsample),
            "scale": float(cfg.scale),
            "expected_output_hw": list(cfg.expected_output_hw),
        }

        expected_mean = normalization_cfg.get("expected_mean")
        expected_std = normalization_cfg.get("expected_std")
        tolerance = float(normalization_cfg.get("expected_tolerance", 0.0))
        if expected_mean is not None:
            stats["expected_mean"] = float(expected_mean)
            stats["mean_delta"] = mean - float(expected_mean)
            if abs(stats["mean_delta"]) > tolerance:
                raise ValueError(
                    "Computed image mean %.8f differs from expected %.8f by %.8f "
                    "(tolerance %.8f)"
                    % (mean, float(expected_mean), stats["mean_delta"], tolerance)
                )
        if expected_std is not None:
            stats["expected_std"] = float(expected_std)
            stats["std_delta"] = std - float(expected_std)
            if abs(stats["std_delta"]) > tolerance:
                raise ValueError(
                    "Computed image std %.8f differs from expected %.8f by %.8f "
                    "(tolerance %.8f)"
                    % (std, float(expected_std), stats["std_delta"], tolerance)
                )
        return resolved, stats

    def load_preprocessed(self, image_id: int) -> torch.Tensor:
        if self._prepared is None or self._id_to_index is None:
            raise RuntimeError("ImageStore.prepare() must be called before dataset use")
        try:
            index = self._id_to_index[int(image_id)]
        except KeyError as exc:
            raise KeyError("Image ID %d not present in prepared bank" % int(image_id)) from exc
        return torch.from_numpy(self._prepared[index])


# Backwards-compatible name for older scripts; the implementation is now format-agnostic.
NpyImageStore = ImageStore


class CombinedNaturalImageDataset(Dataset):
    def __init__(
        self,
        table: CombinedResponseTable,
        image_store: ImageStore,
        preprocess_cfg: PreprocessConfig,
    ) -> None:
        self.table = table
        self.image_store = image_store
        self.preprocess_cfg = preprocess_cfg

    def __len__(self) -> int:
        return len(self.table.image_ids)

    def __getitem__(self, index: int):
        image_id = int(self.table.image_ids[index])
        return {
            "image": self.image_store.load_preprocessed(image_id),
            "response": torch.from_numpy(self.table.responses[index]).float(),
            "mask": torch.from_numpy(self.table.mask[index]).bool(),
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "repeat_index": torch.tensor(
                int(self.table.repeat_index[index]),
                dtype=torch.long,
            ),
        }


def dataset_manifest_fingerprint(
    data_root: str,
    full_sha256: bool = False,
) -> str:
    root = Path(data_root)
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        stat = path.stat()
        h.update(rel.encode("utf-8"))
        h.update(str(stat.st_size).encode("ascii"))
        if full_sha256:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
        else:
            h.update(str(stat.st_mtime_ns).encode("ascii"))
    return h.hexdigest()