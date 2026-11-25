

# --- test_preprocessing.py ---
import numpy as np

from sinz_model.data import PreprocessConfig, transform_image_numpy


def test_233_to_93_crop_no_resize():
    cfg = PreprocessConfig(
        crop=((70, 70), (70, 70)),
        subsample=1,
        scale=1.0,
        expected_output_hw=(93, 93),
        mean=124.54466,
        std=70.28,
    )
    image = np.arange(233 * 233, dtype=np.float32).reshape(233, 233)
    out = transform_image_numpy(image, cfg)
    assert out.shape == (1, 93, 93)

    expected_center = (image[70, 70] - cfg.mean) / cfg.std
    assert np.isclose(out[0, 0, 0], expected_center)


# --- test_splits.py ---
import numpy as np

from sinz_model.data import create_split


def test_split_is_reproducible_and_disjoint():
    ids = np.arange(100)
    a = create_split(ids, 0.8, 8000)
    b = create_split(ids, 0.8, 8000)
    assert a.sha256 == b.sha256
    assert np.array_equal(a.train_image_ids, b.train_image_ids)
    assert np.array_equal(a.val_image_ids, b.val_image_ids)
    assert set(a.train_image_ids).isdisjoint(set(a.val_image_ids))
    assert len(a.train_image_ids) == 80


# --- test_dataset.py ---
import numpy as np

from sinz_model.data import parse_response_cell


def test_parse_response_units_by_time_sum():
    arr = np.arange(2 * 12, dtype=float).reshape(2, 12)
    out = parse_response_cell(
        arr.tolist(),
        num_units=2,
        time_bins=list(range(12)),
        average_bins=False,
    )
    assert out.shape == (2,)
    assert np.allclose(out, arr.sum(axis=1))


def test_parse_response_time_by_units_sum():
    arr = np.arange(2 * 12, dtype=float).reshape(12, 2)
    out = parse_response_cell(
        arr.tolist(),
        num_units=2,
        time_bins=list(range(12)),
        average_bins=False,
    )
    assert np.allclose(out, arr.sum(axis=0))


def test_parse_numpy_style_string_without_commas():
    text = "[[0. 1. 2.] [3. 4. 5.]]"
    out = parse_response_cell(
        text,
        num_units=2,
        time_bins=[0, 1, 2],
        average_bins=False,
    )
    assert np.allclose(out, [3.0, 12.0])


def test_parse_slices_selected_bins_before_summing():
    arr = np.arange(2 * 16, dtype=float).reshape(2, 16)
    out = parse_response_cell(
        arr.tolist(), num_units=2, time_bins=[2, 3, 4], average_bins=False
    )
    assert np.allclose(out, arr[:, [2, 3, 4]].sum(axis=1))
