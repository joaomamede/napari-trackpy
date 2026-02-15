import numpy as np
import pandas as pd
import pytest
import importlib


def _gaussian_spot_image(shape, centers, amplitude=300.0, sigma=1.0, noise=0.5):
    yy, xx = np.indices(shape, dtype=np.float32)
    image = np.random.default_rng(0).normal(0.0, noise, shape).astype(np.float32)
    for cy, cx in centers:
        image += amplitude * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))
    image -= image.min()
    return image


def test_trackpy_locates_synthetic_gaussian_spots_close_to_truth():
    tp = pytest.importorskip("trackpy")

    true_centers = np.array([[20.0, 25.0], [45.0, 50.0], [30.0, 40.0]], dtype=np.float32)
    image = _gaussian_spot_image((64, 64), true_centers, amplitude=350.0, sigma=1.1, noise=0.3)

    found = tp.locate(image, diameter=5, minmass=25, engine="numba")
    assert len(found) >= len(true_centers)

    detected = found[["y", "x"]].to_numpy(dtype=np.float32)
    for center in true_centers:
        dist = np.sqrt(((detected - center) ** 2).sum(axis=1)).min()
        assert dist < 1.5


def test_trackpy_properties_keeps_all_non_coordinate_columns():
    pytest.importorskip("napari")
    widget_mod = importlib.import_module("napari_trackpy._widget")

    df = pd.DataFrame(
        {
            "frame": [0, 1],
            "y": [10.0, 11.0],
            "x": [20.0, 21.0],
            "mass": [100.0, 120.0],
            "size": [1.4, 1.6],
            "ecc": [0.2, 0.1],
            "signal": [15.0, 18.0],
        }
    )

    props = widget_mod.IdentifyQWidget._trackpy_properties(None, df, ["frame", "y", "x"])
    assert set(props) == {"mass", "size", "ecc", "signal"}
    assert np.allclose(props["mass"], [100.0, 120.0])


def test_linking_sanitize_link_input_drops_non_finite_rows_and_casts_frame():
    pytest.importorskip("napari")
    widget_mod = importlib.import_module("napari_trackpy._widget")

    df = pd.DataFrame(
        {
            "frame": [0.0, np.nan, 2.4, np.inf],
            "y": [10.0, 11.0, np.inf, 13.0],
            "x": [20.0, 21.0, 22.0, np.nan],
            "mass": [1, 2, 3, 4],
        }
    )

    clean = widget_mod.LinkingQWidget._sanitize_link_input(None, df)
    assert len(clean) == 1
    assert clean["frame"].dtype.kind in {"i", "u"}
    assert clean.iloc[0]["frame"] == 0


def test_append_layer_properties_ignores_series_index_and_keeps_row_alignment():
    pytest.importorskip("napari")
    widget_mod = importlib.import_module("napari_trackpy._widget")

    base = pd.DataFrame(
        {
            "frame": [0.0, 1.0, 2.0],
            "y": [10.0, 11.0, 12.0],
            "x": [20.0, 21.0, 22.0],
        }
    )
    props = {
        "mass": pd.Series([100.0, 101.0, 102.0], index=[100, 101, 102]),
        "ep_x": pd.Series([0.1, 0.2, 0.3], index=[200, 201, 202]),
    }

    widget_mod._append_layer_properties(base, props)
    assert len(base) == 3
    assert np.allclose(base["mass"].to_numpy(), [100.0, 101.0, 102.0])
    assert np.allclose(base["ep_x"].to_numpy(), [0.1, 0.2, 0.3])


def test_drop_invalid_coordinate_rows_removes_nan_and_inf():
    pytest.importorskip("napari")
    widget_mod = importlib.import_module("napari_trackpy._widget")

    df = pd.DataFrame(
        {
            "frame": [0, 1, np.inf, 3],
            "z": [1.0, np.nan, 2.0, 3.0],
            "y": [10.0, 11.0, 12.0, np.nan],
            "x": [20.0, 21.0, 22.0, 23.0],
            "mass": [100, 200, 300, 400],
        }
    )
    out = widget_mod._drop_invalid_coordinate_rows(df, ["frame", "z", "y", "x"])
    assert len(out) == 1
    assert out.iloc[0]["frame"] == 0


def test_apply_bounds_filter_supports_min_max_and_range():
    pytest.importorskip("napari")
    widget_mod = importlib.import_module("napari_trackpy._widget")

    df = pd.DataFrame(
        {
            "size_x": [0.8, 1.2, 1.8, 2.5],
            "size_y": [0.9, 1.3, 1.7, 2.6],
        }
    )

    min_only = widget_mod.IdentifyQWidget._apply_bounds_filter(None, df, ["size_x", "size_y"], True, 1.0, False, 0.0)
    assert len(min_only) == 3

    max_only = widget_mod.IdentifyQWidget._apply_bounds_filter(None, df, ["size_x", "size_y"], False, 0.0, True, 2.0)
    assert len(max_only) == 3

    in_range = widget_mod.IdentifyQWidget._apply_bounds_filter(None, df, ["size_x", "size_y"], True, 1.0, True, 2.0)
    assert len(in_range) == 2
