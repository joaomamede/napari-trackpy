import numpy as np
import pandas as pd

from napari_trackpy.mask_measurements import make_labels_trackpy_links, measure_label_intensity


def test_make_labels_trackpy_links_keeps_point_index_as_label_cpu():
    points = pd.DataFrame(
        {
            "y": [6.0, 14.0, 22.0],
            "x": [6.0, 14.0, 22.0],
        }
    )

    labels, pos = make_labels_trackpy_links((30, 30), points, radius=2.5, _algo="cpu")

    assert labels.shape == (30, 30)
    assert pos.shape == (3, 2)
    # The center pixel for each point should keep the original label id (1-based order).
    assert labels[6, 6] == 1
    assert labels[14, 14] == 2
    assert labels[22, 22] == 3


def test_make_labels_trackpy_links_numba_matches_cpu_ids_when_available():
    try:
        import numba  # noqa: F401
    except Exception:
        return

    points = pd.DataFrame(
        {
            "z": [2.0, 4.0],
            "y": [8.0, 18.0],
            "x": [8.0, 18.0],
        }
    )

    labels_cpu, _ = make_labels_trackpy_links((7, 32, 32), points, radius=2.0, ratio=1.0, _algo="cpu")
    labels_numba, _ = make_labels_trackpy_links((7, 32, 32), points, radius=2.0, ratio=1.0, _algo="numba")

    assert labels_cpu[2, 8, 8] == 1
    assert labels_cpu[4, 18, 18] == 2
    assert labels_numba[2, 8, 8] == 1
    assert labels_numba[4, 18, 18] == 2


def test_measure_label_intensity_cpu_returns_expected_stats():
    labels = np.array(
        [
            [0, 1, 1],
            [0, 2, 2],
            [0, 0, 2],
        ],
        dtype=np.uint32,
    )
    image = np.array(
        [
            [5.0, 1.0, 3.0],
            [8.0, 10.0, 4.0],
            [7.0, 6.0, 2.0],
        ],
        dtype=np.float32,
    )

    out = measure_label_intensity(labels, image, backend="cpu")
    out = out.sort_values("label").reset_index(drop=True)

    assert list(out["label"].astype(int)) == [1, 2]
    assert np.isclose(out.loc[0, "intensity_mean"], 2.0)
    assert np.isclose(out.loc[0, "intensity_min"], 1.0)
    assert np.isclose(out.loc[0, "intensity_max"], 3.0)
    assert np.isclose(out.loc[1, "intensity_mean"], (10.0 + 4.0 + 2.0) / 3.0)
    assert np.isclose(out.loc[1, "intensity_min"], 2.0)
    assert np.isclose(out.loc[1, "intensity_max"], 10.0)
