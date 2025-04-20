# helpers_axes.py  (drop it anywhere on your PYTHONPATH)
from __future__ import annotations
from typing import NamedTuple, Literal

import numpy as np
import napari


AxisOrder = Literal["yx", "zyx", "tyx", "tzyx"]


class Axes(NamedTuple):
    order: AxisOrder         # one of the four strings above
    t:    int | None         # index of time axis   in layer.data
    z:    int | None         # index of z axis      in layer.data
    y:    int                # index of y (rows)
    x:    int                # index of x (cols)


def infer_axes(layer: napari.layers.Image) -> Axes:
    """Best‑effort guess of axis role based on ndim and layer.scale."""
    ndim = layer.data.ndim
    y, x = ndim - 2, ndim - 1           # always true in napari

    # ---------------- 2‑D   ------------------------------------------------
    if ndim == 2:
        return Axes("yx", None, None, y, x)

    # ---------------- 3‑D   ------------------------------------------------
    if ndim == 3:
        first_scale = float(layer.scale[0])
        spatial_scale = float(layer.scale[y])          # same as scale[x]

        if np.isclose(first_scale, 1):                 # TYX (time first)
            return Axes("tyx", 0, None, y, x)
        else:                                          # ZYX   (z first)
            return Axes("zyx", None, 0, y, x)

    # ---------------- 4‑D   ------------------------------------------------
    if ndim == 4:
        # convention in bioformats & AICS: T Z Y X
        return Axes("tzyx", 0, 1, 2, 3)

    # ---------------- fallback ---------------------------------------------
    raise ValueError(
        f"Unsupported data dimensionality: {ndim}D. "
        "Only YX, ZYX, TYX, and TZYX are handled."
    )
