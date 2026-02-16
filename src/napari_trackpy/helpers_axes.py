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
        # Try explicit metadata first (e.g. from AICS/BioFormats).
        axis_spec = None
        meta = getattr(layer, "metadata", {}) or {}
        for key in ("axes", "axis_order", "dims", "dimension_order", "DimensionOrder"):
            value = meta.get(key)
            if isinstance(value, str) and len(value) == ndim:
                axis_spec = value.upper()
                break

        if axis_spec is not None:
            # If channel is present we cannot map directly to one image layer.
            if "C" not in axis_spec and {"T", "Z", "Y", "X"}.issubset(set(axis_spec)):
                return Axes(
                    "tzyx",
                    axis_spec.index("T"),
                    axis_spec.index("Z"),
                    axis_spec.index("Y"),
                    axis_spec.index("X"),
                )

        # Heuristic fallback for TZYX vs ZTYX.
        # pick Y/X as last two axes; resolve T/Z among first two axes.
        scale0 = float(layer.scale[0])
        scale1 = float(layer.scale[1])
        n0 = int(layer.data.shape[0])
        n1 = int(layer.data.shape[1])

        # Time axis is frequently unit-scaled while z is not.
        if np.isclose(scale0, 1.0) and not np.isclose(scale1, 1.0):
            t_axis, z_axis = 0, 1
        elif np.isclose(scale1, 1.0) and not np.isclose(scale0, 1.0):
            t_axis, z_axis = 1, 0
        else:
            # Ambiguous: pick larger extent as time by default.
            if n0 >= n1:
                t_axis, z_axis = 0, 1
            else:
                t_axis, z_axis = 1, 0

        return Axes("tzyx", t_axis, z_axis, 2, 3)

    # ---------------- fallback ---------------------------------------------
    raise ValueError(
        f"Unsupported data dimensionality: {ndim}D. "
        "Only YX, ZYX, TYX, and TZYX are handled."
    )
