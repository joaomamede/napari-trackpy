from __future__ import annotations

import importlib.util

import numpy as np


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


try:
    from numba import njit
except Exception:
    njit = None


if njit is not None:

    @njit(cache=True)
    def _fill_labels_numba_2d(mask, pos, rad):
        h, w = mask.shape
        r2 = rad * rad
        n = pos.shape[0]
        for i in range(n):
            py = pos[i, 0]
            px = pos[i, 1]
            y0 = max(0, int(np.floor(py - rad)))
            y1 = min(h - 1, int(np.ceil(py + rad)))
            x0 = max(0, int(np.floor(px - rad)))
            x1 = min(w - 1, int(np.ceil(px + rad)))
            for y in range(y0, y1 + 1):
                dy2 = (y - py) * (y - py)
                for x in range(x0, x1 + 1):
                    if mask[y, x] != 0:
                        continue
                    dx2 = (x - px) * (x - px)
                    if dy2 + dx2 <= r2:
                        mask[y, x] = i + 1


    @njit(cache=True)
    def _fill_labels_numba_3d(mask, pos, rad, rat):
        zdim, ydim, xdim = mask.shape
        rz = rad * rat
        if rz <= 0:
            rz = 1e-6
        rz2 = rz * rz
        r2 = rad * rad
        n = pos.shape[0]
        for i in range(n):
            pz = pos[i, 0]
            py = pos[i, 1]
            px = pos[i, 2]
            z0 = max(0, int(np.floor(pz - rz)))
            z1 = min(zdim - 1, int(np.ceil(pz + rz)))
            y0 = max(0, int(np.floor(py - rad)))
            y1 = min(ydim - 1, int(np.ceil(py + rad)))
            x0 = max(0, int(np.floor(px - rad)))
            x1 = min(xdim - 1, int(np.ceil(px + rad)))

            for z in range(z0, z1 + 1):
                dz2 = (z - pz) * (z - pz)
                for y in range(y0, y1 + 1):
                    dy2 = (y - py) * (y - py)
                    for x in range(x0, x1 + 1):
                        if mask[z, y, x] != 0:
                            continue
                        dx2 = (x - px) * (x - px)
                        if (dz2 / rz2) + ((dy2 + dx2) / r2) <= 1.0:
                            mask[z, y, x] = i + 1


def available_mask_algorithms() -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    if _has_module("numba"):
        options.append(("numba", "numba (default)"))
    options.append(("cpu", "CPU numpy"))
    if _has_module("cupy"):
        options.append(("gpu", "GPU cupy"))
    if _has_module("pyopencl"):
        options.append(("opencl", "openCL"))
    return options


def available_intensity_backends() -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = [("cpu", "CPU numpy")]
    if _has_module("cupy") and _has_module("cucim"):
        options.append(("gpu", "GPU cupy/cucim"))
    return options


def release_backend_memory(backend: str | None = None) -> None:
    """Best-effort memory cleanup for long-running batch jobs."""
    b = (backend or "all").strip().lower()

    if b in ("all", "gpu", "cupy"):
        try:
            import cupy as cp

            cp.cuda.runtime.deviceSynchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    if b in ("all", "opencl", "pyopencl"):
        try:
            import pyopencl as cl

            try:
                cl.tools.clear_first_arg_caches()
            except Exception:
                pass
        except Exception:
            pass


def _as_positions(points_df, ndim: int) -> np.ndarray:
    if points_df is None or len(points_df) == 0:
        return np.empty((0, ndim), dtype=np.float32)

    cols = list(points_df.columns)
    if ndim == 3 and all(c in cols for c in ("z", "y", "x")):
        pos = points_df.loc[:, ["z", "y", "x"]].to_numpy(dtype=np.float32)
    elif ndim == 2 and all(c in cols for c in ("y", "x")):
        pos = points_df.loc[:, ["y", "x"]].to_numpy(dtype=np.float32)
    elif ndim == 2 and all(c in cols for c in ("z", "y", "x")):
        pos = points_df.loc[:, ["y", "x"]].to_numpy(dtype=np.float32)
    elif ndim == 3 and all(c in cols for c in ("y", "x")):
        z = np.zeros((len(points_df), 1), dtype=np.float32)
        pos = np.hstack([z, points_df.loc[:, ["y", "x"]].to_numpy(dtype=np.float32)])
    else:
        take = min(ndim, len(cols))
        pos = points_df.iloc[:, -take:].to_numpy(dtype=np.float32)
        if pos.shape[1] != ndim:
            out = np.zeros((len(points_df), ndim), dtype=np.float32)
            out[:, -pos.shape[1] :] = pos
            pos = out

    return np.atleast_2d(pos)


def _labels_numpy(shape, positions: np.ndarray, radius: float, ratio: float = 1.0) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.uint32)
    if len(shape) == 2:
        h, w = shape
        r2 = float(radius) * float(radius)
        for i, (py, px) in enumerate(positions, start=1):
            y0 = max(0, int(np.floor(py - radius)))
            y1 = min(h - 1, int(np.ceil(py + radius)))
            x0 = max(0, int(np.floor(px - radius)))
            x1 = min(w - 1, int(np.ceil(px + radius)))
            if y1 < y0 or x1 < x0:
                continue
            yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
            inside = (yy - py) ** 2 + (xx - px) ** 2 <= r2
            roi = labels[y0 : y1 + 1, x0 : x1 + 1]
            roi[(roi == 0) & inside] = i
        return labels

    zdim, ydim, xdim = shape
    rz = max(float(radius) * float(ratio), 1e-6)
    rz2 = rz * rz
    r2 = float(radius) * float(radius)

    for i, (pz, py, px) in enumerate(positions, start=1):
        z0 = max(0, int(np.floor(pz - rz)))
        z1 = min(zdim - 1, int(np.ceil(pz + rz)))
        y0 = max(0, int(np.floor(py - radius)))
        y1 = min(ydim - 1, int(np.ceil(py + radius)))
        x0 = max(0, int(np.floor(px - radius)))
        x1 = min(xdim - 1, int(np.ceil(px + radius)))
        if z1 < z0 or y1 < y0 or x1 < x0:
            continue

        zz, yy, xx = np.ogrid[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
        inside = ((zz - pz) ** 2) / rz2 + ((yy - py) ** 2 + (xx - px) ** 2) / r2 <= 1.0
        roi = labels[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
        roi[(roi == 0) & inside] = i

    return labels


def _labels_numba(shape, positions: np.ndarray, radius: float, ratio: float = 1.0) -> np.ndarray:
    if njit is None:
        return _labels_numpy(shape, positions, radius, ratio)

    labels = np.zeros(shape, dtype=np.uint32)
    if len(shape) == 2:
        _fill_labels_numba_2d(labels, np.asarray(positions, dtype=np.float32), float(radius))
    else:
        _fill_labels_numba_3d(
            labels,
            np.asarray(positions, dtype=np.float32),
            float(radius),
            float(ratio),
        )
    return labels


def _labels_cupy(shape, positions: np.ndarray, radius: float, ratio: float = 1.0) -> np.ndarray:
    import cupy as cp

    labels = None
    pos_cp = None
    out = None
    try:
        labels = cp.zeros(shape, dtype=cp.uint32)
        pos_cp = cp.asarray(positions, dtype=cp.float32)

        if len(shape) == 2:
            h, w = shape
            rad = float(radius)
            r2 = rad * rad
            for i in range(pos_cp.shape[0]):
                py = float(pos_cp[i, 0].item())
                px = float(pos_cp[i, 1].item())
                y0 = max(0, int(np.floor(py - rad)))
                y1 = min(h - 1, int(np.ceil(py + rad)))
                x0 = max(0, int(np.floor(px - rad)))
                x1 = min(w - 1, int(np.ceil(px + rad)))
                if y1 < y0 or x1 < x0:
                    continue
                yy, xx = cp.ogrid[y0 : y1 + 1, x0 : x1 + 1]
                inside = (yy - py) ** 2 + (xx - px) ** 2 <= r2
                roi = labels[y0 : y1 + 1, x0 : x1 + 1]
                labels[y0 : y1 + 1, x0 : x1 + 1] = cp.where((roi == 0) & inside, i + 1, roi)
            out = cp.asnumpy(labels)
            return out

        zdim, ydim, xdim = shape
        rad = float(radius)
        rz = max(float(radius) * float(ratio), 1e-6)
        r2 = rad * rad
        rz2 = rz * rz
        for i in range(pos_cp.shape[0]):
            pz = float(pos_cp[i, 0].item())
            py = float(pos_cp[i, 1].item())
            px = float(pos_cp[i, 2].item())

            z0 = max(0, int(np.floor(pz - rz)))
            z1 = min(zdim - 1, int(np.ceil(pz + rz)))
            y0 = max(0, int(np.floor(py - rad)))
            y1 = min(ydim - 1, int(np.ceil(py + rad)))
            x0 = max(0, int(np.floor(px - rad)))
            x1 = min(xdim - 1, int(np.ceil(px + rad)))
            if z1 < z0 or y1 < y0 or x1 < x0:
                continue

            zz, yy, xx = cp.ogrid[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
            inside = ((zz - pz) ** 2) / rz2 + ((yy - py) ** 2 + (xx - px) ** 2) / r2 <= 1.0
            roi = labels[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
            labels[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = cp.where((roi == 0) & inside, i + 1, roi)

        out = cp.asnumpy(labels)
        return out
    finally:
        del pos_cp
        del labels
        if out is not None:
            del out
        release_backend_memory("gpu")


def _labels_opencl(shape, positions: np.ndarray, radius: float, ratio: float = 1.0) -> np.ndarray:
    import pyopencl as cl

    mask = np.zeros(shape, dtype=np.uint32)
    positions_flat = np.asarray(positions, dtype=np.float32).ravel()
    radius_xy = np.float32(radius)
    radius_z = np.float32(max(float(radius) * float(ratio), 1e-6))

    platforms = cl.get_platforms()
    if not platforms:
        raise RuntimeError("No OpenCL platform found")

    devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
    if not devices:
        devices = platforms[0].get_devices()
    if not devices:
        raise RuntimeError("No OpenCL devices available")

    ctx = None
    queue = None
    mask_buf = None
    positions_buf = None
    program = None
    try:
        ctx = cl.Context(devices=devices)
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        mask_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=mask)
        positions_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=positions_flat)

        if len(shape) == 3:
            kernel_code = """
            __kernel void fill_mask(
                __global uint *mask,
                __global const float *positions,
                const float radius_z,
                const float radius_xy,
                const int n_positions
            ) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int z = get_global_id(2);
                int width = get_global_size(0);
                int height = get_global_size(1);
                int idx = x + y * width + z * width * height;

                for (int i = 0; i < n_positions; ++i) {
                    float pz = positions[i * 3];
                    float py = positions[i * 3 + 1];
                    float px = positions[i * 3 + 2];

                    float dz = (pz - z) / radius_z;
                    float dxy = ((px - x) * (px - x) + (py - y) * (py - y)) / (radius_xy * radius_xy);
                    if ((dz * dz + dxy) <= 1.0f) {
                        if (mask[idx] == 0) {
                            mask[idx] = (uint)(i + 1);
                        }
                        break;
                    }
                }
            }
            """
        else:
            kernel_code = """
            __kernel void fill_mask(
                __global uint *mask,
                __global const float *positions,
                const float radius_z,
                const float radius_xy,
                const int n_positions
            ) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int width = get_global_size(0);
                int idx = x + y * width;

                for (int i = 0; i < n_positions; ++i) {
                    float py = positions[i * 2];
                    float px = positions[i * 2 + 1];
                    float dxy = ((px - x) * (px - x) + (py - y) * (py - y)) / (radius_xy * radius_xy);
                    if (dxy <= 1.0f) {
                        if (mask[idx] == 0) {
                            mask[idx] = (uint)(i + 1);
                        }
                        break;
                    }
                }
            }
            """

        program = cl.Program(ctx, kernel_code).build()
        global_size = tuple(reversed(shape))
        program.fill_mask(
            queue,
            global_size,
            None,
            mask_buf,
            positions_buf,
            radius_z,
            radius_xy,
            np.int32(len(positions)),
        )
        queue.finish()
        cl.enqueue_copy(queue, mask, mask_buf)
        queue.finish()
        return mask
    finally:
        for obj in (mask_buf, positions_buf, queue, program, ctx):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass
        release_backend_memory("opencl")


def make_labels_trackpy_links(
    shape,
    j,
    radius: float = 5,
    ratio: float = 1.0,
    _algo: str = "numba",
):
    """Create per-particle label masks, preserving the original point order as label IDs."""
    if len(shape) not in (2, 3):
        raise ValueError(f"Expected 2D or 3D shape, got {shape}")

    algo = (_algo or "numba").strip().lower()
    aliases = {
        "gpu": "gpu",
        "cpu": "cpu",
        "opencl": "opencl",
        "numba": "numba",
        "fast": "cpu",
    }
    algo = aliases.get(algo, algo)

    positions = _as_positions(j, len(shape))
    if len(positions) == 0:
        return np.zeros(shape, dtype=np.uint32), positions

    if algo == "numba":
        labels = _labels_numba(shape, positions, radius=radius, ratio=ratio)
    elif algo == "gpu":
        labels = _labels_cupy(shape, positions, radius=radius, ratio=ratio)
    elif algo == "opencl":
        labels = _labels_opencl(shape, positions, radius=radius, ratio=ratio)
    elif algo == "cpu":
        labels = _labels_numpy(shape, positions, radius=radius, ratio=ratio)
    else:
        labels = _labels_numba(shape, positions, radius=radius, ratio=ratio)

    return labels.astype(np.uint32, copy=False), positions


def _measure_cpu(labels: np.ndarray, intensity_image: np.ndarray):
    import pandas as pd

    lab = np.asarray(labels)
    img = np.asarray(intensity_image, dtype=np.float64)
    if lab.shape != img.shape:
        raise ValueError("labels and intensity_image must have the same shape")

    flat_lab = lab.ravel().astype(np.int64)
    flat_img = img.ravel()
    valid = flat_lab > 0
    if not np.any(valid):
        return pd.DataFrame(columns=["label", "intensity_mean", "intensity_max", "intensity_min"])

    flat_lab = flat_lab[valid]
    flat_img = flat_img[valid]
    max_label = int(flat_lab.max())

    count = np.bincount(flat_lab, minlength=max_label + 1)
    sums = np.bincount(flat_lab, weights=flat_img, minlength=max_label + 1)

    mins = np.full(max_label + 1, np.inf, dtype=np.float64)
    maxs = np.full(max_label + 1, -np.inf, dtype=np.float64)
    np.minimum.at(mins, flat_lab, flat_img)
    np.maximum.at(maxs, flat_lab, flat_img)

    used = np.where(count[1:] > 0)[0] + 1
    mean = np.divide(sums[used], count[used], out=np.zeros_like(sums[used]), where=count[used] > 0)

    return pd.DataFrame(
        {
            "label": used.astype(np.int64),
            "intensity_mean": mean,
            "intensity_max": maxs[used],
            "intensity_min": mins[used],
        }
    )


def _measure_gpu(labels: np.ndarray, intensity_image: np.ndarray):
    import pandas as pd
    import cupy as cp
    from cucim.skimage.measure import regionprops_table

    labels_cp = None
    image_cp = None
    props = None
    try:
        labels_cp = cp.asarray(labels)
        image_cp = cp.asarray(intensity_image)
        props = regionprops_table(
            labels_cp,
            intensity_image=image_cp,
            properties=("label", "intensity_mean", "intensity_max", "intensity_min"),
        )
        if not props:
            return pd.DataFrame(columns=["label", "intensity_mean", "intensity_max", "intensity_min"])

        out = {}
        for key, values in props.items():
            try:
                out[key] = cp.asnumpy(values)
            except Exception:
                out[key] = np.asarray(values)
        return pd.DataFrame(out)
    finally:
        del labels_cp
        del image_cp
        del props
        release_backend_memory("gpu")


def measure_label_intensity(labels, intensity_image, backend: str = "cpu"):
    backend = (backend or "cpu").strip().lower()
    if backend == "gpu":
        if not (_has_module("cupy") and _has_module("cucim")):
            raise RuntimeError("GPU intensity backend requested but cupy/cucim are not available")
        return _measure_gpu(labels, intensity_image)
    return _measure_cpu(labels, intensity_image)
