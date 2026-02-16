"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from pathlib import Path
from typing import TYPE_CHECKING
import json
from magicgui import magic_factory
from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout,QPushButton, QCheckBox, QComboBox, QSpinBox
from qtpy.QtWidgets import QLabel, QDoubleSpinBox, QWidget, QGridLayout
from qtpy.QtWidgets import (
    QSpacerItem,
    QSizePolicy,
    QFileDialog,
    QLineEdit,
    QDialog,
    QMessageBox,
    QStyle,
    QProgressBar,
    QPlainTextEdit,
    QApplication,
)
from napari.qt.threading import thread_worker
import numpy as np
from .mask_measurements import (
    available_intensity_backends,
    available_mask_algorithms,
    make_labels_trackpy_links,
    measure_label_intensity,
    release_backend_memory,
)

# viewer.layers[0].source.path


# from support_libraries import make_labels_trackpy_links

if TYPE_CHECKING:
    import napari



# def open_file_dialog(self):
#     from pathlib import Path
#     filename, ok = QFileDialog.getSaveFileName(
#         self,
#         "Select a File", 
#         #modify for last folder used
#         "/tmp/", 
#         "Comma Separated Files (*.csv)"
#     )
#     if filename:
#         path = Path(filename)
#         return str(path)
#     else:
#         return "/tmp/Null.csv"

def _populate_layers(self,_widget,_type='image'):
    # self.layersbox.clear()
    for layer in self.viewer.layers:
        if layer._type_string == _type:
            _widget.addItem(layer.name)


def _points_kwargs_compat(viewer, kwargs):
    import inspect

    try:
        params = inspect.signature(viewer.add_points).parameters
    except Exception:
        return dict(kwargs)

    out = dict(kwargs)
    if (
        "border_width" in params
        and "edge_width" in out
        and "border_width" not in out
    ):
        out["border_width"] = out.pop("edge_width")
    elif "edge_width" in params and "border_width" in out and "edge_width" not in out:
        out["edge_width"] = out.pop("border_width")

    if (
        "border_color" in params
        and "edge_color" in out
        and "border_color" not in out
    ):
        out["border_color"] = out.pop("edge_color")
    elif "edge_color" in params and "border_color" in out and "edge_color" not in out:
        out["edge_color"] = out.pop("border_color")

    return out


def _add_points_compat(viewer, data, **kwargs):
    compat_kwargs = _points_kwargs_compat(viewer, kwargs)
    try:
        return viewer.add_points(data, **compat_kwargs)
    except TypeError:
        fallback = dict(kwargs)
        if "edge_width" in fallback and "border_width" not in fallback:
            fallback["border_width"] = fallback.pop("edge_width")
        elif "border_width" in fallback and "edge_width" not in fallback:
            fallback["edge_width"] = fallback.pop("border_width")

        if "edge_color" in fallback and "border_color" not in fallback:
            fallback["border_color"] = fallback.pop("edge_color")
        elif "border_color" in fallback and "edge_color" not in fallback:
            fallback["edge_color"] = fallback.pop("border_color")

        if fallback == compat_kwargs:
            raise
        return viewer.add_points(data, **fallback)

def _get_choice_layer(self,_widget):    
    if len(self.viewer.layers) == 0:
        raise ValueError("No layers available.")

    current_text = _widget.currentText()
    for j, layer in enumerate(self.viewer.layers):
        if layer.name == current_text:
            print("Layer where points are is:", j)
            return j

    # Fallback to the combobox index if names changed but order still matches.
    idx = _widget.currentIndex()
    if 0 <= idx < len(self.viewer.layers):
        print("Layer where points are is:", idx)
        return idx

    raise ValueError(f"Could not find selected layer '{current_text}'.")

def _get_open_filename(self,type='image',separator= " :: ",choice_widget=None):
    import os
    # from napari.utils import history
    # _last_folder = history.get_open_history()[0]
    # for i in range(len(self.viewer.layers)-1,-1,-1):
    #     if self.viewer.layers[i]._type_string == type:
    #         _filename = self.viewer.layers[i].name.split(separator)[0]
    #         _filename = _last_folder +"/"+ _filename
    #         break
    try:
        j = _get_choice_layer(self,self.layersbox)
    except:
        j = 0
    try:
        _filename = os.path.splitext(self.viewer.layers[j]._source.path)[0]
    except:
        try:
            _filename = os.path.splitext(self.viewer.layers[j]._filename)[0]
        except:
            _filename = 'unknown_file.nd2'
    return _filename


def _append_layer_properties(df, props):
    """Append napari layer properties to a dataframe by row position."""
    n_rows = len(df)
    for key, values in (props or {}).items():
        try:
            arr = np.asarray(values)
        except Exception:
            continue

        if arr.ndim == 0 or arr.shape[0] != n_rows:
            continue

        if arr.ndim == 1:
            df[key] = arr
            continue

        if arr.ndim == 2 and arr.shape[1] == 1:
            df[key] = arr[:, 0]
            continue

        # Multi-component properties become numbered columns.
        if arr.ndim == 2:
            for i in range(arr.shape[1]):
                df[f"{key}_{i}"] = arr[:, i]
            continue

        # Higher-dimensional payloads are stored as objects.
        df[key] = [v for v in values]


def _drop_invalid_coordinate_rows(df, coord_cols):
    """Drop rows with non-finite coordinate values for robust napari/trackpy usage."""
    import pandas as pd

    if not coord_cols:
        return df

    out = df.copy()
    for col in coord_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out.dropna(subset=coord_cols).copy()
    
# def make_labels_trackpy_links(shape,j,radius=5,_algo="GPU"):
#     import trackpy as tp
#     import scipy.ndimage as ndi
#     from scipy.ndimage import binary_dilation

#     if _algo == "GPU":
#         import cupy as cp

#         #outputsomehow is 3D, we want 2
#         # pos = cp.dstack((round(j.y),round(j.x)))[0].astype(int)
#         # if j.z:
#         if 'z' in j:
#             # "Need to loop each t and do one at a time"
#             pos = cp.dstack((j.z,j.y,j.x))[0]#.astype(int)
#             print("3D",j)
#         else:
#             pos = cp.dstack((j.y,j.x))[0]#.astype(int)
#             print("2D",j)


#         ##this is what tp.masks.mask_image does maybe put a cupy here to make if faster.
#         ndim = len(shape)
#         # radius = validate_tuple(radius, ndim)
#         pos = cp.atleast_2d(pos)

#         # if include_edge:
#         in_mask = cp.array([cp.sum(((cp.indices(shape).T - p) / radius)**2, -1) <= 1
#                     for p in pos])
#         # else:
#         #     in_mask = [np.sum(((np.indices(shape).T - p) / radius)**2, -1) < 1
#         #                for p in pos]
#         mask_total = cp.any(in_mask, axis=0).T

#         ##if they overlap the labels won't match the points
#         #we can make np.ones * ID of the point and then np.max(axis=-1)
#         labels, nb = ndi.label(cp.asnumpy(mask_total))
#         # image * mask_cluster.astype(np.uint8)
        
#         #this is super slow
#         # ~ masks = tp.masks.mask_image(coords,np.ones(image.shape),size/2)
#     elif _algo=='CPU':
#         if 'z' in j:
#             # "Need to loop each t and do one at a time"
#             pos = np.dstack((j.z,j.y,j.x))[0]#.astype(int)
#             print("3D",j)
#         else:
#             pos = np.dstack((j.y,j.x))[0]#.astype(int)
#             print("2D",j)

#         ##this is what tp.masks.mask_image does maybe put a cupy here to make if faster.
#         ndim = len(shape)
#         # radius = validate_tuple(radius, ndim)
#         pos = np.atleast_2d(pos)
#         # if include_edge:
#         in_mask = np.array([np.sum(((np.indices(shape).T - p) / radius)**2, -1) <= 1
#                     for p in pos])
#         # else:
#         #     in_mask = [np.sum(((np.indices(shape).T - p) / radius)**2, -1) < 1
#         #                for p in pos]
#         mask_total = np.any(in_mask, axis=0).T
#         ##if they overlap the labels won't match the points
#         #we can make np.ones * ID of the point and then np.max(axis=-1)
#         labels, nb = ndi.label(mask_total)
#     elif _algo=='fast':
#     #This is faster
        
#         # r = (radius-1)/2 # Radius of circles
#         # print(radius,r)
#     #     #make 3D compat
#         disk_mask = tp.masks.binary_mask(radius,len(shape))
#         # print(disk_mask)
#     #     # Initialize output array and set the maskcenters as 1s
#         out = np.zeros(shape,dtype=bool)

#         if 'z' in j:
#             pos = np.dstack((j.z,j.y,j.x))[0].astype(int)
#             pos = np.atleast_2d(pos)
#             print(pos)
#             out[pos[:,0],pos[:,1],pos[:,2]] = 1            

#         else:
#             pos = np.dstack((j.y,j.x))[0].astype(int)
#             pos = np.atleast_2d(pos)
#             print(pos)
#             out[pos[:,0],pos[:,1]] = 1
#     #     # Use binary dilation to get the desired output
    
#         out = binary_dilation(out,disk_mask)

#         labels, nb = ndi.label(out)
#         print("Number of labels:",nb)
#         # if _round:
#         #     return labels, coords
#         # else:
#         #     if image.ndim == 2:
#         #         # coords = j.loc[:,['particle','frame','y','x']]
#         #         coords = j.loc[:,['frame','y','x']]
#         #         # coords = np.dstack((j.particle,j.y,j.x))[0]
#         #         return labels, coords
#     return labels, pos

class IdentifyQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def _parse_diameter(self) -> int | tuple[int, int, int]:
            """
            Return an int if the user typed one number,
            otherwise a (z, y, x) tuple of three ints.
            """
            txt = self.diameter_input.text().strip()
            if "," in txt:
                parts = [int(p) for p in txt.split(",") if p.strip()]
                if len(parts) != 3:
                    raise ValueError(
                        "Diameter tuple must have exactly three integers, e.g. 3,5,5"
                    )
                return tuple(parts)
            return int(txt or 5)   # fall back to default if box is empty
        
    def _diameter_scalar(self) -> int:
        """
        Return an **int** you can plug into radius‑type maths.

        * integer → that same integer  
        * tuple   → its last element (x‑axis), e.g. (3,5,5) → 5
        """
        d = self._parse_diameter()          # from the previous patch
        return d[-1] if isinstance(d, tuple) else d

    def _points_size_from_detection(self) -> float:
        """Napari points marker size in pixels derived from detection diameter."""
        try:
            diameter = float(self._diameter_scalar())
        except Exception:
            diameter = 5.0
        return max(1.0, diameter * 2.0)

    def _trackpy_properties(self, df, coord_cols) -> dict:
        """Keep every trackpy output column except point coordinates."""
        if df is None or len(df) == 0:
            return {}
        exclude = set(coord_cols)
        prop_cols = [c for c in df.columns if c not in exclude]
        return {c: df[c].to_numpy() for c in prop_cols}

    def _apply_bounds_filter(self, df, cols, min_enabled, min_value, max_enabled, max_value):
        if not min_enabled and not max_enabled:
            return df
        if isinstance(cols, str):
            cols = [cols]
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return df

        values = df[cols]
        mask = np.ones(len(df), dtype=bool)
        if min_enabled:
            mask &= (values >= float(min_value)).all(axis=1)
        if max_enabled:
            mask &= (values <= float(max_value)).all(axis=1)
        return df[mask]

    def _default_identify_settings(self) -> dict:
        return {
            "mass_threshold": 0,
            "diameter": "3,5,5",
            "size_min_enabled": False,
            "size_min": 0.0,
            "size_max_enabled": False,
            "size_max": 1.6,
            "ecc_min_enabled": False,
            "ecc_min": 0.0,
            "ecc_max_enabled": False,
            "ecc_max": 0.35,
        }

    def _global_settings_path(self) -> Path:
        return Path.home() / ".napari-trackpy" / "identify_last_run.json"

    def _layer_source_path(self, index_layer=None):
        if index_layer is None:
            try:
                index_layer = _get_choice_layer(self, self.layersbox)
            except Exception:
                return None
        if index_layer is None or index_layer < 0 or index_layer >= len(self.viewer.layers):
            return None

        layer = self.viewer.layers[index_layer]
        for src_attr in ("source", "_source"):
            src = getattr(layer, src_attr, None)
            p = getattr(src, "path", None)
            if p:
                return Path(str(p))

        fallback = getattr(layer, "_filename", None)
        if fallback:
            return Path(str(fallback))
        return None

    def _local_settings_path(self, index_layer=None):
        source_path = self._layer_source_path(index_layer)
        if source_path is None:
            return None
        base_dir = source_path if source_path.is_dir() else source_path.parent
        return base_dir / ".napari-trackpy.json"

    def _read_settings_file(self, path):
        if not path or not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
            return payload if isinstance(payload, dict) else {}
        except Exception as err:
            print(f"Could not read settings file {path}: {err}")
            return {}

    def _write_settings_file(self, path, settings):
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(settings, indent=2, sort_keys=True))
        except Exception as err:
            print(f"Could not write settings file {path}: {err}")

    def _batch_channel_key(self, layer_name: str) -> str:
        return layer_name.split("::")[-1].strip()

    def _selected_channel_key(self, index_layer=None):
        if index_layer is None:
            try:
                index_layer = _get_choice_layer(self, self.layersbox)
            except Exception:
                return None
        if index_layer is None or index_layer < 0 or index_layer >= len(self.viewer.layers):
            return None
        return self._batch_channel_key(self.viewer.layers[index_layer].name)

    def _collect_batch_channel_settings(self) -> dict:
        cfg = {}
        rows = self.batch_grid_layout.rowCount()
        for r in range(rows):
            w_label = self.batch_grid_layout.itemAtPosition(r, 0)
            w_spin = self.batch_grid_layout.itemAtPosition(r, 1)
            w_tick = self.batch_grid_layout.itemAtPosition(r, 2)
            if not (w_label and w_spin and w_tick):
                continue
            key = self._batch_channel_key(w_label.widget().text())
            cfg[key] = {
                "mass_threshold": int(w_spin.widget().value()),
                "enabled": bool(w_tick.widget().isChecked()),
            }
        return cfg

    def _apply_batch_channel_settings(self, batch_channels: dict):
        rows = self.batch_grid_layout.rowCount()
        for r in range(rows):
            w_label = self.batch_grid_layout.itemAtPosition(r, 0)
            w_spin = self.batch_grid_layout.itemAtPosition(r, 1)
            w_tick = self.batch_grid_layout.itemAtPosition(r, 2)
            if not (w_label and w_spin and w_tick):
                continue
            key = self._batch_channel_key(w_label.widget().text())
            if key not in batch_channels:
                continue
            channel_cfg = batch_channels[key] or {}
            if "mass_threshold" in channel_cfg:
                w_spin.widget().setValue(int(channel_cfg["mass_threshold"]))
            if "enabled" in channel_cfg:
                w_tick.widget().setChecked(bool(channel_cfg["enabled"]))

    def _collect_identify_settings(self, index_layer=None) -> dict:
        batch_channels = self._collect_batch_channel_settings()
        selected_key = self._selected_channel_key(index_layer=index_layer)
        if selected_key:
            batch_channels.setdefault(selected_key, {})
            batch_channels[selected_key]["mass_threshold"] = int(self.mass_slider.value())
        size_max_enabled = bool(self.size_max_tick.isChecked())
        size_max = float(self.size_max_input.value())
        ecc_max_enabled = bool(self.ecc_max_tick.isChecked())
        ecc_max = float(self.ecc_max_input.value())
        return {
            "mass_threshold": int(self.mass_slider.value()),
            "diameter": (self.diameter_input.text().strip() or "3,5,5"),
            "size_min_enabled": bool(self.size_min_tick.isChecked()),
            "size_min": float(self.size_min_input.value()),
            "size_max_enabled": size_max_enabled,
            "size_max": size_max,
            "ecc_min_enabled": bool(self.ecc_min_tick.isChecked()),
            "ecc_min": float(self.ecc_min_input.value()),
            "ecc_max_enabled": ecc_max_enabled,
            "ecc_max": ecc_max,
            # Backward-compatible keys for older persisted settings readers.
            "size_filter_enabled": size_max_enabled,
            "size_cutoff": size_max,
            "ecc_enabled": ecc_max_enabled,
            "ecc_cutoff": ecc_max,
            "batch_channels": batch_channels,
        }

    def _apply_identify_settings(self, settings: dict, index_layer=None):
        defaults = self._default_identify_settings()
        cfg = {**defaults, **(settings or {})}
        batch_channels = cfg.get("batch_channels", {}) or {}

        mass_value = cfg["mass_threshold"]
        selected_key = self._selected_channel_key(index_layer=index_layer)
        if selected_key and selected_key in batch_channels:
            channel_cfg = batch_channels[selected_key] or {}
            if "mass_threshold" in channel_cfg:
                mass_value = channel_cfg["mass_threshold"]

        try:
            self.mass_slider.setValue(int(mass_value))
        except Exception:
            self.mass_slider.setValue(defaults["mass_threshold"])

        self.diameter_input.setText(str(cfg.get("diameter", defaults["diameter"])))

        size_min_enabled = bool(cfg.get("size_min_enabled", defaults["size_min_enabled"]))
        size_min = cfg.get("size_min", defaults["size_min"])
        size_max_enabled = bool(cfg.get("size_max_enabled", cfg.get("size_filter_enabled", defaults["size_max_enabled"])))
        size_max = cfg.get("size_max", cfg.get("size_cutoff", defaults["size_max"]))

        self.size_min_tick.setChecked(size_min_enabled)
        self.size_max_tick.setChecked(size_max_enabled)
        try:
            self.size_min_input.setValue(float(size_min))
        except Exception:
            self.size_min_input.setValue(defaults["size_min"])
        try:
            self.size_max_input.setValue(float(size_max))
        except Exception:
            self.size_max_input.setValue(defaults["size_max"])

        ecc_min_enabled = bool(cfg.get("ecc_min_enabled", defaults["ecc_min_enabled"]))
        ecc_min = cfg.get("ecc_min", defaults["ecc_min"])
        ecc_max_enabled = bool(cfg.get("ecc_max_enabled", cfg.get("ecc_enabled", defaults["ecc_max_enabled"])))
        ecc_max = cfg.get("ecc_max", cfg.get("ecc_cutoff", defaults["ecc_max"]))

        self.ecc_min_tick.setChecked(ecc_min_enabled)
        self.ecc_max_tick.setChecked(ecc_max_enabled)
        try:
            self.ecc_min_input.setValue(float(ecc_min))
        except Exception:
            self.ecc_min_input.setValue(defaults["ecc_min"])
        try:
            self.ecc_max_input.setValue(float(ecc_max))
        except Exception:
            self.ecc_max_input.setValue(defaults["ecc_max"])

        self._apply_batch_channel_settings(batch_channels)

    def _load_identify_settings(self, index_layer=None):
        # Precedence: defaults < global last-run < local file beside the dataset
        settings = dict(self._default_identify_settings())
        settings.update(self._read_settings_file(self._global_settings_path()))
        settings.update(self._read_settings_file(self._local_settings_path(index_layer)))
        self._apply_identify_settings(settings, index_layer=index_layer)

    def _save_identify_settings(self, index_layer=None):
        settings = self._collect_identify_settings(index_layer=index_layer)
        self._write_settings_file(self._global_settings_path(), settings)
        self._write_settings_file(self._local_settings_path(index_layer), settings)

    def _default_ipcluster_cores(self):
        import os

        total = max(1, int(os.cpu_count() or 1))
        return max(1, total - 2), total

    def _set_ipcluster_button_state(self, running):
        self._ipcluster_running = bool(running)
        if self._ipcluster_running:
            self.ipcluster_btn.setText("Stop ipcluster")
            self.ipcluster_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserStop))
        else:
            self.ipcluster_btn.setText("Start ipcluster")
            self.ipcluster_btn.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))

    def _make_batch_progress_dialog(self, total_steps):
        dlg = QDialog(self)
        dlg.setWindowTitle("Batch Processing Progress")
        dlg.setModal(False)
        layout = QVBoxLayout(dlg)

        dlg.file_label = QLabel("File: -", dlg)
        dlg.stage_label = QLabel("Status: waiting", dlg)
        dlg.progress_bar = QProgressBar(dlg)
        dlg.progress_bar.setRange(0, max(1, int(total_steps)))
        dlg.progress_bar.setValue(0)
        dlg.progress_bar.setFormat("%p%")
        dlg.progress_bar.setStyleSheet(
            "QProgressBar {height: 16px; text-align: center;} "
            "QProgressBar::chunk {background-color: #3b82f6;}"
        )
        dlg.log_box = QPlainTextEdit(dlg)
        dlg.log_box.setReadOnly(True)
        dlg.log_box.setMinimumHeight(140)
        dlg.close_btn = QPushButton("Close", dlg)
        dlg.close_btn.setEnabled(False)
        dlg.close_btn.clicked.connect(dlg.close)

        layout.addWidget(dlg.file_label)
        layout.addWidget(dlg.stage_label)
        layout.addWidget(dlg.progress_bar)
        layout.addWidget(dlg.log_box)
        layout.addWidget(dlg.close_btn)
        dlg.resize(560, 300)
        dlg.show()
        QApplication.processEvents()
        return dlg

    def _batch_progress_update(
        self,
        dlg,
        completed_steps,
        total_steps,
        file_path,
        stage_text,
        log_message=None,
    ):
        if dlg is None:
            return
        total_steps = max(1, int(total_steps))
        completed_steps = max(0, min(int(completed_steps), total_steps))
        dlg.progress_bar.setRange(0, total_steps)
        dlg.progress_bar.setValue(completed_steps)
        dlg.file_label.setText(f"File: {Path(str(file_path)).name if file_path else '-'}")
        dlg.stage_label.setText(f"Status: {stage_text}")
        if log_message:
            dlg.log_box.appendPlainText(str(log_message))
        QApplication.processEvents()

    def _toggle_ipcluster(self):
        if getattr(self, "_ipcluster_running", False):
            self._stop_ipcluster()
            return
        self._start_ipcluster_dialog()

    def _start_ipcluster_dialog(self):
        import shutil
        import subprocess

        if shutil.which("ipcluster") is None:
            QMessageBox.warning(self, "ipcluster", "Could not find 'ipcluster' in PATH.")
            return

        default_cores, max_cores = self._default_ipcluster_cores()
        dlg = QDialog(self)
        dlg.setWindowTitle("Start ipcluster")
        layout = QVBoxLayout(dlg)

        grid = QGridLayout()
        n_cores = QSpinBox(dlg)
        n_cores.setRange(1, max_cores)
        n_cores.setValue(default_cores)
        grid.addWidget(QLabel("Number of engines"), 0, 0)
        grid.addWidget(n_cores, 0, 1)
        layout.addLayout(grid)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK", dlg)
        cancel_btn = QPushButton("Cancel", dlg)
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        if dlg.exec() != QDialog.Accepted:
            return

        cmd = ["ipcluster", "start", "-n", str(int(n_cores.value()))]
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._set_ipcluster_button_state(True)
            QMessageBox.information(
                self,
                "ipcluster",
                f"Started ipcluster with {int(n_cores.value())} engines in background.",
            )
        except Exception as err:
            QMessageBox.warning(self, "ipcluster", f"Could not start ipcluster: {err}")

    def _stop_ipcluster(self):
        import shutil
        import subprocess

        if shutil.which("ipcluster") is None:
            QMessageBox.warning(self, "ipcluster", "Could not find 'ipcluster' in PATH.")
            return
        try:
            subprocess.Popen(
                ["ipcluster", "stop"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._set_ipcluster_button_state(False)
            QMessageBox.information(self, "ipcluster", "Requested ipcluster stop.")
        except Exception as err:
            QMessageBox.warning(self, "ipcluster", f"Could not stop ipcluster: {err}")

    def _update_time_bounds_for_selected_layer(self):
        from .helpers_axes import infer_axes

        if not hasattr(self, "min_timer") or not hasattr(self, "max_timer"):
            return

        prev_max_bound = int(self.max_timer.maximum())
        prev_max_value = int(self.max_timer.value())
        prev_min_value = int(self.min_timer.value())

        try:
            index_layer = _get_choice_layer(self, self.layersbox)
            layer = self.viewer.layers[index_layer]
            axes = infer_axes(layer)
            if axes.t is not None:
                max_t = max(0, int(layer.data.shape[axes.t]) - 1)
            else:
                max_t = 0
        except Exception:
            max_t = 0

        self.min_timer.setRange(0, max_t)
        self.max_timer.setRange(0, max_t)
        self.min_timer.setSingleStep(1)
        self.max_timer.setSingleStep(1)

        # Keep user selections when possible, but initialize/expand max to last frame.
        if prev_min_value > max_t:
            self.min_timer.setValue(0)
        else:
            self.min_timer.setValue(prev_min_value)

        if prev_max_bound == 0 and prev_max_value == 0:
            # First-time initialization for this widget.
            self.max_timer.setValue(max_t)
        elif prev_max_value == prev_max_bound and max_t > prev_max_bound:
            # If user was at previous end, keep them at new end.
            self.max_timer.setValue(max_t)
        elif prev_max_value > max_t:
            self.max_timer.setValue(max_t)
        else:
            self.max_timer.setValue(prev_max_value)
        
            
    def __init__(self, napari_viewer):
    
        super().__init__()
        self.viewer = napari_viewer
        
        self.points_options = dict(face_color=[0]*4,opacity=0.75,size=10,blending='additive',border_width=0.15)
        # edge_color='red'
        self.points_options2 = dict(face_color=[0]*4,opacity=0.75,size=10,blending='additive',border_width=0.15)
        # ,edge_color='green'
        #comboBox for layer selection
        self.llayer = QLabel()
        self.llayer.setText("Layer to detect")
        self.layersbox = QComboBox()
        self.layersbox.currentIndexChanged.connect(self._select_layer)
        
        _populate_layers(self,self.layersbox,"image")

        l1 = QLabel()
        l1.setText("Mass Threshold")
        self.mass_slider = QSpinBox()
        self.mass_slider.setRange(0, int(1e6))
        self.mass_slider.setSingleStep(200)
        self.mass_slider.setValue(0)
        l2 = QLabel()
        
        # l2.setText("Diameter of the particle")
        # self.diameter_input = QSpinBox()
        # self.diameter_input.setRange(1, 19)
        # self.diameter_input.setSingleStep(2)
        # self.diameter_input.setValue(5)
        
        l2.setText("Diameter (int or tuple)")
        self.diameter_input = QLineEdit()
        self.diameter_input.setPlaceholderText("5    or    3,5,5")
        self.diameter_input.setText("3,5,5")        
        
        l3 = QLabel()
        l3.setText("Size filter (optional min/max)")
        self.layoutH0 = QHBoxLayout()
        self.size_min_tick = QCheckBox(">= ")
        self.size_min_tick.setChecked(False)
        self.size_min_input = QDoubleSpinBox()
        self.size_min_input.setRange(0, 10)
        self.size_min_input.setSingleStep(0.05)
        self.size_min_input.setValue(0.00)
        self.size_max_tick = QCheckBox("<= ")
        self.size_max_tick.setChecked(False)
        self.size_max_input = QDoubleSpinBox()
        self.size_max_input.setRange(0, 10)
        self.size_max_input.setSingleStep(0.05)
        self.size_max_input.setValue(1.60)
        self.layoutH0.addWidget(self.size_min_tick)
        self.layoutH0.addWidget(self.size_min_input)
        self.layoutH0.addWidget(self.size_max_tick)
        self.layoutH0.addWidget(self.size_max_input)

        l4 = QLabel()
        l4.setText("Eccentricity filter (optional min/max)")
        self.layoutH0p = QHBoxLayout()
        self.ecc_min_tick = QCheckBox(">= ")
        self.ecc_min_tick.setChecked(False)
        self.ecc_min_input = QDoubleSpinBox()
        self.ecc_min_input.setRange(0, 2)
        self.ecc_min_input.setSingleStep(0.05)
        self.ecc_min_input.setValue(0.00)
        self.ecc_max_tick = QCheckBox("<= ")
        self.ecc_max_tick.setChecked(False)
        self.ecc_max_input = QDoubleSpinBox()
        self.ecc_max_input.setRange(0, 2)
        self.ecc_max_input.setSingleStep(0.05)
        self.ecc_max_input.setValue(0.35)
        self.layoutH0p.addWidget(self.ecc_min_tick)
        self.layoutH0p.addWidget(self.ecc_min_input)
        self.layoutH0p.addWidget(self.ecc_max_tick)
        self.layoutH0p.addWidget(self.ecc_max_input)


        self.layoutH1 = QHBoxLayout()
        l_choice = QLabel()
        l_choice.setText("Check for All Frames identification")
        self.choice = QCheckBox()
        self.layoutH1.addWidget(l_choice)
        self.layoutH1.addWidget(self.choice)

        self.layoutH2 = QHBoxLayout()
        l_min_time = QLabel("Min Time")
        # l_min_time.setText("Mass Threshold")
        l_max_time = QLabel("Max Time")
        self.min_timer = QSpinBox()
        self.min_timer.setRange(0, 0)
        self.min_timer.setSingleStep(1)
        self.min_timer.setValue(0)
        self.max_timer = QSpinBox()
        self.max_timer.setRange(0, 0)
        self.max_timer.setSingleStep(1)
        self.max_timer.setValue(0)
        self.layoutH2.addWidget(l_min_time)
        self.layoutH2.addWidget(self.min_timer)
        self.layoutH2.addWidget(l_max_time)
        self.layoutH2.addWidget(self.max_timer)

        self.ipcluster_btn = QPushButton("Start ipcluster")
        self.ipcluster_btn.clicked.connect(self._toggle_ipcluster)
        self._set_ipcluster_button_state(False)

        label_masks = QLabel()
        label_masks.setText("Make Masks?")
        self.layout_masks = QHBoxLayout()
        self.make_masks_box = QCheckBox()
        self.make_masks_box.setChecked(False)
        self.masks_option = QComboBox()
        self._mask_algo_options = available_mask_algorithms()
        self.masks_option.addItems([label for _, label in self._mask_algo_options])
        self.masks_dict = {i: key for i, (key, _) in enumerate(self._mask_algo_options)}
        if self._mask_algo_options:
            numba_idx = next(
                (i for i, (key, _) in enumerate(self._mask_algo_options) if key == "numba"),
                0,
            )
            self.masks_option.setCurrentIndex(numba_idx)
        self.intensity_option = QComboBox()
        self._intensity_backend_options = available_intensity_backends()
        self.intensity_option.addItems([label for _, label in self._intensity_backend_options])
        self.intensity_dict = {i: key for i, (key, _) in enumerate(self._intensity_backend_options)}
        self.layout_masks.addWidget(label_masks)
        self.layout_masks.addWidget(self.make_masks_box)
        self.layout_masks.addWidget(QLabel("Mask algo"))
        self.layout_masks.addWidget(self.masks_option)
        self.layout_masks.addWidget(QLabel("Intensity"))
        self.layout_masks.addWidget(self.intensity_option)

        btn = QPushButton("Identify Spots")
        btn.clicked.connect(self._on_click)
        self.btn2 = QPushButton("Filter with new settings from already identified")
        self.btn2.setEnabled(False)
        self.btn2.clicked.connect(self._on_click2)

        layoutSpacer = QSpacerItem(5, 5, QSizePolicy.Minimum, QSizePolicy.Expanding) 
        # self.save_spots = QFileDialog()
        save_btn = QPushButton("Save current Spots")
        save_btn.clicked.connect(self._save_results)
        # line0 = QSplitter()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.llayer)
        self.layout().addWidget(self.layersbox)
        self.layout().addWidget(l1)
        self.layout().addWidget(self.mass_slider )
        self.layout().addWidget(l2)
        self.layout().addWidget(self.diameter_input)
        self.layout().addWidget(l3)
        self.layout().addLayout(self.layoutH0)
        self.layout().addWidget(l4)
        self.layout().addLayout(self.layoutH0p)
        self.layout().addLayout(self.layoutH1)
        self.layout().addLayout(self.layoutH2)
        self.layout().addWidget(self.ipcluster_btn)
        self.layout().addLayout(self.layout_masks)
        self.layout().addWidget(btn)
        self.layout().addWidget(self.btn2)
        # self.layout.addSpacing(10)
        self.layout().addSpacerItem(layoutSpacer)

        file_browse = QPushButton('Browse')
        file_browse.clicked.connect(self.open_file_dialog)
        # file_browse.clicked.connect(open_file_dialog,self)
        self.filename_edit = QLineEdit()
        # try:
        #     self.filename_edit.setText(_get_open_filename(self)+"_Spots.csv")
        # except:
        self.filename_edit.setText("Spots.csv")
        self.auto_save = QCheckBox()
        self.auto_save.setChecked(True)
        grid_layout = QGridLayout()
        grid_layout.addWidget(QLabel('File:'), 0, 0)
        grid_layout.addWidget(self.filename_edit, 0, 1)
        grid_layout.addWidget(file_browse, 0 ,2)
        grid_layout.addWidget(self.auto_save, 0 ,3)
        self.layout().addLayout(grid_layout)
        self.layout().addWidget(save_btn)

        

        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.reordered.connect(self._refresh_layers)

        # self._connect_layer()
        ###Run Batch Idea

        ### for every channel create a 
        # for 
        self.batch_grid_layout = QGridLayout()
        for k in range(self.layersbox.count()):
            
            batch_mass_slider = QSpinBox()
            batch_mass_slider.setRange(0, int(1e6))
            batch_mass_slider.setSingleStep(200)
            batch_mass_slider.setValue(self.mass_slider.value())
            
            batch_check_box = QCheckBox()
            batch_check_box.setChecked(True)

            self.batch_grid_layout.addWidget(QLabel(self.layersbox.itemText(k)), k, 0)
            self.batch_grid_layout.addWidget(batch_mass_slider, k, 1)
            self.batch_grid_layout.addWidget(batch_check_box, k ,2)

        # QDoubleSpinBox()
        # QCheckBox()
        #Then for all the checked items
            #update mass to match the layer to analyse
            #Create a QPushButton
        _batch_btn = QPushButton('Run in All Channels Selected')
        _batch_btn.clicked.connect(self.batch_on_click)

        self.layout().addLayout(self.batch_grid_layout)
        self.layout().addWidget(_batch_btn)

        # NEW — batch across multiple files
        self.batch_files_run_coloc = QCheckBox("Also run colocalization after identify")
        self.batch_files_run_coloc.setChecked(False)
        self.layout().addWidget(self.batch_files_run_coloc)

        self._batch_coloc_cfg = {
            "distance": 5.0,
            "anchor_suffix": None,
            "question_suffix": None,
            "run_all_to_anchor": False,
        }

        _batch_files_btn = QPushButton('Batch Identify Files')
        _batch_files_btn.clicked.connect(self._batch_files)
        self.layout().addWidget(_batch_files_btn)
        self._load_identify_settings()
        self._update_time_bounds_for_selected_layer()


    def open_file_dialog(self):
        from pathlib import Path
        filename, ok = QFileDialog.getSaveFileName(
            self,
            "Select a File", 
            #modify for last folder used
            "/tmp/", 
            "Comma Separated Files (*.csv)"
        )
        if filename:
            path = Path(filename)
            self.filename_edit.setText(str(path))
            return str(path)
        else:
            return "/tmp/Null.csv"
    
    # def _refresh_layers(self):
    #     current = self.layersbox.currentIndex()
    #     self.layersbox.clear()
    #     for layer in self.viewer.layers:
    #         if layer._type_string == 'image':
    #             self.layersbox.addItem(layer.name)
    #     self.layersbox.setCurrentIndex(current)


    def _refresh_layers(self):
        if getattr(self, "_batch_running", False):
            return
        # update the combobox first
        current = self.layersbox.currentText()
        self.layersbox.blockSignals(True)        # avoid re‑entrancy
        self.layersbox.clear()
        for layer in self.viewer.layers:
            if layer._type_string == "image":
                self.layersbox.addItem(layer.name)
        if current:
            self.layersbox.setCurrentText(current)
        self.layersbox.blockSignals(False)

        # now rebuild the per‑channel grid
        self._rebuild_batch_grid()
        if getattr(self, "_batch_files_running", False):
            frozen_cfg = getattr(self, "_batch_files_frozen_cfg", None)
            if frozen_cfg is not None:
                self._batch_cfg = dict(frozen_cfg)
                self._apply_batch_config()
        else:
            self._load_identify_settings()
        self._update_time_bounds_for_selected_layer()
        
    # def _refresh_layers(self,_widget,_type='image'):
    #     i = _widget.currentIndex()
    #     _widget.clear()
    #     for layer in self.viewer.layers:
    #         if layer._type_string == 'image':
    #             _widget.addItem(layer.name)
    #     _widget.setCurrentIndex(i)

    # def _connect_layer(self):
    #     self.viewer.layers.events.changed.connect(self._populate_layers)


    def _select_layer(self,i):
        
        # for j,layer in enumerate(self.viewer.layers):
        #     if layer.name == self.layersbox.value():
        #         index_layer = j
        #         break
        # return index_layer
        # # self.layersbox.currentIndex()
        ##should change it to name because it ignores non image layers
        print("Layer to detect:", i, self.layersbox.currentIndex())
        if getattr(self, "_batch_running", False):
            return
        try:
            index_layer = _get_choice_layer(self, self.layersbox)
            self._load_identify_settings(index_layer=index_layer)
        except Exception:
            pass
        self._update_time_bounds_for_selected_layer()
        # self.llayer.setText(
        # return j
        
    
    def _selected_mask_algorithm(self):
        if not self.masks_dict:
            return "numba"
        idx = self.masks_option.currentIndex()
        return self.masks_dict.get(idx, next(iter(self.masks_dict.values())))

    def _selected_intensity_backend(self):
        if not self.intensity_dict:
            return "cpu"
        idx = self.intensity_option.currentIndex()
        return self.intensity_dict.get(idx, next(iter(self.intensity_dict.values())))

    def _image_slice_for_frame(self, layer, frame_idx):
        from .helpers_axes import infer_axes

        axes = infer_axes(layer)
        if axes.t is None:
            return np.asarray(layer.data), axes

        t_max = max(0, int(layer.data.shape[axes.t]) - 1)
        frame_idx = max(0, min(int(frame_idx), t_max))
        sl = [slice(None)] * layer.data.ndim
        sl[axes.t] = frame_idx
        return np.asarray(layer.data[tuple(sl)]), axes

    def make_masks(self, points_layer=None):
        import gc
        import pandas as pd
        from .helpers_axes import infer_axes

        if points_layer is None:
            points_layer = self.viewer.layers.selection.active
        if points_layer is None or points_layer._type_string != "points":
            raise ValueError("make_masks expects a points layer.")

        index_layer = _get_choice_layer(self, self.layersbox)
        image_layer = self.viewer.layers[index_layer]
        image_axes = infer_axes(image_layer)

        points_data = np.asarray(points_layer.data)
        if points_data.ndim != 2 or points_data.shape[1] < 3:
            return np.zeros(image_layer.data.shape, dtype=np.uint32), pd.DataFrame(index=np.arange(len(points_data)))

        if points_data.shape[1] <= 3:
            df = pd.DataFrame(points_data, columns=["frame", "y", "x"])
            coord_cols = ["frame", "y", "x"]
        else:
            df = pd.DataFrame(points_data, columns=["frame", "z", "y", "x"])
            coord_cols = ["frame", "z", "y", "x"]
        _append_layer_properties(df, getattr(points_layer, "properties", {}) or {})

        masks = np.zeros(image_layer.data.shape, dtype=np.uint32)
        intensity_df = pd.DataFrame(index=df.index)
        algo = self._selected_mask_algorithm()
        backend = self._selected_intensity_backend()
        _diam = self._diameter_scalar()
        _radius = (_diam / 2.0) - 0.5

        ratio = 1.0
        try:
            if image_axes.z is not None and image_axes.y is not None:
                z_scale = float(image_layer.scale[image_axes.z])
                y_scale = float(image_layer.scale[image_axes.y])
                if z_scale > 0:
                    ratio = max(1e-6, y_scale / z_scale)
        except Exception:
            ratio = 1.0

        frame_series = pd.to_numeric(df["frame"], errors="coerce")
        valid_coord_mask = np.isfinite(frame_series.to_numpy())
        for col in coord_cols[1:]:
            valid_coord_mask &= np.isfinite(pd.to_numeric(df[col], errors="coerce").to_numpy())
        df["frame"] = frame_series.fillna(0).round().astype(int)

        if image_axes.t is None:
            frame_values = [0]
        else:
            frame_values = sorted(df.loc[valid_coord_mask, "frame"].unique().tolist())
            if not frame_values:
                frame_values = [0]

        image_layers = [layer for layer in self.viewer.layers if layer._type_string == "image"]

        for frame_counter, frame_idx in enumerate(frame_values, start=1):
            frame_img = None
            mask_temp = None
            mask_fixed = None
            temp = None
            try:
                temp = df[(df["frame"] == int(frame_idx)) & valid_coord_mask]
                if temp.empty:
                    continue

                frame_img, _ = self._image_slice_for_frame(image_layer, frame_idx)
                if frame_img.ndim not in (2, 3):
                    continue

                print(
                    "Doing Masks with option:",
                    self.masks_option.currentIndex(),
                    algo,
                    "for frame",
                    int(frame_idx),
                )
                mask_temp, _ = make_labels_trackpy_links(
                    frame_img.shape,
                    temp,
                    radius=_radius,
                    ratio=ratio,
                    _algo=algo,
                )
                mask_fixed = np.asarray(mask_temp, dtype=np.uint32)

                if image_axes.t is None:
                    masks = mask_fixed
                else:
                    sl = [slice(None)] * masks.ndim
                    t_max = max(0, int(masks.shape[image_axes.t]) - 1)
                    t_idx = max(0, min(int(frame_idx), t_max))
                    sl[image_axes.t] = t_idx
                    masks[tuple(sl)] = mask_fixed

                row_index = temp.index.to_numpy(dtype=np.int64)
                for img_layer in image_layers:
                    try:
                        intensity_image, _ = self._image_slice_for_frame(img_layer, frame_idx)
                    except Exception:
                        continue
                    if intensity_image.shape != mask_fixed.shape:
                        continue
                    try:
                        meas = measure_label_intensity(mask_fixed, intensity_image, backend=backend)
                    except Exception as err:
                        print(f"Could not measure intensities on {img_layer.name}: {err}")
                        continue
                    if meas is None or len(meas) == 0:
                        continue

                    suffix = img_layer.name.split("::")[-1].strip() or img_layer.name.strip()
                    for stat_col in ("intensity_mean", "intensity_max", "intensity_min"):
                        if stat_col not in meas.columns:
                            continue
                        out_col = f"{stat_col}::{suffix}"
                        values = intensity_df.get(out_col)
                        if values is None:
                            intensity_df[out_col] = np.nan
                        for _, row in meas.iterrows():
                            lbl = int(row.get("label", 0))
                            if lbl <= 0 or lbl > len(row_index):
                                continue
                            point_idx = int(row_index[lbl - 1])
                            intensity_df.at[point_idx, out_col] = float(row[stat_col])
            finally:
                del frame_img
                del mask_temp
                del mask_fixed
                del temp
                if algo in ("gpu", "opencl"):
                    release_backend_memory(algo)
                if backend == "gpu":
                    release_backend_memory("gpu")
                if frame_counter % 5 == 0:
                    gc.collect()

        return masks, intensity_df
    
    # @thread_worker
    def _on_click(self, minmass_override=None):
        from .helpers_axes import infer_axes
        import trackpy as tp
        import pandas as pd
        
        index_layer = _get_choice_layer(self,self.layersbox)
        self.viewer.layers[index_layer].name
        ##this only works for napari-bioformats
        # if self.filename_edit.text() == "Spots.csv":
        # name_points = self.viewer.layers[index_layer].name.split(":")[0]
        name_points = self.viewer.layers[index_layer].name.split("::")[-1]
        self.filename_edit.setText(_get_open_filename(
                self)+'_'+name_points+"_Spots.csv")
        # else:
        #     temp_path = self.filename_edit.txt()
        #     self.filename_edit.setText(temp_path.strip_get_open_filename(
        #             self)+'_'+self.viewer.layers[index_layer].name+"_Spots_.csv")

        layer = self.viewer.layers[index_layer]
        axes  = infer_axes(layer)         # Axes(order='tzyx', t=0, z=1, y=2, x=3)
        diam    = self._parse_diameter()
        if minmass_override is not None:
            self.mass_slider.setValue(int(minmass_override))
            minmass = int(minmass_override)
        else:
            minmass = int(self.mass_slider.value())
        self._save_identify_settings(index_layer=index_layer)
        print(axes)
        print("Axis order detected:", axes.order)
                
        # ------------------------------------------------------------------
        # Helpers to slice data
        # ------------------------------------------------------------------
        def _plane2d(t_idx=0, z_idx=0):
            """Return a 2‑D YX plane."""
            sl = [slice(None)] * layer.data.ndim
            if axes.t is not None:
                sl[axes.t] = t_idx
            if axes.z is not None:
                sl[axes.z] = z_idx
            return layer.data[tuple(sl)]

        def _stack3d(t_idx=0):
            """Return a full Z‑stack (ZYX) at a given time index."""
            sl = [slice(None)] * layer.data.ndim
            if axes.t is not None:
                sl[axes.t] = t_idx
            # leave Z slice(None) → full stack
            return layer.data[tuple(sl)]

        # ------------------------------------------------------------------
        # A)  TIME‑LAPSE  — run on **all** frames if the user ticked the box
        # ------------------------------------------------------------------
        if axes.t is not None and self.choice.isChecked():
            print(f"Time‑series detected ({axes.order}); running across frames")
            t0, t1 = self.min_timer.value(), self.max_timer.value()
            max_t = max(0, int(layer.data.shape[axes.t]) - 1)
            t0 = max(0, min(int(t0), max_t))
            t1 = max(0, min(int(t1), max_t))
            if t1 < t0:
                t0, t1 = t1, t0
            # Inclusive range to ensure the last selected frame is processed.
            t_range = range(t0, t1 + 1)

            def _loc_worker(frame_idx, img):
                import trackpy as tp

                try:
                    img = img.compute()      # dask → numpy if needed
                except AttributeError:
                    pass
                df = tp.locate(img, diam, minmass=minmass, engine="numba")
                df["frame"] = frame_idx
                return df

            results = None
            try:
                from ipyparallel import Client

                rc = Client()
                if len(rc) == 0:
                    raise RuntimeError("No ipyparallel engines available.")
                v = rc.load_balanced_view()
                print("Using", len(rc), "ipyparallel engines")
                max_pending = max(2, int(len(rc) * 2))
                pending = []
                results = []
                for frame_idx in t_range:
                    img = _plane2d(t_idx=frame_idx) if axes.z is None else _stack3d(t_idx=frame_idx)
                    pending.append(v.apply_async(_loc_worker, frame_idx, img))
                    if len(pending) >= max_pending:
                        results.append(pending.pop(0).get())
                while pending:
                    results.append(pending.pop(0).get())
            except Exception as err:
                print(f"ipyparallel unavailable ({err}); falling back to local loop.")
                results = []
                for frame_idx in t_range:
                    img = _plane2d(t_idx=frame_idx) if axes.z is None else _stack3d(t_idx=frame_idx)
                    results.append(_loc_worker(frame_idx, img))

            self.f = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

            # ------------------------------------------------------------------
            # ## Alternative multiprocessing.Pool method (commented, ready)
            # ------------------------------------------------------------------
            # from multiprocessing import Pool
            # with Pool() as pool:
            #     results = pool.starmap(
            #         _loc_worker, [(i, img) for i, img in zip(t_range, image_seq)]
            #     )
            # self.f = pd.concat(results, ignore_index=True)
            # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # B)  SINGLE time‑point when data have a T axis  (TYX or TZYX)
        # ------------------------------------------------------------------
        elif axes.t is not None:
            current_t = self.viewer.dims.current_step[axes.t]
            print(f"Single time‑point (t={current_t}) detected")

            if axes.z is None:                       # TYX  → 2‑D frame
                img = _plane2d(t_idx=current_t)
            else:                                    # TZYX → full 3‑D stack
                img = _stack3d(t_idx=current_t)
            try:
                img = img.compute()
            except AttributeError:
                pass

            self.f = tp.locate(img, diam, minmass=minmass, engine="numba")
            self.f["frame"] = current_t

        # ------------------------------------------------------------------
        # C)  One 3‑D Z‑stack (no time)   ZYX
        # ------------------------------------------------------------------
        elif axes.z is not None:
            print("3‑D z‑stack detected (ZYX)")
            img3d = np.asarray(layer.data)
            self.f = tp.locate(img3d, diam, minmass=minmass, engine="numba")
            self.f["frame"] = 0

        # ------------------------------------------------------------------
        # D)  Plain 2‑D image (YX)
        # ------------------------------------------------------------------
        else:
            print("Single 2‑D image detected (YX)")
            img2d = np.asarray(layer.data)
            self.f = tp.locate(img2d, diam, minmass=minmass, engine="numba")
            self.f["frame"] = 0
        
        # filtering steps
        if len(self.viewer.layers[index_layer].data.shape) <= 3 and "ecc" in self.f.columns:
            self.f = self._apply_bounds_filter(
                self.f,
                "ecc",
                self.ecc_min_tick.isChecked(),
                self.ecc_min_input.value(),
                self.ecc_max_tick.isChecked(),
                self.ecc_max_input.value(),
            )
        if "size" in self.f.columns:
            self.f = self._apply_bounds_filter(
                self.f,
                "size",
                self.size_min_tick.isChecked(),
                self.size_min_input.value(),
                self.size_max_tick.isChecked(),
                self.size_max_input.value(),
            )
        else:
            axes = [c for c in ("size_x", "size_y", "size_z") if c in self.f.columns]
            self.f = self._apply_bounds_filter(
                self.f,
                axes,
                self.size_min_tick.isChecked(),
                self.size_min_input.value(),
                self.size_max_tick.isChecked(),
                self.size_max_input.value(),
            )
                
        #transforming data to pandas ready for spots
        if len(self.viewer.layers[index_layer].data.shape) <= 3:
            #XYZ
            if len(self.viewer.layers[index_layer].data.shape) == 2:
                _points = self.f.loc[:,['frame','y','x']]
            elif self.viewer.layers[index_layer].scale[0] != 1:
                _points = self.f.loc[:,['frame','z','y','x']]
            #TYX PRJ
            else:    
                _points = self.f.loc[:,['frame','y','x']]
        #TZYX
        elif len(self.viewer.layers[index_layer].data.shape) > 3:
            _points = self.f.loc[:,['frame','z','y','x']]
            
        _metadata = self._trackpy_properties(self.f, _points.columns)
        #self.viewer.layers[index_layer]
        # self.viewer.layers[index_layer].colormap.name)
        #like this is opposite color of the image
        #make if smarter self.viewer.layers[index_layer].colormap.color has an array with the colors, we should be able to flip universaly 
        clr_name = self.viewer.layers[index_layer].colormap.name
        clr_dict = {'lime':'red',
                    # 'green':'magenta',
                    'green':'red',
                    'red' : 'lime',
                    # 'red':'cyan',
                    'blue':'yellow',
                    'magenta':'green',
                    'cyan':'red',
                    'yellow':'blue'}
        # if clr_name == 'green':
        #     point_colors = 'magenta'
        # elif clr_name == 'red':
        #     point_colors = 'cyan'
        # elif clr_name == 'blue':
        #     point_colors = 'yellow'
        point_colors = clr_dict.get(clr_name, "white")
        if len(_points) > 0:
            points_options = dict(self.points_options)
            points_options["size"] = self._points_size_from_detection()
            self._points_layer = _add_points_compat(
                self.viewer,
                _points,
                name="Points "+name_points,
                properties=_metadata,
                edge_color=point_colors,
                **points_options,
            )
            self._points_layer.scale = self.viewer.layers[index_layer].scale

            self.btn2.setEnabled(True)

            _masks = None
            if self.make_masks_box.isChecked():
                _masks, intensity_df = self.make_masks(points_layer=self._points_layer)
                if intensity_df is not None and len(intensity_df) == len(self._points_layer.data):
                    merged_props = dict(getattr(self._points_layer, "properties", {}) or {})
                    for col in intensity_df.columns:
                        merged_props[col] = intensity_df[col].to_numpy()
                    self._points_layer.properties = merged_props

            if self.auto_save.isChecked():
                self._save_results(layer=self._points_layer)

            if _masks is not None:
                self._masks_layer = self.viewer.add_labels(_masks)
                self._masks_layer.scale = self.viewer.layers[index_layer].scale

    def _batch_files(self):
        """Open selected files with aicsimageio and run batch_on_click on each."""
        # 1) ask for one or more filenames
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select image files",
            "",
            "Bio‑Formats images (*.nd2 *.czi *.ome.tif *.ome.tiff *.tif *.tiff);;"
            "All files (*)",
        )
        if not files:
            return

        coloc_cfg = None
        coloc_widget = None
        if self.batch_files_run_coloc.isChecked():
            coloc_cfg = self._configure_batch_colocalization()
            if coloc_cfg is None:
                print("Batch colocalization canceled.")
                return
            coloc_widget = ColocalizationQWidget(self.viewer)
            coloc_widget.auto_save.setChecked(True)

        # Progress window setup
        batch_channels = self._collect_batch_channel_settings()
        enabled_channels = sum(1 for v in batch_channels.values() if bool(v.get("enabled", False)))
        identify_units = max(1, enabled_channels)
        steps_per_file = 1 + identify_units + (1 if coloc_widget is not None else 0)
        total_steps = max(1, len(files) * steps_per_file)
        progress_dlg = self._make_batch_progress_dialog(total_steps)
        completed_steps = 0

        # Freeze the current batch-grid values for this multi-file run.
        self._capture_batch_config()
        self._batch_files_frozen_cfg = dict(getattr(self, "_batch_cfg", {}))
        self._batch_files_running = True
        try:
            # 2) loop over files
            for fname in files:
                print(f"\n=== Processing {fname} ===")
                self._batch_progress_update(
                    progress_dlg,
                    completed_steps,
                    total_steps,
                    fname,
                    "Opening file",
                    f"Opening {fname}",
                )
                try:
                    # Clear existing layers to avoid name clashes (optional; comment out
                    # if you prefer to keep everything in the viewer)
                    self.viewer.layers.clear()

                    # Load with the napari‑aicsimageio plugin
                    self.viewer.open(fname, plugin="napari-aicsimageio")

                    # Re-apply frozen per-channel config after loading.
                    self._batch_cfg = dict(self._batch_files_frozen_cfg)
                    self._apply_batch_config()
                except Exception as err:
                    print(f"Could not open {fname}: {err}")
                    completed_steps += 1
                    self._batch_progress_update(
                        progress_dlg,
                        completed_steps,
                        total_steps,
                        fname,
                        "Open failed",
                        f"Open failed: {err}",
                    )
                    continue
                completed_steps += 1
                self._batch_progress_update(
                    progress_dlg,
                    completed_steps,
                    total_steps,
                    fname,
                    "Open complete",
                    "Open complete",
                )

                # 3) run channel batch identification exactly as usual
                try:
                    if enabled_channels > 0:
                        def _identify_cb(ch_idx, ch_total, ch_name):
                            stage = f"Identify channel {ch_idx}/{ch_total}: {ch_name}"
                            self._batch_progress_update(
                                progress_dlg,
                                completed_steps + max(0, ch_idx - 1),
                                total_steps,
                                fname,
                                stage,
                                stage,
                            )

                        self.batch_on_click(progress_callback=_identify_cb)
                        completed_steps += identify_units
                    else:
                        self.batch_on_click()
                        completed_steps += identify_units
                    self._batch_progress_update(
                        progress_dlg,
                        completed_steps,
                        total_steps,
                        fname,
                        "Identify complete",
                        "Identify complete",
                    )
                except Exception as err:
                    print(f"Identify failed on {fname}: {err}")
                    completed_steps += identify_units
                    self._batch_progress_update(
                        progress_dlg,
                        completed_steps,
                        total_steps,
                        fname,
                        "Identify failed",
                        f"Identify failed: {err}",
                    )
                    continue
                if coloc_widget is not None:
                    try:
                        self._batch_progress_update(
                            progress_dlg,
                            completed_steps,
                            total_steps,
                            fname,
                            "Running colocalization",
                            "Running colocalization",
                        )
                        self._run_batch_colocalization(coloc_widget, coloc_cfg, fname)
                        completed_steps += 1
                        self._batch_progress_update(
                            progress_dlg,
                            completed_steps,
                            total_steps,
                            fname,
                            "Colocalization complete",
                            "Colocalization complete",
                        )
                    except Exception as err:
                        print(f"Colocalization failed on {fname}: {err}")
                        completed_steps += 1
                        self._batch_progress_update(
                            progress_dlg,
                            completed_steps,
                            total_steps,
                            fname,
                            "Colocalization failed",
                            f"Colocalization failed: {err}",
                        )
                        continue
        finally:
            self._batch_files_running = False
            self._batch_files_frozen_cfg = {}
            if coloc_widget is not None:
                coloc_widget.deleteLater()
            self._batch_progress_update(
                progress_dlg,
                total_steps,
                total_steps,
                "",
                "Done",
                "Batch processing finished.",
            )
            progress_dlg.close_btn.setEnabled(True)

    def _image_channel_suffixes(self):
        suffixes = []
        for i in range(self.layersbox.count()):
            suffix = self._batch_channel_key(self.layersbox.itemText(i))
            if suffix and suffix not in suffixes:
                suffixes.append(suffix)
        return suffixes

    def _configure_batch_colocalization(self):
        suffixes = self._image_channel_suffixes()
        if not suffixes:
            print("No image channels available to configure batch colocalization.")
            return None

        cfg_prev = {
            "distance": 5.0,
            "anchor_suffix": suffixes[0],
            "question_suffix": suffixes[min(1, len(suffixes) - 1)],
            "run_all_to_anchor": False,
            **(getattr(self, "_batch_coloc_cfg", {}) or {}),
        }

        dlg = QDialog(self)
        dlg.setWindowTitle("Batch Colocalization Settings")
        layout = QVBoxLayout(dlg)
        grid = QGridLayout()

        anchor_combo = QComboBox(dlg)
        anchor_combo.addItems(suffixes)
        if cfg_prev["anchor_suffix"] in suffixes:
            anchor_combo.setCurrentText(cfg_prev["anchor_suffix"])

        question_combo = QComboBox(dlg)
        question_combo.addItems(suffixes)
        if cfg_prev["question_suffix"] in suffixes:
            question_combo.setCurrentText(cfg_prev["question_suffix"])
        elif len(suffixes) > 1:
            question_combo.setCurrentIndex(1)

        distance = QDoubleSpinBox(dlg)
        distance.setRange(0, 20)
        distance.setSingleStep(0.2)
        distance.setValue(float(cfg_prev.get("distance", 5.0)))

        run_all = QCheckBox("Run all combinations to anchor", dlg)
        run_all.setChecked(bool(cfg_prev.get("run_all_to_anchor", False)))

        grid.addWidget(QLabel("Anchor channel"), 0, 0)
        grid.addWidget(anchor_combo, 0, 1)
        grid.addWidget(QLabel("Comparison channel"), 1, 0)
        grid.addWidget(question_combo, 1, 1)
        grid.addWidget(QLabel("Max Distance (sub-pixels)"), 2, 0)
        grid.addWidget(distance, 2, 1)
        grid.addWidget(run_all, 3, 0, 1, 2)
        layout.addLayout(grid)

        btn_row = QHBoxLayout()
        run_btn = QPushButton("Run", dlg)
        cancel_btn = QPushButton("Cancel", dlg)
        run_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(run_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        if dlg.exec() != QDialog.Accepted:
            return None

        cfg = {
            "distance": float(distance.value()),
            "anchor_suffix": anchor_combo.currentText(),
            "question_suffix": question_combo.currentText(),
            "run_all_to_anchor": bool(run_all.isChecked()),
        }

        if (not cfg["run_all_to_anchor"]) and cfg["anchor_suffix"] == cfg["question_suffix"]:
            for suffix in suffixes:
                if suffix != cfg["anchor_suffix"]:
                    cfg["question_suffix"] = suffix
                    break

        self._batch_coloc_cfg = dict(cfg)
        return cfg

    def _point_layer_suffix(self, point_layer_name: str) -> str:
        text = point_layer_name.strip()
        if text.lower().startswith("points"):
            text = text[len("points"):].strip()
        return self._batch_channel_key(text)

    def _point_layer_name_by_suffix(self, coloc_widget, suffix):
        for layer in coloc_widget._points_layers_for_coloc():
            if self._point_layer_suffix(layer.name) == suffix:
                return layer.name
        return None

    def _run_batch_colocalization(self, coloc_widget, cfg, source_file):
        coloc_widget._refresh_layers()
        coloc_widget.euc_distance.setValue(float(cfg["distance"]))
        coloc_widget._filename_user_set = False
        coloc_widget._set_default_output_filename(force=True)
        coloc_widget.auto_save.setChecked(True)

        anchor_name = self._point_layer_name_by_suffix(coloc_widget, cfg["anchor_suffix"])
        if not anchor_name:
            print(f"Colocalization skipped for {source_file}: anchor points not found for '{cfg['anchor_suffix']}'.")
            return
        coloc_widget.points_anchor.setCurrentText(anchor_name)

        if cfg.get("run_all_to_anchor", False):
            coloc_widget.calculate_all_colocalizing()
            return

        question_name = self._point_layer_name_by_suffix(coloc_widget, cfg["question_suffix"])
        if not question_name:
            print(
                f"Colocalization skipped for {source_file}: comparison points not found for "
                f"'{cfg['question_suffix']}'."
            )
            return
        coloc_widget.points_question.setCurrentText(question_name)
        coloc_widget.calculate_colocalizing()
            
    def _capture_batch_config(self):
        cfg = {}
        rows = self.batch_grid_layout.rowCount()
        for r in range(rows):
            w_label = self.batch_grid_layout.itemAtPosition(r, 0)
            w_spin  = self.batch_grid_layout.itemAtPosition(r, 1)
            w_tick  = self.batch_grid_layout.itemAtPosition(r, 2)
            if not (w_label and w_spin and w_tick):        # <-- guard
                continue
            name  = w_label.widget().text()
            mass  = w_spin.widget().value()
            check = w_tick.widget().isChecked()
            cfg[self._batch_channel_key(name)] = (mass, check)
        self._batch_cfg = cfg                  # new attribute

    def _rebuild_batch_grid(self):
        #save config
        self._capture_batch_config()
        # ---------- remove old widgets safely ---------------------------
        while self.batch_grid_layout.count():
            item = self.batch_grid_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()          # <- enough; no setParent(None)

        # ---------- add fresh widgets -----------------------------------
        for k in range(self.layersbox.count()):
            # **always give ‘self’ as parent** → they’re born inside the panel
            label = QLabel(self.layersbox.itemText(k), self)
            spin  = QSpinBox(self)
            tick  = QCheckBox(self)

            spin.setRange(0, int(1e6))
            spin.setSingleStep(200)
            spin.setValue(self.mass_slider.value())
            tick.setChecked(True)

            self.batch_grid_layout.addWidget(label, k, 0)
            self.batch_grid_layout.addWidget(spin,  k, 1)
            self.batch_grid_layout.addWidget(tick,  k, 2)

        self._apply_batch_config()        # restore saved masses / checks
        
    def _apply_batch_config(self):
        cfg  = getattr(self, "_batch_cfg", {})
        rows = self.batch_grid_layout.rowCount()
        for r in range(rows):
            w_label = self.batch_grid_layout.itemAtPosition(r, 0)
            if not w_label:
                continue                                   # <-- guard
            key = self._batch_channel_key(w_label.widget().text())
            if key not in cfg:
                continue
            mass, checked = cfg[key]
            w_spin = self.batch_grid_layout.itemAtPosition(r, 1)
            w_tick = self.batch_grid_layout.itemAtPosition(r, 2)
            if w_spin and w_tick:                          # <-- guard
                w_spin.widget().setValue(mass)
                w_tick.widget().setChecked(checked)

    def batch_on_click(self, progress_callback=None):
        print(self.batch_grid_layout.count())

        # Snapshot selections first so UI updates/signals cannot mutate values
        # mid-run.
        selected_rows = []
        for i in range(self.layersbox.count()):
            tick_item = self.batch_grid_layout.itemAtPosition(i, 2)
            mass_item = self.batch_grid_layout.itemAtPosition(i, 1)
            if not tick_item or not mass_item:
                continue
            if tick_item.widget().isChecked():
                selected_rows.append((i, int(mass_item.widget().value())))

        if not selected_rows:
            if progress_callback is not None:
                progress_callback(0, 0, "no channels selected")
            return

        self._batch_running = True
        try:
            total_rows = len(selected_rows)
            for idx, (row_idx, mass_value) in enumerate(selected_rows, start=1):
                channel_name = self.layersbox.itemText(row_idx)
                if progress_callback is not None:
                    progress_callback(idx, total_rows, channel_name)
                self.layersbox.setCurrentIndex(row_idx)
                self.mass_slider.setValue(mass_value)
                self._on_click(minmass_override=mass_value)
        finally:
            self._batch_running = False
            self._refresh_layers()
            


    # @thread_worker
    def _on_click2(self):
        index_layer = _get_choice_layer(self,self.layersbox)
        self._save_identify_settings(index_layer=index_layer)
        # print("napari has", len(self.viewer.layers), "layers")
        # f = tp.locate(self.viewer.layers[2].data,5,minmass=500)
        f2 = self.f
        # add the filters here if selected
        if "ecc" in f2.columns:
            f2 = self._apply_bounds_filter(
                f2,
                "ecc",
                self.ecc_min_tick.isChecked(),
                self.ecc_min_input.value(),
                self.ecc_max_tick.isChecked(),
                self.ecc_max_input.value(),
            )
        if "size" in f2.columns:
            f2 = self._apply_bounds_filter(
                f2,
                "size",
                self.size_min_tick.isChecked(),
                self.size_min_input.value(),
                self.size_max_tick.isChecked(),
                self.size_max_input.value(),
            )
        else:
            axes = [c for c in ("size_x", "size_y", "size_z") if c in f2.columns]
            f2 = self._apply_bounds_filter(
                f2,
                axes,
                self.size_min_tick.isChecked(),
                self.size_min_input.value(),
                self.size_max_tick.isChecked(),
                self.size_max_input.value(),
            )
        f2 = f2[(f2['mass'] > self.mass_slider.value())]

        
        if len(self.viewer.layers[index_layer].data.shape) < 3:
            _points = f2.loc[:,['frame','y','x']]
        elif len(self.viewer.layers[index_layer].data.shape) >= 3:
            if self.viewer.layers[index_layer].scale[0] != 1:
                _points = f2.loc[:,['frame','z','y','x']]
            else: 
                _points = f2.loc[:,['frame','y','x']]
        _metadata = self._trackpy_properties(f2, _points.columns)

        self.f2 = f2
        points_options2 = dict(self.points_options2)
        points_options2["size"] = self._points_size_from_detection()
        self._points_layer_filter = _add_points_compat(
            self.viewer,
            _points,
            properties=_metadata,
            **points_options2,
        )
        self._points_layer_filter.scale = self.viewer.layers[index_layer].scale

    def _save_results(self, layer=None):
        import pandas as pd

        layer = layer or self.viewer.layers.selection.active
        if layer is None or getattr(layer, "_type_string", "") != "points":
            print("No points layer selected to save.")
            return

        if layer.data.shape[1] < 3:
            #manbearpig time lapse vs Zstack
            df = pd.DataFrame(layer.data, columns = ['frame','y','x'])
        elif layer.data.shape[1] >= 3:
            df = pd.DataFrame(layer.data, columns = ['frame','z','y','x'])
            
        b = layer.properties
        _append_layer_properties(df, b)
        coord_cols = ['frame', 'z', 'y', 'x'] if 'z' in df.columns else ['frame', 'y', 'x']
        df = _drop_invalid_coordinate_rows(df, coord_cols)
        df.to_csv(self.filename_edit.text())

class LinkingQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        l5 = QLabel()
        l5.setText("Max Distance")
        self.distance = QSpinBox()
        self.distance.setRange(0,512)
        self.distance.setSingleStep(1)
        self.distance.setValue(20)
        self.layout_link_H0 = QHBoxLayout()
        self.layout_link_H0.addWidget(l5)
        self.layout_link_H0.addWidget(self.distance
                                )
        l6 = QLabel()
        l6.setText("Memory parameter")
        self.memory = QSpinBox()
        self.memory.setRange(0, 10)
        self.memory.setSingleStep(1)
        self.memory.setValue(3)
        self.layout_link_H1 = QHBoxLayout()
        self.layout_link_H1.addWidget(l6)
        self.layout_link_H1.addWidget(self.memory)


        l7 = QLabel()
        l7.setText("Filter Stubs")
        self.layout_link_H2 = QHBoxLayout()
        self.stubs_tick = QCheckBox()
        # self.stubs_tick.setChecked(True)
        self.stubs_input = QSpinBox()
        self.stubs_input.setRange(0, 200)
        self.stubs_input.setSingleStep(1)
        self.stubs_input.setValue(5)

        self.layout_link_H2.addWidget(l7)
        self.layout_link_H2.addWidget(self.stubs_tick)
        self.layout_link_H2.addWidget(self.stubs_input)         
        self.layout_link_H2.setEnabled(False)
        
        self.btn = QPushButton("Track everything")
        self.btn.clicked.connect(self._track)
        self.btn.setEnabled(False)
        self._enable_tracking()
        self.setLayout(QVBoxLayout())
        self.layout().addLayout(self.layout_link_H0)
        self.layout().addLayout(self.layout_link_H1)
        self.layout().addLayout(self.layout_link_H2)
        self.layout().addWidget(self.btn)

        self.viewer.layers.events.removed.connect(self._enable_tracking)
        self.viewer.layers.events.inserted.connect(self._enable_tracking)
        self.viewer.layers.events.reordered.connect(self._enable_tracking)
        self.viewer.layers.selection.events.changed.connect(self._enable_tracking)
        #Selecting
        # self.viewer.layers.events.selecting.connect(self._enable_tracking)

        file_browse = QPushButton('Browse')
        file_browse.clicked.connect(self.open_file_dialog)
        self.filename_edit_links = QLineEdit()
        if len(self.viewer.layers) > 0:
            self.filename_edit_links.setText(_get_open_filename(self)+"_Tracks.csv")
        grid_layout = QGridLayout()
        grid_layout.addWidget(QLabel('File:'), 0, 0)
        grid_layout.addWidget(self.filename_edit_links, 0, 1)
        grid_layout.addWidget(file_browse, 0 ,2)
        self.layout().addLayout(grid_layout)

        # self.save_spots = QFileDialog()
        save_btn = QPushButton("Save current Spots")
        save_btn.clicked.connect(self._save_results_links)
        self.layout().addWidget(save_btn)

    def open_file_dialog(self):
        from pathlib import Path
        filename, ok = QFileDialog.getSaveFileName(
            self,
            "Select a File", 
            "/tmp/", 
            "Comma Separated Files (*.csv)"
        )
        if filename:
            path = Path(filename)
            self.filename_edit_links.setText(str(path))
    
    def _save_results_links(self):
        import pandas as pd
        self.links.to_csv(self.filename_edit_links.text())

    def _sanitize_link_input(self, df):
        import pandas as pd

        required_cols = ["frame", "y", "x"]
        if "z" in df.columns:
            required_cols.insert(1, "z")

        out = df.copy()
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in required_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        before = len(out)
        out = out.dropna(subset=required_cols).copy()
        dropped = before - len(out)
        if dropped:
            print(f"Dropped {dropped} point rows with invalid frame/coordinate values before linking.")

        if len(out) == 0:
            return out

        out["frame"] = np.rint(out["frame"]).astype(np.int64)
        return out

    def _enable_tracking(self):
        self.btn.setEnabled(False)
        if len(self.viewer.layers.selection) == 1:
            if self.viewer.layers.selection.active._type_string == 'points':
                #if  self.viewer.layers.selection.active.data == more than one time:
                self.btn.setEnabled(True)

# def _get_choice_layer(self,_widget):    
#     for j,layer in enumerate(self.viewer.layers):
#         if layer.name == _widget.currentText():
#             index_layer = j
#             break
#     print("Layer where points are is:",j)
#     return index_layer


    def _track(self):
        import pandas as pd
        import trackpy as tp
        ##if 2d
        #this is from the other widget..dang
        # self.layersbox.currentIndex()
        #ALTERNATIVE:if there is a column named 'z'
        if len(self.viewer.layers[0].data.shape) <= 3:
            df = pd.DataFrame(self.viewer.layers.selection.active.data, columns = ['frame','y','x'])
        elif len(self.viewer.layers[0].data.shape) > 3:
            df = pd.DataFrame(self.viewer.layers.selection.active.data, columns = ['frame','z','y','x'])
        b = self.viewer.layers.selection.active.properties
        _append_layer_properties(df, b)
        coord_cols = ['frame', 'z', 'y', 'x'] if 'z' in df.columns else ['frame', 'y', 'x']
        df = _drop_invalid_coordinate_rows(df, coord_cols)

        df = self._sanitize_link_input(df)
        if len(df) == 0:
            print("No valid points left after removing NaN/inf rows. Linking aborted.")
            return

        print(df)
        pos_columns = ['z', 'y', 'x'] if 'z' in df.columns else ['y', 'x']
        links = tp.link(
            df,
            search_range=self.distance.value(),
            memory=self.memory.value(),
            pos_columns=pos_columns,
            t_column='frame',
        )
        if self.stubs_tick.isChecked():
            links = tp.filter_stubs(links, self.stubs_input.value())
        # if 2D:
        if 'z' in df:
            _tracks = links.loc[:,['particle','frame','z','y','x']]
        else:
            _tracks = links.loc[:,['particle','frame','y','x']]
        # if 3d:
        # _tracks = links.loc[:,['particle','frame','z','y','x']]

        _tracks = self.viewer.add_tracks(_tracks,name='trackpy')
        _tracks.scale = self.viewer.layers.selection.active.scale
        self.links = links
        print(links)



class ColocalizationQWidget(QWidget):

    # from support_libraries import _get_open_filename

    def __init__(self, napari_viewer):
        super().__init__()
        import pyqtgraph as pg
        self.viewer = napari_viewer

        l1 = QLabel('Points that are the "anchor"')
        # l1.setText()
        self.points_anchor = QComboBox()
        # self.points_anchor.currentIndexChanged.connect(self._select_layer)
        
        l2 = QLabel('Points that are the "comparison"')
        # l2.setText('Points that are the "comparison"')
        self.points_question = QComboBox()
        # self.points_question.currentIndexChanged.connect(self._select_layer)

        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.reordered.connect(self._refresh_layers)
        self.viewer.layers.events.changed.connect(self._refresh_layers)
        
        l4 = QLabel("Max Distance (in sub-pixels)")
        self.layoutH0 = QHBoxLayout()
        self.euc_distance = QDoubleSpinBox()
        self.euc_distance.setRange(0, 20)
        self.euc_distance.setSingleStep(0.2)
        self.euc_distance.setValue(5)
        self.layoutH0.addWidget(l4)
        self.layoutH0.addWidget(self.euc_distance)
        #make label asking the distance (in pixel)
        #QSpinBox for input
        ##another label for translation to uM
        run_btn = QPushButton("Run Colocalization")
        run_btn.clicked.connect(self.calculate_colocalizing)
        run_all_btn = QPushButton("Run All Combinations to Anchor")
        run_all_btn.clicked.connect(self.calculate_all_colocalizing)

        file_browse = QPushButton('Browse')
        file_browse.clicked.connect(self.open_file_dialog)
        self.filename_edit = QLineEdit()
        self._filename_user_set = False
        self.filename_edit.textEdited.connect(self._on_filename_edited)
        self.filename_edit.setText("Spots_Coloc.csv")
        grid_layout = QGridLayout()
        grid_layout.addWidget(QLabel('File:'), 0, 0)
        grid_layout.addWidget(self.filename_edit, 0, 1)
        grid_layout.addWidget(file_browse, 0 ,2)
        self.auto_save = QCheckBox()
        self.auto_save.setChecked(True)
        grid_layout.addWidget(self.auto_save, 0 ,3)


        save_btn = QPushButton("Save current Spots")
        save_btn.clicked.connect(self._save_results)

        # save_btn = QPushButton("Save current colocalized Spots")
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(l1)
        self.layout().addWidget(self.points_anchor)

        self.layout().addWidget(l2)
        self.layout().addWidget(self.points_question)
        self.layout().addLayout(self.layoutH0)
        self.layout().addWidget(run_btn)
        self.layout().addWidget(run_all_btn)
        self.layout().addLayout(grid_layout)
        self.layout().addWidget(save_btn)


        
        self._plt = pg.plot()
        self.layout().addWidget(self._plt)
        self._refresh_layers()
        self._set_default_output_filename()

    def _points_layers_for_coloc(self):
        from napari.layers import Points

        # Colocalization should work with any napari Points layer, regardless of name.
        return [layer for layer in self.viewer.layers if isinstance(layer, Points)]

    def _get_selected_points_layer(self, widget):
        layer_name = widget.currentText()
        for layer in self._points_layers_for_coloc():
            if layer.name == layer_name:
                return layer
        raise ValueError(f"No valid Points layer selected for '{layer_name}'")

    def _coloc_points_size(self) -> float:
        """Use the selected anchor points marker size for generated coloc points."""
        try:
            layer = self._get_selected_points_layer(self.points_anchor)
        except Exception:
            return 10.0

        current_size = getattr(layer, "current_size", None)
        if current_size is not None:
            try:
                return float(current_size)
            except Exception:
                pass

        size = getattr(layer, "size", None)
        if size is None:
            return 10.0

        size_array = np.asarray(size)
        if size_array.size == 0:
            return 10.0
        return float(size_array.ravel()[0])

    def _refresh_layers(self, event=None):
        anchor_current = self.points_anchor.currentText()
        question_current = self.points_question.currentText()
        layer_names = [layer.name for layer in self._points_layers_for_coloc()]

        for widget, current_text in (
            (self.points_anchor, anchor_current),
            (self.points_question, question_current),
        ):
            widget.blockSignals(True)
            widget.clear()
            widget.addItems(layer_names)
            if current_text in layer_names:
                widget.setCurrentText(current_text)
            widget.blockSignals(False)

        if (
            self.points_anchor.count() > 1
            and self.points_anchor.currentText() == self.points_question.currentText()
        ):
            next_idx = (self.points_anchor.currentIndex() + 1) % self.points_question.count()
            self.points_question.setCurrentIndex(next_idx)

        self._set_default_output_filename()

    def _layer_file_path(self, layer):
        for src_attr in ("source", "_source"):
            src = getattr(layer, src_attr, None)
            path = getattr(src, "path", None)
            if path:
                return Path(str(path))
        fallback = getattr(layer, "_filename", None)
        if fallback:
            return Path(str(fallback))
        return None

    def _on_filename_edited(self, _text):
        self._filename_user_set = True

    def _default_output_path(self):
        out_dir = self._output_directory()

        image_source = None
        for layer in self.viewer.layers:
            if layer._type_string == "image":
                image_source = self._layer_file_path(layer)
                if image_source is not None:
                    break

        if image_source is None:
            try:
                anchor_layer = self._get_selected_points_layer(self.points_anchor)
                image_source = self._layer_file_path(anchor_layer)
            except Exception:
                image_source = None

        stem = image_source.stem if image_source is not None else "unknown_file"
        return out_dir / f"{stem}_Spots_Coloc.csv"

    def _set_default_output_filename(self, force=False):
        if force or not getattr(self, "_filename_user_set", False):
            self.filename_edit.setText(str(self._default_output_path()))

    def _output_directory(self):
        try:
            anchor_layer = self._get_selected_points_layer(self.points_anchor)
            source_path = self._layer_file_path(anchor_layer)
            if source_path is not None:
                return source_path if source_path.is_dir() else source_path.parent
        except Exception:
            pass

        for layer in self.viewer.layers:
            if layer._type_string != "image":
                continue
            source_path = self._layer_file_path(layer)
            if source_path is not None:
                return source_path if source_path.is_dir() else source_path.parent

        return Path.cwd()

    def _spots_output_path(self):
        raw = (self.filename_edit.text() or "").strip()
        if not raw:
            return self._default_output_path()
        path = Path(raw)
        if not path.is_absolute():
            path = self._output_directory() / path
        return path

    def _master_stats_path(self):
        return self._output_directory() / "colocalization_master.csv"

    def _source_file_for_stats(self, anchor_layer=None):
        source_path = self._layer_file_path(anchor_layer) if anchor_layer is not None else None
        if source_path is not None:
            return str(source_path)
        for layer in self.viewer.layers:
            if layer._type_string != "image":
                continue
            source_path = self._layer_file_path(layer)
            if source_path is not None:
                return str(source_path)
        return ""

    def _points_dataframe(self, layer):
        import pandas as pd

        points = np.asarray(layer.data)
        n_coords = points.shape[1] if points.ndim == 2 else 0
        if n_coords == 2:
            coord_cols = ['y', 'x']
        elif n_coords == 3:
            # Keep backward compatibility with existing identify outputs.
            coord_cols = ['frame', 'y', 'x']
        elif n_coords == 4:
            coord_cols = ['frame', 'z', 'y', 'x']
        else:
            coord_cols = [f'axis_{i}' for i in range(n_coords)]
        df = pd.DataFrame(points, columns=coord_cols)
        df.attrs["coord_cols"] = coord_cols

        props = getattr(layer, "properties", {}) or {}
        _append_layer_properties(df, props)
        df = _drop_invalid_coordinate_rows(df, coord_cols)
        df.attrs["coord_cols"] = coord_cols

        return df

    def _coords_view(self, df):
        coord_cols = df.attrs.get("coord_cols")
        if not coord_cols:
            coord_cols = [c for c in ('frame', 'z', 'y', 'x') if c in df.columns]
        if not coord_cols:
            coord_cols = list(df.columns)
        return coord_cols, df[coord_cols].dropna()

    def _time_gradient_color(self, idx, total):
        # Light-to-dark blue gradient for increasing time.
        if total <= 1:
            return (70, 150, 245)
        t = float(idx) / float(total - 1)
        r = int(170 + (40 - 170) * t)
        g = int(220 + (90 - 220) * t)
        b = int(255 + (180 - 255) * t)
        return (r, g, b)

    def _append_stats_rows(self, rows):
        import pandas as pd

        if not rows:
            return
        path = self._master_stats_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        df_new = pd.DataFrame(rows)
        if path.exists():
            try:
                df_prev = pd.read_csv(path)
                cols = list(dict.fromkeys(list(df_prev.columns) + list(df_new.columns)))
                merged = pd.concat(
                    [
                        df_prev.reindex(columns=cols),
                        df_new.reindex(columns=cols),
                    ],
                    ignore_index=True,
                )
                merged.to_csv(path, index=False)
                return
            except Exception as err:
                print(f"Could not merge existing master stats ({err}); appending rows.")
        df_new.to_csv(path, mode='a', header=not path.exists(), index=False)

    def _run_colocalization(self, anchor_layer, question_layer):
        from sklearn.neighbors import KDTree
        import pandas as pd

        anchor_name = anchor_layer.name
        question_name = question_layer.name
        source_file = self._source_file_for_stats(anchor_layer)
        anchor_df = self._points_dataframe(anchor_layer)
        question_df = self._points_dataframe(question_layer)
        anchor_cols, anchor_coords = self._coords_view(anchor_df)
        _, question_coords = self._coords_view(question_df)

        if anchor_coords.empty or question_coords.empty:
            print("Selected points layer has no points.")
            return None
        if anchor_coords.shape[1] != question_coords.shape[1]:
            print("Anchor and comparison points have different dimensionality.")
            return None

        spatial_cols = [c for c in anchor_cols if c != "frame"] or anchor_cols
        threshold = float(self.euc_distance.value())
        stats_rows = []
        coloc_index_parts = []
        coloc_points_parts = []

        per_frame_mode = "frame" in anchor_cols and "frame" in question_coords.columns
        if per_frame_mode:
            anchor_work = anchor_coords.copy()
            question_work = question_coords.copy()
            anchor_work["__frame"] = pd.to_numeric(anchor_work["frame"], errors="coerce").round()
            question_work["__frame"] = pd.to_numeric(question_work["frame"], errors="coerce").round()
            anchor_work["__frame"].replace([np.inf, -np.inf], np.nan, inplace=True)
            question_work["__frame"].replace([np.inf, -np.inf], np.nan, inplace=True)
            anchor_work = anchor_work.dropna(subset=["__frame"]).copy()
            question_work = question_work.dropna(subset=["__frame"]).copy()
            anchor_work["__frame"] = anchor_work["__frame"].astype(np.int64)
            question_work["__frame"] = question_work["__frame"].astype(np.int64)

            frames = sorted(anchor_work["__frame"].unique().tolist())
            for idx, frame_value in enumerate(frames):
                anchor_frame = anchor_work[anchor_work["__frame"] == frame_value]
                question_frame = question_work[question_work["__frame"] == frame_value]
                anchor_count = len(anchor_frame)
                question_count = len(question_frame)

                if anchor_count == 0:
                    continue

                if question_count == 0:
                    coloc_count = 0
                    ratio = 0.0
                else:
                    tree = KDTree(question_frame[spatial_cols].to_numpy(), leaf_size=1)
                    distances_list = tree.query(anchor_frame[spatial_cols].to_numpy())[0].ravel()
                    hist, bins = np.histogram(distances_list, bins=100)
                    self._plt.plot(
                        bins[:-1],
                        hist,
                        pen=self._time_gradient_color(idx, len(frames)),
                        name=f"t={frame_value}",
                    )

                    mask = distances_list < threshold
                    frame_coloc_idx = anchor_frame.index[mask]
                    if len(frame_coloc_idx) > 0:
                        coloc_index_parts.append(np.asarray(frame_coloc_idx))
                        coloc_points_parts.append(anchor_frame.loc[frame_coloc_idx, anchor_cols].to_numpy())
                    coloc_count = int(mask.sum())
                    ratio = (coloc_count / anchor_count) if anchor_count else 0.0

                stats_rows.append(
                    {
                        "anchor_layer": anchor_name,
                        "comparison_layer": question_name,
                        "analysis_mode": "per_frame",
                        "frame": int(frame_value),
                        "anchor_particles": anchor_count,
                        "comparison_particles": question_count,
                        "colocalized_particles": coloc_count,
                        "colocalized_percent_anchor": ratio * 100.0,
                        "distance_threshold": threshold,
                        "spots_output_file": str(self._spots_output_path()),
                        "source_file": source_file,
                    }
                )
        else:
            tree = KDTree(question_coords[spatial_cols].to_numpy(), leaf_size=1)
            distances_list = tree.query(anchor_coords[spatial_cols].to_numpy())[0].ravel()
            hist, bins = np.histogram(distances_list, bins=100)
            self._plt.plot(
                bins[:-1],
                hist,
                pen=self._time_gradient_color(0, 1),
                name="all",
            )

            mask = distances_list < threshold
            coloc_indices = anchor_coords.index[mask]
            if len(coloc_indices) > 0:
                coloc_index_parts.append(np.asarray(coloc_indices))
                coloc_points_parts.append(anchor_coords.loc[coloc_indices, anchor_cols].to_numpy())

            anchor_count = len(anchor_coords)
            question_count = len(question_coords)
            coloc_count = int(mask.sum())
            ratio = (coloc_count / anchor_count) if anchor_count else 0.0
            stats_rows.append(
                {
                    "anchor_layer": anchor_name,
                    "comparison_layer": question_name,
                    "analysis_mode": "all_points",
                    "frame": "all",
                    "anchor_particles": anchor_count,
                    "comparison_particles": question_count,
                    "colocalized_particles": coloc_count,
                    "colocalized_percent_anchor": ratio * 100.0,
                    "distance_threshold": threshold,
                    "spots_output_file": str(self._spots_output_path()),
                    "source_file": source_file,
                }
            )

        if coloc_index_parts:
            coloc_indices = np.unique(np.concatenate(coloc_index_parts))
        else:
            coloc_indices = np.array([], dtype=int)

        if coloc_points_parts:
            colocalizing_points = np.concatenate(coloc_points_parts, axis=0)
        else:
            colocalizing_points = np.empty((0, len(anchor_cols)), dtype=float)

        coloc_name = f"Coloc_{question_name}_in_{anchor_name}"

        coloc_points = _add_points_compat(
            self.viewer,
            colocalizing_points,
            opacity=0.31,
            size=self._coloc_points_size(),
            blending='additive',
            border_width=0.15,
            symbol='square',
            name=coloc_name,
        )
        coloc_points.scale = anchor_layer.scale

        total_anchor = int(sum(row["anchor_particles"] for row in stats_rows))
        total_coloc = int(sum(row["colocalized_particles"] for row in stats_rows))
        ratio = (total_coloc / total_anchor) if total_anchor else 0.0

        label = QLabel(
            f"Number of colocalizing in {coloc_name}: "
            f"{total_anchor} {total_coloc} {ratio:.4f}"
        )
        self.layout().addWidget(label)

        column_name = f"colocalized {question_name} to {anchor_name}"
        df_out = anchor_df.copy()
        df_out[column_name] = 0
        df_out.loc[coloc_indices, column_name] = 1

        stats_suffix = f"{question_name} to {anchor_name}"
        anchor_stats_col = "anchor_particles per frame"
        question_stats_col = f"comparison_particles per frame {stats_suffix}"
        coloc_stats_col = f"colocalized_particles per frame {stats_suffix}"
        coloc_pct_col = f"colocalized_percent_anchor per frame {stats_suffix}"

        per_frame_rows = [r for r in stats_rows if r.get("analysis_mode") == "per_frame"]
        if "frame" in df_out.columns and per_frame_rows:
            frame_series = pd.to_numeric(df_out["frame"], errors="coerce").round()
            per_frame_map = {
                int(r["frame"]): r
                for r in per_frame_rows
                if isinstance(r.get("frame"), (int, np.integer))
            }

            def _lookup(row_frame, key):
                if pd.isna(row_frame):
                    return np.nan
                row = per_frame_map.get(int(row_frame))
                if row is None:
                    return np.nan
                return row.get(key, np.nan)

            df_out[anchor_stats_col] = frame_series.map(lambda f: _lookup(f, "anchor_particles"))
            df_out[question_stats_col] = frame_series.map(lambda f: _lookup(f, "comparison_particles"))
            df_out[coloc_stats_col] = frame_series.map(lambda f: _lookup(f, "colocalized_particles"))
            df_out[coloc_pct_col] = frame_series.map(lambda f: _lookup(f, "colocalized_percent_anchor"))
        elif stats_rows:
            total_anchor = int(sum(r.get("anchor_particles", 0) for r in stats_rows))
            total_question = int(sum(r.get("comparison_particles", 0) for r in stats_rows))
            total_coloc = int(sum(r.get("colocalized_particles", 0) for r in stats_rows))
            total_pct = (total_coloc / total_anchor * 100.0) if total_anchor else 0.0
            df_out[anchor_stats_col] = total_anchor
            df_out[question_stats_col] = total_question
            df_out[coloc_stats_col] = total_coloc
            df_out[coloc_pct_col] = total_pct

        return {
            "stats_rows": stats_rows,
            "column_name": column_name,
            "spots_df": df_out,
            "coloc_indices": coloc_indices,
            "colocalizing_points": colocalizing_points,
        }

    def _save_results(self):
        if hasattr(self, "_latest_coloc_df") and self._latest_coloc_df is not None:
            path = self._spots_output_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._latest_coloc_df.to_csv(path, index=False)
            print(f"Saved colocalization spots file: {path}")
            return

        # Fallback: save selected points layer if no colocalization dataframe exists.
        if self.viewer.layers.selection.active._type_string == 'points':
            layer = self.viewer.layers.selection.active
            df = self._points_dataframe(layer)
            out = self._spots_output_path()
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            print(f"Saved selected points file: {out}")



    def open_file_dialog(self):
        from pathlib import Path
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Select a File", 
            "/tmp/", 
            "Comma Separated Files (*.csv)"
        )
        if filename:
            path = Path(filename)
            self._filename_user_set = True
            self.filename_edit.setText(str(path))
    

    def calculate_colocalizing(self):
        print("Doing Colocalization")
        if self.points_anchor.count() == 0 or self.points_question.count() == 0:
            print("No points layers available for colocalization.")
            return

        self._plt.clear()
        anchor_layer = self._get_selected_points_layer(self.points_anchor)
        question_layer = self._get_selected_points_layer(self.points_question)
        result = self._run_colocalization(anchor_layer, question_layer)
        if result is None:
            return

        self._colocalizing_points = result["colocalizing_points"]
        self._latest_coloc_df = result["spots_df"]
        self._latest_stats_rows = list(result["stats_rows"])
        self._append_stats_rows(self._latest_stats_rows)

        if self.auto_save.isChecked():
            self._save_results()

    def calculate_all_colocalizing(self):
        if self.points_anchor.count() == 0 or self.points_question.count() == 0:
            print("No points layers available for colocalization.")
            return

        self._plt.clear()
        anchor_idx = self.points_anchor.currentIndex()
        anchor_layer = self._get_selected_points_layer(self.points_anchor)
        final_df = self._points_dataframe(anchor_layer).copy()
        anchor_base_cols = set(final_df.columns)
        stats_rows = []

        for i in range(self.points_question.count()):
            if i == anchor_idx:
                continue
            self.points_question.setCurrentIndex(i)
            question_layer = self._get_selected_points_layer(self.points_question)
            result = self._run_colocalization(anchor_layer, question_layer)
            if result is None:
                continue

            result_df = result["spots_df"]
            for col in result_df.columns:
                if col in anchor_base_cols:
                    continue
                final_df[col] = result_df[col]
            stats_rows.extend(result["stats_rows"])

        if not stats_rows:
            print("No valid comparisons were generated.")
            return

        self._latest_coloc_df = final_df
        self._latest_stats_rows = stats_rows
        self._append_stats_rows(stats_rows)

        if self.auto_save.isChecked():
            self._save_results()
    
    # def _select_layer(self,i):
    #     ##needs to be by name
    #     print("Layer to detect:", i)
    #     # self.llayer.setText(


    
    def _get_points(self,_widget):
        layer = self._get_selected_points_layer(_widget)
        df = self._points_dataframe(layer)
        _, coords = self._coords_view(df)
        return coords.to_numpy()
