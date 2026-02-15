"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from pathlib import Path
from typing import TYPE_CHECKING
import json
import trackpy as tp
from magicgui import magic_factory
from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout,QPushButton, QCheckBox, QComboBox, QSpinBox
from qtpy.QtWidgets import QLabel, QDoubleSpinBox, QWidget, QGridLayout
from qtpy.QtWidgets import QSpacerItem, QSizePolicy, QFileDialog, QLineEdit
from napari.qt.threading import thread_worker
import numpy as np

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
    for j,layer in enumerate(self.viewer.layers):
        if layer.name == _widget.currentText():
            index_layer = j
            break
    print("Layer where points are is:",j)
    return index_layer

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

##Numba doesn't work for 2D yet.
from numba import jit, prange
@jit(nopython=True, parallel=True)
def fill_mask_numba(mask, positions, radius):
    shape = mask.shape
    for i in prange(len(positions)):
        pz, py, px = positions[i]
        for x in range(shape[2]):
            for y in range(shape[1]):
                for z in range(shape[0]):
                    distance = np.sqrt((px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2)
                    if distance <= radius:
                        mask[z, y, x] = 1

def fill_mask_cupy(mask, positions, radius):
    import cupy as cp
    shape = mask.shape
    for i in range(len(positions)):
        pz, py, px = positions[i]
        x, y, z = cp.meshgrid(cp.arange(shape[2]), cp.arange(shape[1]), cp.arange(shape[0]))
        distance = cp.sqrt((px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2)
        mask[distance <= radius] = 1
        
        
def make_labels_trackpy_links(shape,j,radius=5,_algo="GPU"):
    """
    Creates binary masks around given positions with a specified radius in a 3D space using PyOpenCL.

    :param shape: Tuple of the output volume shape (Z, Y, X).
    :param positions: NumPy array of positions (Z, Y, X).
    :param radius: The radius around each position to fill in the mask.
    :return: A 3D NumPy array representing the labeled masks. The positions as array
    """
    import trackpy as tp
    import scipy.ndimage as ndi
    from scipy.ndimage import binary_dilation
    
    if 'z' in j:
    # "Need to loop each t and do one at a time"
        pos = np.dstack((j.z,j.y,j.x))[0]#.astype(int)
        print("3D",j)
    else:
        pos = np.dstack((j.y,j.x))[0]#.astype(int)
        print("2D",j)
    
    if _algo == "GPU":
        import cupy as cp
        
        pos_cp = cp.asarray(pos)

        ##this is what tp.masks.mask_image does maybe put a cupy here to make if faster.
        ndim = len(shape)
        # radius = validate_tuple(radius, ndim)
        pos_cp = cp.atleast_2d(pos_cp)

        # if include_edge:
        in_mask = cp.array([cp.sum(((cp.indices(shape).T - p) / radius)**2, -1) <= 1
                    for p in pos_cp])
        # else:
        #     in_mask = [np.sum(((np.indices(shape).T - p) / radius)**2, -1) < 1
        #                for p in pos]
        mask_total = cp.any(in_mask, axis=0).T
        
        ##if they overlap the labels won't match the points
        #we can make np.ones * ID of the point and then np.max(axis=-1)
        labels, nb = ndi.label(cp.asnumpy(mask_total))
        # image * mask_cluster.astype(np.uint8)
        
        #this is super slow
        # ~ masks = tp.masks.mask_image(coords,np.ones(image.shape),size/2)
    elif _algo=='CPU':


        ##this is what tp.masks.mask_image does maybe put a cupy here to make if faster.
        ndim = len(shape)
        # radius = validate_tuple(radius, ndim)
        pos = np.atleast_2d(pos)
        # if include_edge:
        in_mask = np.array([np.sum(((np.indices(shape).T - p) / radius)**2, -1) <= 1
                    for p in pos])
        # else:
        #     in_mask = [np.sum(((np.indices(shape).T - p) / radius)**2, -1) < 1
        #                for p in pos]
        mask_total = np.any(in_mask, axis=0).T
        ##if they overlap the labels won't match the points
        #we can make np.ones * ID of the point and then np.max(axis=-1)
        labels, nb = ndi.label(mask_total)
    elif _algo=='fast':
    #This is faster
        
        # r = (radius-1)/2 # Radius of circles
        # print(radius,r)
    #     #make 3D compat
        disk_mask = tp.masks.binary_mask(radius,len(shape))
        # print(disk_mask)
    #     # Initialize output array and set the maskcenters as 1s
        out = np.zeros(shape,dtype=bool)

        if 'z' in j:
            pos = np.dstack((j.z,j.y,j.x))[0].astype(int)
            pos = np.atleast_2d(pos)
            print(pos)
            out[pos[:,0],pos[:,1],pos[:,2]] = 1            

        else:
            pos = np.dstack((j.y,j.x))[0].astype(int)
            pos = np.atleast_2d(pos)
            print(pos)
            out[pos[:,0],pos[:,1]] = 1
    #     # Use binary dilation to get the desired output
    
        out = binary_dilation(out,disk_mask)

        labels, nb = ndi.label(out)
        print("Number of labels:",nb)
        # if _round:
        #     return labels, coords
        # else:
        #     if image.ndim == 2:
        #         # coords = j.loc[:,['particle','frame','y','x']]
        #         coords = j.loc[:,['frame','y','x']]
        #         # coords = np.dstack((j.particle,j.y,j.x))[0]
        #         return labels, coords
    elif _algo == 'numba': 
   
        # Prepare data
        mask = np.zeros(shape, dtype=np.uint8)
        
        # Fill mask using Numba
        fill_mask_numba(mask, pos, radius)
        
        # Use label function from scipy to identify connected components
        labels, _ = ndi.label(mask)

    elif _algo == 'openCL':
        print("Using openCL function")
        import pyopencl as cl

            # Prepare data
        mask = np.zeros(shape, dtype=np.uint8)
        positions_flat = pos.flatten().astype(np.float32)
        radius = np.float32(radius)
        
        platform = cl.get_platforms()
        my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        ctx = cl.Context(devices=my_gpu_devices)

        # PyOpenCL setup
    #     ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        # Create buffers
        mask_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=mask)
        positions_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=positions_flat)
        
        # Kernel code
        if 'z' in j:
            kernel_code = """
            __kernel void fill_mask(__global uchar *mask, __global const float *positions, const float radius, const int num_positions) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int z = get_global_id(2);
                int width = get_global_size(0);
                int height = get_global_size(1);
                int depth = get_global_size(2);
                int idx = x + y * width + z * width * height;
                
                for (int i = 0; i < num_positions; ++i) {
                    float pz = positions[i * 3];
                    float py = positions[i * 3 + 1];
                    float px = positions[i * 3 + 2];
                    
                    float distance = sqrt(pow(px - x, 2) + pow(py - y, 2) + pow(pz - z, 2));
                    if (distance <= radius) {
                        mask[idx] = 1;
                        break;
                    }
                }
            }
            """
        else:
            kernel_code = """
            __kernel void fill_mask(__global uchar *mask, __global const float *positions, const float radius, const int num_positions) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int width = get_global_size(0);
                int height = get_global_size(1);
                int idx = x + y * width;
                
                for (int i = 0; i < num_positions; ++i) {
                    float py = positions[i * 2];
                    float px = positions[i * 2 + 1];
                    
                    float distance = sqrt(pow(px - x, 2) + pow(py - y, 2));
                    if (distance <= radius) {
                        mask[idx] = 1;
                        break;
                    }
                }
            }
            """
        # Build kernel
        prg = cl.Program(ctx, kernel_code).build()
        
        # Execute kernel
        global_size = shape[::-1]  # Note: PyOpenCL uses column-major order, so we reverse the dimensions
        prg.fill_mask(queue, global_size, None, mask_buf, positions_buf, radius, np.int32(len(pos)))
        
        # Read back the results
        cl.enqueue_copy(queue, mask, mask_buf)
        labels, nb = ndi.label(mask)
        print("End of openCL function")
    return labels, pos



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

    def _default_identify_settings(self) -> dict:
        return {
            "mass_threshold": 0,
            "diameter": "3,5,5",
            "size_filter_enabled": False,
            "size_cutoff": 1.6,
            "ecc_enabled": False,
            "ecc_cutoff": 0.35,
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
        return {
            "mass_threshold": int(self.mass_slider.value()),
            "diameter": (self.diameter_input.text().strip() or "3,5,5"),
            "size_filter_enabled": bool(self.size_filter_tick.isChecked()),
            "size_cutoff": float(self.size_filter_input.value()),
            "ecc_enabled": bool(self.ecc_tick.isChecked()),
            "ecc_cutoff": float(self.ecc_input.value()),
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

        self.size_filter_tick.setChecked(bool(cfg.get("size_filter_enabled", defaults["size_filter_enabled"])))
        try:
            self.size_filter_input.setValue(float(cfg["size_cutoff"]))
        except Exception:
            self.size_filter_input.setValue(defaults["size_cutoff"])

        self.ecc_tick.setChecked(bool(cfg.get("ecc_enabled", defaults["ecc_enabled"])))
        try:
            self.ecc_input.setValue(float(cfg["ecc_cutoff"]))
        except Exception:
            self.ecc_input.setValue(defaults["ecc_cutoff"])

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
        l3.setText("Size cutoff (Select for usage)")
        self.layoutH0 = QHBoxLayout()
        self.size_filter_tick = QCheckBox()
        self.size_filter_tick.setChecked(False)
        self.size_filter_input = QDoubleSpinBox()
        self.size_filter_input.setRange(0, 10)
        self.size_filter_input.setSingleStep(0.05)
        self.size_filter_input.setValue(1.60)
        self.layoutH0.addWidget(self.size_filter_tick)
        self.layoutH0.addWidget(self.size_filter_input)

        l4 = QLabel()
        l4.setText("Eccentricity cutoff (Select for usage)")
        self.layoutH0p = QHBoxLayout()
        self.ecc_tick = QCheckBox()
        self.ecc_tick.setChecked(False)
        self.ecc_input = QDoubleSpinBox()
        self.ecc_input.setRange(0, 2)
        self.ecc_input.setSingleStep(0.05)
        self.ecc_input.setValue(0.35)
        self.layoutH0p.addWidget(self.ecc_tick)
        self.layoutH0p.addWidget(self.ecc_input)


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
        self.min_timer.setRange(0,int(self.viewer.dims.range[0][0]))
        self.min_timer.setSingleStep(int(self.viewer.dims.range[0][2]))
        self.min_timer.setValue(0)
        self.max_timer = QSpinBox()
        self.max_timer.setRange(0,int(self.viewer.dims.range[0][1]))
        self.max_timer.setSingleStep(int(self.viewer.dims.range[0][2]))
        self.max_timer.setValue(int(self.viewer.dims.range[0][1]))
        self.layoutH2.addWidget(l_min_time)
        self.layoutH2.addWidget(self.min_timer)
        self.layoutH2.addWidget(l_max_time)
        self.layoutH2.addWidget(self.max_timer)


        label_masks = QLabel()
        label_masks.setText("Make Masks?")
        self.layout_masks = QHBoxLayout()
        self.make_masks_box = QCheckBox()
        self.make_masks_box.setChecked(False)
        self.masks_option = QComboBox()
        self.masks_option.addItems(["openCL","numba","subpixel GPU Cupy","subpixel CPU Numpy","coarse CPU",])
        self.layout_masks.addWidget(label_masks)
        self.layout_masks.addWidget(self.make_masks_box)
        self.layout_masks.addWidget(self.masks_option)
        self.masks_dict = {0:'openCL',1:'numba',2:'GPU',3:'CPU',4:'fast',}

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
        _batch_files_btn = QPushButton('Batch Identify Files')
        _batch_files_btn.clicked.connect(self._batch_files)
        self.layout().addWidget(_batch_files_btn)
        self._load_identify_settings()


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
        # self.llayer.setText(
        # return j
        
    
    def make_masks(self):
        import pandas as pd
        # if self.viewer.layers[index_layer].scale[0] != 1:
        index_layer = _get_choice_layer(self,self.layersbox)

        if self.viewer.layers.selection.active.data.shape[1] <= 3:
            ##fix here to distinguish between ZYX TYX
            df = pd.DataFrame(self.viewer.layers.selection.active.data, columns = ['frame','y','x'])
        elif self.viewer.layers.selection.active.data.shape[1] > 3:
            # if self.viewer.layers[index_layer].scale[0] != 1
            df = pd.DataFrame(self.viewer.layers.selection.active.data, columns = ['frame','z','y','x'])
            # else:
                # df = pd.DataFrame(self.viewer.layers.selection.active.data, columns = ['frame','y','x'])
        b = self.viewer.layers.selection.active.properties
        for key in b.keys():
            df[key] = b[key]

        masks = np.zeros(self.viewer.layers[index_layer].data.shape).astype('int64')
        idx = []
        # links.loc[:,['particle','frame','y','x']]


        for i in df.frame.unique():
            i = int(i)
            temp = df[df['frame'] == i].sort_values(by=['y'])
            #0 returns mask, 1 index returns coords
            self.masks_dict[self.masks_option.currentIndex()]
            print("Doing Masks with option:",self.masks_option.currentIndex(), self.masks_dict[self.masks_option.currentIndex()])
            if self.viewer.layers[index_layer].scale[0] == 1:
                input_shape = self.viewer.layers[index_layer].data[i].shape
            else: input_shape = self.viewer.layers[index_layer].data.shape
            _diam = self._diameter_scalar()         # int no matter what user typed
            _radius = (_diam / 2) - 0.5              # same formula you had before

            mask_temp, idx_temp = make_labels_trackpy_links(
                input_shape,
                # self.viewer.lcp ayers[index_layer].data[i],
                temp,
                radius=_radius,
                # _round=False,
                _algo=self.masks_dict[self.masks_option.currentIndex()],
                )
                # temp,size=5-1)
        #     print(mask_temp.max(),len(temp.index))
        #     idx.append(idx_temp)
            mask_fixed = np.copy(mask_temp)
            ##this is needed when doing from links because all labels set as 'particles'
            # for j in np.unique(mask_temp).astype('int'):
            #     if j != 0:
            #         # mask_fixed[mask_temp == j] = temp.iloc[j-1]['particle'].astype('int')
            #         mask_fixed[mask_temp == j] = temp.iloc[j-1][:].astype('int')
        #     print(np.unique(mask_fixed),temp['particle'].unique())
            if self.viewer.layers[index_layer].scale[0] == 1:
                masks[i,...] = mask_fixed
            else: masks = mask_fixed

        return masks
    
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
            return np.asarray(layer.data[tuple(sl)])

        def _stack3d(t_idx=0):
            """Return a full Z‑stack (ZYX) at a given time index."""
            sl = [slice(None)] * layer.data.ndim
            if axes.t is not None:
                sl[axes.t] = t_idx
            # leave Z slice(None) → full stack
            return np.asarray(layer.data[tuple(sl)])

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
            image_seq = (
                [_plane2d(t_idx=t) for t in t_range]     # TYX
                if axes.z is None
                else [_stack3d(t_idx=t) for t in t_range]  # TZYX
            )

            # ---------- ipyparallel (default) ------------------------------------
            from ipyparallel import Client
            def _loc_worker(frame_idx, img):
                import trackpy as tp, pandas as pd
                try: img = img.compute()      # dask → numpy if needed
                except AttributeError: pass
                df = tp.locate(img, diam, minmass=minmass, engine="numba")
                df["frame"] = frame_idx
                return df

            rc = Client()
            v  = rc.load_balanced_view()
            print("Using", len(rc), "ipyparallel engines")

            async_res = v.map(_loc_worker, list(t_range), image_seq)
            self.f = pd.concat(async_res.get(), ignore_index=True)

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
        
        #filtering steps
        if len(self.viewer.layers[index_layer].data.shape) <= 3:
            if self.ecc_tick.isChecked():
                    self.f = self.f[ (self.f['ecc'] < self.ecc_input.value())
                    ]
        if self.size_filter_tick.isChecked():
            cutoff = self.size_filter_input.value()

            if "size" in self.f.columns:                      # isotropic case
                self.f = self.f[self.f["size"] < cutoff]

            else:                                             # anisotropic columns
                # pick the axes you care about
                axes = [c for c in ("size_x", "size_y") if c in self.f.columns]

                if axes:                                      # protect against typos
                    mask = (self.f[axes] < cutoff).all(axis=1)
                    self.f = self.f[mask]
                # if neither size_x nor size_y exists, do nothing (or raise an error)
                
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
        point_colors = clr_dict[clr_name]
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

            #auto_save depends on the last created spots layer, if it's done after make_masks, it segfaults
            #Keep the same order or redo the code
            if self.auto_save.isChecked():
                self._save_results()

            if self.make_masks_box.isChecked():
                _masks = self.make_masks()
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

        # Freeze the current batch-grid values for this multi-file run.
        self._capture_batch_config()
        self._batch_files_frozen_cfg = dict(getattr(self, "_batch_cfg", {}))
        self._batch_files_running = True
        try:
            # 2) loop over files
            for fname in files:
                print(f"\n=== Processing {fname} ===")
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
                    continue

                # 3) run channel batch identification exactly as usual
                try:
                    self.batch_on_click()
                except Exception as err:
                    print(f"Identify failed on {fname}: {err}")
                    continue
        finally:
            self._batch_files_running = False
            self._batch_files_frozen_cfg = {}
            
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

    def batch_on_click(self):
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

        self._batch_running = True
        try:
            for row_idx, mass_value in selected_rows:
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
        #add the filters here if check size ..if ecc...
        if self.ecc_tick.isChecked():
            f2 = f2[ (f2['ecc'] < self.ecc_input.value())
                   ]
        if self.size_filter_tick.isChecked():
            f2 = f2[ (f2['size'] < self.size_filter_input.value())
                   ]
        f2 = f2[(self.f['mass'] > self.mass_slider.value())    
                   ]

        
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

    def _save_results(self):
        import pandas as pd
        ##TODO
        ##pull from points layer see example below
        selected_layer = _get_choice_layer(self,self.layersbox)
        # selected_layer = self.viewer.layers.selection
        if self.viewer.layers.selection.active.data.shape[1] < 3:
            #manbearpig time lapse vs Zstack
            df = pd.DataFrame(self.viewer.layers.selection.active.data, columns = ['frame','y','x'])
        elif self.viewer.layers.selection.active.data.shape[1] >= 3:
            df = pd.DataFrame(self.viewer.layers.selection.active.data, columns = ['frame','z','y','x'])
            
        b = self.viewer.layers.selection.active.properties
        for key in b.keys():
            df[key] = b[key]
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
            self.filename_edit.setText(str(path))
    
    def _save_results_links(self):
        import pandas as pd
        self.links.to_csv(self.filename_edit_links.text())

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
        ##if 2d
        #this is from the other widget..dang
        # self.layersbox.currentIndex()
        #ALTERNATIVE:if there is a column named 'z'
        if len(self.viewer.layers[0].data.shape) <= 3:
            df = pd.DataFrame(self.viewer.layers.selection.active.data, columns = ['frame','y','x'])
        elif len(self.viewer.layers[0].data.shape) > 3:
            df = pd.DataFrame(self.viewer.layers.selection.active.data, columns = ['frame','z','y','x'])
        b = self.viewer.layers.selection.active.properties
        for key in b.keys():
            df[key] = b[key]

        print(df)
        links = tp.link(df, search_range=self.distance.value(),memory=self.memory.value())
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
        _tracks.scale = self.viewer.layers[0].scale[1:]
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

    def _points_dataframe(self, layer):
        import pandas as pd

        points = np.asarray(layer.data)
        ndim = points.shape[1] if points.ndim == 2 else 3
        coord_cols = ['frame', 'z', 'y', 'x'] if ndim >= 4 else ['frame', 'y', 'x']
        df = pd.DataFrame(points, columns=coord_cols)

        props = getattr(layer, "properties", {}) or {}
        for key, values in props.items():
            try:
                if len(values) == len(df):
                    df[key] = values
            except Exception:
                continue

        return df

    def _coords_view(self, df):
        coord_cols = ['frame', 'z', 'y', 'x'] if 'z' in df.columns else ['frame', 'y', 'x']
        return coord_cols, df[coord_cols].dropna()

    def _append_stats_rows(self, rows):
        import pandas as pd

        if not rows:
            return
        path = self._master_stats_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(path, mode='a', header=not path.exists(), index=False)

    def _run_colocalization(self, anchor_layer, question_layer):
        from sklearn.neighbors import KDTree

        anchor_name = anchor_layer.name
        question_name = question_layer.name
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

        tree = KDTree(question_coords.to_numpy(), leaf_size=1)
        distances_list = tree.query(anchor_coords.to_numpy())[0].ravel()
        hist, bins = np.histogram(distances_list, bins=100)
        self._plt.plot(
            bins[:-1],
            hist,
            pen=(np.random.randint(16), 16),
            name='green',
        )

        mask = distances_list < self.euc_distance.value()
        coloc_indices = anchor_coords.index[mask]
        colocalizing_points = anchor_coords.to_numpy()[mask]
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

        anchor_count = len(anchor_coords)
        question_count = len(question_coords)
        coloc_count = int(mask.sum())
        ratio = (coloc_count / anchor_count) if anchor_count else 0.0
        stats = {
            "anchor_layer": anchor_name,
            "comparison_layer": question_name,
            "anchor_particles": anchor_count,
            "comparison_particles": question_count,
            "colocalized_particles": coloc_count,
            "colocalized_percent_anchor": ratio * 100.0,
            "distance_threshold": float(self.euc_distance.value()),
            "spots_output_file": str(self._spots_output_path()),
        }

        label = QLabel(
            f"Number of colocalizing in {coloc_name}: "
            f"{anchor_count} {coloc_count} {ratio:.4f}"
        )
        self.layout().addWidget(label)

        column_name = f"colocalized {question_name} to {anchor_name}"
        df_out = anchor_df.copy()
        df_out[column_name] = 0
        df_out.loc[coloc_indices, column_name] = 1

        return {
            "stats": stats,
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

        anchor_layer = self._get_selected_points_layer(self.points_anchor)
        question_layer = self._get_selected_points_layer(self.points_question)
        result = self._run_colocalization(anchor_layer, question_layer)
        if result is None:
            return

        self._colocalizing_points = result["colocalizing_points"]
        self._latest_coloc_df = result["spots_df"]
        self._latest_stats_rows = [result["stats"]]
        self._append_stats_rows(self._latest_stats_rows)

        if self.auto_save.isChecked():
            self._save_results()

    def calculate_all_colocalizing(self):
        if self.points_anchor.count() == 0 or self.points_question.count() == 0:
            print("No points layers available for colocalization.")
            return

        anchor_idx = self.points_anchor.currentIndex()
        anchor_layer = self._get_selected_points_layer(self.points_anchor)
        final_df = self._points_dataframe(anchor_layer).copy()
        stats_rows = []

        for i in range(self.points_question.count()):
            if i == anchor_idx:
                continue
            self.points_question.setCurrentIndex(i)
            question_layer = self._get_selected_points_layer(self.points_question)
            result = self._run_colocalization(anchor_layer, question_layer)
            if result is None:
                continue

            col_name = result["column_name"]
            final_df[col_name] = 0
            final_df.loc[result["coloc_indices"], col_name] = 1
            stats_rows.append(result["stats"])

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
