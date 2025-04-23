"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
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
        
            
    def __init__(self, napari_viewer):
    
        super().__init__()
        self.viewer = napari_viewer
        
        self.points_options = dict(face_color=[0]*4,opacity=0.75,size=100,blending='additive',edge_width=0.15)
        # edge_color='red'
        self.points_options2 = dict(face_color=[0]*4,opacity=0.75,size=100,blending='additive',edge_width=0.15)
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
        self.mass_slider.setValue(25000)
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
        self.size_filter_tick.setChecked(True)
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
            batch_mass_slider.setValue(4000)
            
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
    def _on_click(self):
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
        minmass = self.mass_slider.value() 
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
            t0, t1   = self.min_timer.value(), self.max_timer.value()
            t_range  = range(t0, t1)
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
            
        try:
            _metadata = self.f.loc[:,['mass','size','ecc']]
        except:
            _metadata = self.f.loc[:,['mass','size_x','size_y','size_z']]
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
            self._points_layer = self.viewer.add_points(_points,name="Points "+name_points,properties=_metadata,**self.points_options,edge_color=point_colors)
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

        # 2) loop over files
        for fname in files:
            print(f"\n=== Processing {fname} ===")
            try:
                # Clear existing layers to avoid name clashes (optional; comment out
                # if you prefer to keep everything in the viewer)
                self._capture_batch_config()
                self.viewer.layers.clear()

                # Load with the napari‑aicsimageio plugin
                self.viewer.open(fname, plugin="napari-aicsimageio")
                
            except Exception as err:
                print(f"Could not open {fname}: {err}")
                continue

            # 3) run channel batch identification exactly as usual
            try:
                self.batch_on_click()
            except Exception as err:
                print(f"Identify failed on {fname}: {err}")
                continue
            
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
            cfg[name.split("::")[-1]] = (mass, check)      # keep suffix only
        self._batch_cfg = cfg                  # new attribute

    def _rebuild_batch_grid(self):
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
            spin.setValue(4000)
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
            key = w_label.widget().text().split("::")[-1]
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
        
        for i in range(self.layersbox.count()):
        #if index is checked, get the check from the grid self:
            if self.batch_grid_layout.itemAt(i*3+2).widget().isChecked():
                #update selection to match
                self.layersbox.setCurrentIndex(i)
                #update mass value get the value from grid
                self.mass_slider.setValue(
                    #no idea how to do this
                    #example
                    self.batch_grid_layout.itemAt(i*3+1).widget().value()
                                          )
                self._on_click()
            


    # @thread_worker
    def _on_click2(self):
        index_layer = _get_choice_layer(self,self.layersbox)
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
        try:
            _metadata = f2.loc[:,['mass','size','ecc']]
        except:
            _metadata = f2.loc[:,['mass','size']]

        self.f2 = f2
        self._points_layer_filter = self.viewer.add_points(_points,properties=_metadata,**self.points_options2)
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

        _populate_layers(self,self.points_anchor,"points")
        _populate_layers(self,self.points_question,"points")

        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.reordered.connect(self._refresh_layers)
        
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
        # self.filename_edit.setText(_get_open_filename(self)+"_Spots_Coloc.csv")
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

    def _save_results(self):
        import pandas as pd
        ##TODO
        ##pull from points layer see example below
        if self.viewer.layers.selection.active._type_string == 'points':
            _index_layer = self.viewer.layers.selection.active
            # df = _get_points(self)
             #manbearpig   
        # self.viewer.layers[0].data
        #need to make this smarter, if the layers are shuffled it breaks.
            
            if _index_layer.data.shape[1] < 3:
                df = pd.DataFrame(_index_layer.data, columns = ['frame','y','x'])
                print("2D",df)
            elif _index_layer.data.shape[1] >= 3:
                df = pd.DataFrame(_index_layer.data, columns = ['frame','z','y','x'])
                print("3D",df)

            # b = self.viewer.layers.selection.active.properties
# self.filename_edit
            _save_name = self.viewer.layers.selection.active.name
            df.to_csv(self.filename_edit.text().strip(".csv")+"_"+_save_name+".csv")



    def open_file_dialog(self):
        from pathlib import Path
        filename, ok = QFileDialog.getSave
        FileName(
            self,
            "Select a File", 
            "/tmp/", 
            "Comma Separated Files (*.csv)"
        )
        if filename:
            path = Path(filename)
            self.filename_edit.setText(str(path))
    

    def calculate_colocalizing(self):
        #makecode from notebooks
        import scipy
        import pyqtgraph as pg
        from sklearn.neighbors import KDTree
        print("Doing Colocalization")
        QuestionPOS = self._get_points(self.points_question)
        #print(QuestionPOS)
        AnchorPOS = self._get_points(self.points_anchor)
        #print(AnchorPOS)

        #scipy way
        # question_kd = scipy.spatial.cKDTree(QuestionPOS,
        # leafsize=10000.
        # # leafsize=0.
        # )
        # distances_list = np.asarray(question_kd.query(AnchorPOS,p=2,
        #         #distance_upper_bound=10*psize
        #         ))[0]
        
        #scikitlearn
        
        tree = KDTree(QuestionPOS, leaf_size=1)
        distances_list = tree.query(AnchorPOS)[0]
        _hist,_bins = np.histogram(distances_list,bins=100)
        # _plt.showGrid(x=True, y=True)()
        print(distances_list,_bins,_hist)
        
        # try:
        #     idx   = int(np.random.randint(16))   # cast NumPy scalar → builtin int
        #     cycle = 16                           # already a plain int
        #     line1 = self._plt.plot(
        #         _bins[:-1], _hist, 
        #         pen=(idx, cycle),  
        #         name="green"
        #         )    
        # except: print("pyqtgraph failed")
        
        line1 = self._plt.plot(
            _bins[:-1],_hist,
            #   distances_list, 
            # pen=pg.mkPen(pg.intColor(np.random.randint(16), 16)),
            pen=(np.random.randint(16), 16),
            # pen='r'
            # pen=self.points_question.colormap.name
            #   symbol='x', symbolPen='g',
            #   symbolBrush=0.2, 
            name='green'
            )
 
        # self._plt.setImage(distances_list
        #               , xvals=np.linspace(0., 100., distances_list.shape[0]))
        _colocalizing = distances_list[distances_list < self.euc_distance.value()]
        #
        #Is this faster than above?
        #print(tree.query_radius(AnchorPOS, r=self.euc_distance.value(), count_only=True))
        #ind = tree.query_radius(AnchorPOS, r=self.euc_distance.value()) 
        
        
        _colocalizing_points = AnchorPOS[distances_list < self.euc_distance.value()]
        print("Remove me:",_colocalizing_points.shape,QuestionPOS.shape)
        
        coloc_name = "Coloc_"+self.points_question.currentText()+'_in_'+self.points_anchor.currentText()
        coloc_points = self.viewer.add_points(_colocalizing_points, opacity=0.31,
                                              size=150,blending='additive',edge_width=0.15,symbol='square',
                                              name=coloc_name)
        coloc_points.scale = self.viewer.layers[0].scale
        
        l_coloc = QLabel("Number of colocalizing in "+coloc_name+":"+str(
            len(AnchorPOS))+" "+str(
            len(_colocalizing))+" "+str(
                    len(_colocalizing)/len(AnchorPOS)))
        self.layout().addWidget(l_coloc)


        self._colocalizing_points = _colocalizing_points
        if self.auto_save.isChecked():
            self.filename_edit.setText(_get_open_filename(
                    self)+'_'+coloc_name+"_Spots.csv")
            self._save_results()

        # return {coloc_name:[AnchorPOS,_colocalizing])}
        # return spots count Filename what channel is anchor which is question and numbers?
        #save directly to a file that is all.csv or file_coloc_counts.csv?

    def calculate_all_colocalizing(self):
        idx_anchor = self.points_anchor.currentIndex()
        _num_items = self.points_question.count()
        self.points_question.setCurrentIndex(0)
        for i in range(_num_items):
            if self.points_question.currentIndex() == idx_anchor:
                if self.points_question.currentIndex() < _num_items-1:
                    self.points_question.setCurrentIndex(
                        self.points_question.currentIndex()+1)
            else:
                self.calculate_colocalizing()
                if self.points_question.currentIndex() < _num_items-1:
                    self.points_question.setCurrentIndex(
                        self.points_question.currentIndex()+1)

        #fortripple coloc
        #if the anchor is the last item, it will not work because it will compare to itself.
        #fix ASAP
        if _num_items == 3:
        #then set anchor to the first result (anchor plus first available) 
            #0 index so +1 is not needed
            self.points_anchor.setCurrentIndex(_num_items)
            if self.points_question.currentIndex() == idx_anchor:
                self.points_question.setCurrentIndex(
                    self.points_question.currentIndex()-1)
        # and compare to the other that was not anchor in the beggining
            #normally the points_question was already in the last
            #point_list that is not the anchor, so it should be ready to go
            self.calculate_colocalizing()
    
    # def _select_layer(self,i):
    #     ##needs to be by name
    #     print("Layer to detect:", i)
    #     # self.llayer.setText(


    
    def _get_points(self,_widget):
        import pandas as pd
        _index_layer = _get_choice_layer(self,_widget)
        
        # self.viewer.layers[_widget.currentIndex()].data
        # if len(self.viewer.layers[_index_layer].data.shape) < 3:
        #     df = pd.DataFrame(self.viewer.layers[_index_layer].data, columns = ['frame','y','x'])
        #     print("2D",df)
        # elif len(self.viewer.layers[_widget.currentIndex()].data.shape) >= 3:
        #     df = pd.DataFrame(self.viewer.layers[_index_layer].data, columns = ['frame','z','y','x'])
        #     print("3D",df)
        # b = self.viewer.layers.selection.active.properties
        # for key in b.keys():
        #     df[key] = b[key]
        
        # self.viewer.layers[0].data
        # if len(self.viewer.layers[0].data.shape) <= 3:
        if self.viewer.layers[_index_layer].data.shape[1] < 3:
        
            df = pd.DataFrame(self.viewer.layers[_index_layer].data, columns = ['frame','y','x']).dropna()
            print("2D",df)
        elif self.viewer.layers[_index_layer].data.shape[1] >= 3:
            df = pd.DataFrame(self.viewer.layers[ _index_layer].data, columns = ['frame','z','y','x']).dropna()
            print("3D",df)

        # b = self.viewer.layers.selection.active.properties
        #error is here somehow now
        # for key in b.keys():
        #     df[key] = b[key]
    
        return df
