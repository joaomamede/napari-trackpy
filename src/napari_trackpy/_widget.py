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
# from './support_libraries' import make_labels_trackpy_links

if TYPE_CHECKING:
    import napari

def make_labels_trackpy_links(image,j,size=5,_round=False):
    import trackpy as tp
    import scipy.ndimage as ndi
    from scipy.ndimage import binary_dilation


    #outputsomehow is 3D, we want 2
    coords = np.dstack((round(j.y),round(j.x)))[0].astype(int)

    #this is super slow
    # ~ masks = tp.masks.mask_image(coords,np.ones(image.shape),size/2)

    #This is faster
    r = (size-1)/2 # Radius of circles
    #make 3D compat
    disk_mask = tp.masks.binary_mask(r,image.ndim)
    # Initialize output array and set the maskcenters as 1s
    out = np.zeros(image.shape,dtype=bool)
    #check if there's a problem with subpixel masking
    out[coords[:,0],coords[:,1]] = 1
    # Use binary dilation to get the desired output
    out = binary_dilation(out,disk_mask)


    labels, nb = ndi.label(out)
    if _round:
        return labels, coords
    else:
        if image.ndim == 2:
            # coords = j.loc[:,['particle','frame','y','x']]
            coords = j.loc[:,['frame','y','x']]
#             coords = np.dstack((j.particle,j.y,j.x))[0]
#             return labels, coords
# #     return labels, coords

class IdentifyQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.points_options = dict(face_color=[0]*4,edge_color='red',opacity=0.5,size=5,blending='additive')
        self.points_options2 = dict(face_color=[0]*4,edge_color='green',opacity=0.5,size=5,blending='additive')
        #comboBox for layer selection
        self.llayer = QLabel()
        self.llayer.setText("Layer to detect")
        self.layersbox = QComboBox()
        self.layersbox.currentIndexChanged.connect(self._select_layer)
        
        self._populate_layers()

        l1 = QLabel()
        l1.setText("Mass Threshold")
        self.mass_slider = QSpinBox()
        self.mass_slider.setRange(0, 1e6)
        self.mass_slider.setSingleStep(200)
        self.mass_slider.setValue(500)
        l2 = QLabel()
        
        l2.setText("Diameter of the particle")
        self.diameter_input = QSpinBox()
        self.diameter_input.setRange(1, 19)
        self.diameter_input.setSingleStep(2)
        self.diameter_input.setValue(5)

        
        l3 = QLabel()
        l3.setText("Size cutoff (Select for usage)")
        self.layoutH0 = QHBoxLayout()
        self.size_filter_tick = QCheckBox()
        self.size_filter_tick.setChecked(True)
        self.size_filter_input = QDoubleSpinBox()
        self.size_filter_input.setRange(0, 10)
        self.size_filter_input.setSingleStep(0.05)
        self.size_filter_input.setValue(1.90)
        self.layoutH0.addWidget(self.size_filter_tick)
        self.layoutH0.addWidget(self.size_filter_input)

        l4 = QLabel()
        l4.setText("Eccentricity cutoff (Select for usage)")
        self.layoutH0p = QHBoxLayout()
        self.ecc_tick = QCheckBox()
        self.ecc_tick.setChecked(True)
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
        self.min_timer.setRange(0,self.viewer.dims.range[0][0])
        self.min_timer.setSingleStep(self.viewer.dims.range[0][2])
        self.min_timer.setValue(0)
        self.max_timer = QSpinBox()
        self.max_timer.setRange(0,self.viewer.dims.range[0][1])
        self.max_timer.setSingleStep(self.viewer.dims.range[0][2])
        self.max_timer.setValue(self.viewer.dims.range[0][1])
        self.layoutH2.addWidget(l_min_time)
        self.layoutH2.addWidget(self.min_timer)
        self.layoutH2.addWidget(l_max_time)
        self.layoutH2.addWidget(self.max_timer)



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
        self.layout().addWidget(btn)
        self.layout().addWidget(self.btn2)
        # self.layout.addSpacing(10)
        self.layout().addSpacerItem(layoutSpacer)

        file_browse = QPushButton('Browse')
        file_browse.clicked.connect(self.open_file_dialog)
        self.filename_edit = QLineEdit()
        self.filename_edit.setText(self._get_open_filename()+"_Spots.csv")
        grid_layout = QGridLayout()
        grid_layout.addWidget(QLabel('File:'), 0, 0)
        grid_layout.addWidget(self.filename_edit, 0, 1)
        grid_layout.addWidget(file_browse, 0 ,2)
        self.layout().addLayout(grid_layout)
        self.layout().addWidget(save_btn)

        

        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.reordered.connect(self._refresh_layers)
        # self._connect_layer()
    
    def _get_open_filename(self):
        from napari.utils import history
        _last_folder = history.get_open_history()[0]
        for i in range(len(self.viewer.layers)-1,-1,-1):
            if self.viewer.layers[i]._type_string == 'image':
                _filename = self.viewer.layers[i].name.split(" :: ")[0]
                _filename = _last_folder +"/"+ _filename
                break
        return _filename
    
    def _populate_layers(self):
        # self.layersbox.clear()
        for layer in self.viewer.layers:
            if layer._type_string == 'image':
                self.layersbox.addItem(layer.name)

    def _refresh_layers(self):
        i = self.layersbox.currentIndex()
        self.layersbox.clear()
        for layer in self.viewer.layers:
            if layer._type_string == 'image':
                self.layersbox.addItem(layer.name)
        self.layersbox.setCurrentIndex(i)

    # def _connect_layer(self):
    #     self.viewer.layers.events.changed.connect(self._populate_layers)

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
        
    def _get_choice_layer(self,_widget):    
        for j,layer in enumerate(self.viewer.layers):
            if layer.name == _widget.currentText():
                index_layer = j
                break
        return index_layer
    
    def make_masks(self):
        import pandas as pd
        index_layer = self._get_choice_layer(self.layersbox)

        if len(self.viewer.layers[0].data.shape) <= 3:
            df = pd.DataFrame(self.viewer.layers.selection.active.data, columns = ['frame','y','x'])
        elif len(self.viewer.layers[0].data.shape) > 3:
            df = pd.DataFrame(self.viewer.layers.selection.active.data, columns = ['frame','z','y','x'])
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
            mask_temp, idx_temp = make_labels_trackpy_links(
                self.viewer.layers[index_layer].data[i],
                # temp,size=self.diameter_input.value()-1)
                temp,size=5-1)
        #     print(mask_temp.max(),len(temp.index))
        #     idx.append(idx_temp)
            mask_fixed = np.copy(mask_temp)

            ##this is needed when doing from links because all labels set as 'particles'
            # for j in np.unique(mask_temp).astype('int'):
            #     if j != 0:
            #         # mask_fixed[mask_temp == j] = temp.iloc[j-1]['particle'].astype('int')
            #         ##manbearpig
            #         mask_fixed[mask_temp == j] = temp.iloc[j-1][:].astype('int')
        #     print(np.unique(mask_fixed),temp['particle'].unique())
            masks[i,...] = mask_fixed

        return masks
    
    # @thread_worker
    def _on_click(self):
        index_layer = self._get_choice_layer(self.layersbox)
        print("Detecting points on layer:",index_layer)
        #if its time or Z
        #locating steps
        if len(self.viewer.layers[index_layer].data.shape) >= 3:
            print("Detected more than 3 dimensions")
            if self.choice.isChecked():
                print("Detected a Time lapse TYX or TZYX image")
                a = self.viewer.layers[index_layer].data[self.min_timer.value():self.max_timer.value()]
                self.f = tp.batch(a,self.diameter_input.value(),minmass=self.mass_slider.value(),
                                engine="numba",
                                #   processes=1,
                                )
                #TODO
                #if min is not 0 we have to adjust F to bump it up
            else:
                #there's possibility of improvement here. I did scale == 1 because I am assuming
                #that the time index is scaled at 1
                #however if there's a 1um Z stack it will bug
                if self.viewer.layers[index_layer].scale[0] != 1:
                    print("Detected a ZYX image")
                    self.f = tp.locate(self.viewer.layers[index_layer].data,self.diameter_input.value(),minmass=self.mass_slider.value())
                    self.f['frame'] = 0
                else:
                    print("Detected a Time lapse ZYX  image")
                    _time_locator = self.viewer.dims.current_step[0]
                    self.f = tp.locate(self.viewer.layers[index_layer].data[_time_locator],self.diameter_input.value(),minmass=self.mass_slider.value())
                    self.f['frame'] = _time_locator
        elif len(self.viewer.layers[index_layer].data.shape) == 2:
            print("Detected only YX")
            self.f = tp.locate(self.viewer.layers[index_layer].data,self.diameter_input.value(),minmass=self.mass_slider.value())
            self.f['frame'] = 0
                #TODO
        
        #filtering steps
        if len(self.viewer.layers[index_layer].data.shape) <= 3:
            if self.ecc_tick.isChecked():
                    self.f = self.f[ (self.f['ecc'] < self.ecc_input.value())
                    ]
        if self.size_filter_tick.isChecked():
                self.f = self.f[ (self.f['size'] < self.size_filter_input.value())
                   ]
        
        #transforming data to pandas ready for spots
        if len(self.viewer.layers[index_layer].data.shape) <= 3:
            #XYZ
            if self.viewer.layers[index_layer].scale[0] != 1:
                _points = self.f.loc[:,['frame','z','y','x']]
            #TYX PRJ
            else:    
                _points = self.f.loc[:,['frame','y','x']]
        #TZYX
        elif len(self.viewer.layers[index_layer].data.shape) > 3:
            _points = self.f.loc[:,['frame','z','y','x']]
        _metadata = self.f.loc[:,['mass','size','ecc']]

        self._points_layer = self.viewer.add_points(_points,properties=_metadata,**self.points_options)
        self._points_layer.scale = self.viewer.layers[index_layer].scale

        self.btn2.setEnabled(True)

        _masks = self.make_masks()
        self._masks_layer = self.viewer.add_labels(_masks)

    # @thread_worker
    def _on_click2(self):
        index_layer = self._get_choice_layer(self.layersbox)
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

        
        if len(self.viewer.layers[index_layer].data.shape) <= 3:
            _points = f2.loc[:,['frame','y','x']]
        elif len(self.viewer.layers[index_layer].data.shape) > 3:
            _points = f2.loc[:,['frame','z','y','x']]

        _metadata = f2.loc[:,['mass','size','ecc']]
        self.f2 = f2
        self._points_layer_filter = self.viewer.add_points(_points,properties=_metadata,**self.points_options2)
        self._points_layer_filter.scale = self.viewer.layers[index_layer].scale

    def _save_results(self):
        import pandas as pd
        ##TODO
        ##pull from points layer see example below
        if len(self.viewer.layers[index_layer].data.shape) <= 3:
            df = pd.DataFrame(self.viewer.layers.selection.active.data, columns = ['frame','y','x'])
        elif len(self.viewer.layers[index_layer].data.shape) > 3:
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
        #Selecting
        # self.viewer.layers.events.selecting.connect(self._enable_tracking)

        file_browse = QPushButton('Browse')
        file_browse.clicked.connect(self.open_file_dialog)
        self.filename_edit_links = QLineEdit()
        if len(self.viewer.layers) > 0:
            self.filename_edit_links.setText(self._get_open_filename()+"_Tracks.csv")
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

    def _get_open_filename(self):
        from napari.utils import history
        _last_folder = history.get_open_history()[0]
        for i in range(len(self.viewer.layers)-1,-1,-1):
            if self.viewer.layers[i]._type_string == 'image':
                _filename = self.viewer.layers[i].name.split(" :: ")[0]
                _filename = _last_folder +"/"+ _filename
                break
        return _filename
    
    def _save_results_links(self):
        import pandas as pd
        self.links.to_csv(self.filename_edit_links.text())

    def _enable_tracking(self):
        self.btn.setEnabled(False)
        if len(self.viewer.layers.selection) == 1:
            if self.viewer.layers.selection.active._type_string == 'points':
                #if  self.viewer.layers.selection.active.data == more than one time:
                self.btn.setEnabled(True)

    
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
        #if 2D:
        _tracks = links.loc[:,['particle','frame','y','x']]
        #if 3d:
        # _tracks = links.loc[:,['particle','frame','z','y','x']]

        self.viewer.add_tracks(_tracks,name='trackpy')
        self.links = links
        print(links)

class ColocalizationQWidget(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        l1 = QLabel('Points that are the "anchor"')
        # l1.setText()
        self.points_anchor = QComboBox()
        # self.points_anchor.currentIndexChanged.connect(self._select_layer)
        
        l2 = QLabel('Points that are the "comparison"')
        # l2.setText('Points that are the "comparison"')
        self.points_question = QComboBox()
        # self.points_question.currentIndexChanged.connect(self._select_layer)


        self._populate_layers(self.points_anchor)
        self._populate_layers(self.points_question)
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
        # save_btn = QPushButton("Save current colocalized Spots")
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(l1)
        self.layout().addWidget(self.points_anchor)

        self.layout().addWidget(l2)
        self.layout().addWidget(self.points_question)
        self.layout().addLayout(self.layoutH0)
        self.layout().addWidget(run_btn)


    def calculate_colocalizing(self):
        #makecode from notebooks
        import scipy
        print("Doing Colocalization")
        QuestionPOS = self._get_points(self.points_question)
        print(QuestionPOS)
        AnchorPOS = self._get_points(self.points_anchor)
        print(AnchorPOS)
        question_kd = scipy.spatial.cKDTree(QuestionPOS,leafsize=10000)
        distances_list = np.asarray(question_kd.query(AnchorPOS,p=2,
                #distance_upper_bound=10*psize
                ))[0]
        _colocalizing = distances_list[distances_list < self.euc_distance.value()]
        print(len(_colocalizing))
        l_coloc = QLabel("Number of colocalizing"+" "+len(_colocalizing)+" "+len(AnchorPOS)+" "+len(_colocalizing)/len(AnchorPOS))
        self.layout().addWidget(l_coloc)
        _colocalizing_points = AnchorPOS[distances_list < self.euc_distance.value()]
        self.viewer.add_points(_colocalizing_points)
        # return _colocalizing

    def _populate_layers(self,_widget):
        # self.layersbox.clear()
        for layer in self.viewer.layers:
            if layer._type_string == 'points':
                _widget.addItem(layer.name)

    def _refresh_layers(self):
        i = self.points_anchor.currentIndex()
        self.points_anchor.clear()
        for layer in self.viewer.layers:
            if layer._type_string == 'points':
                self.points_anchor.addItem(layer.name)
        self.points_anchor.setCurrentIndex(i)

        i = self.points_question.currentIndex()
        self.points_question.clear()
        for layer in self.viewer.layers:
            if layer._type_string == 'points':
                self.points_question.addItem(layer.name)
        self.points_question.setCurrentIndex(i)

    # def _select_layer(self,i):
    #     ##needs to be by name
    #     print("Layer to detect:", i)
    #     # self.llayer.setText(

    def _get_choice_layer(self,_widget):    
        for j,layer in enumerate(self.viewer.layers):
            if layer.name == _widget.currentText():
                index_layer = j
                break
        print("Layer where points are is:",j)
        return index_layer
    
    def _get_points(self,_widget):
        import pandas as pd
        _index_layer = self._get_choice_layer(_widget)
        
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
        _index_layer = self._get_choice_layer(_widget)
        
        # self.viewer.layers[0].data
        if len(self.viewer.layers[0].data.shape) < 3:
            df = pd.DataFrame(self.viewer.layers[_index_layer].data, columns = ['frame','y','x'])
            print("2D",df)
        elif len(self.viewer.layers[0].data.shape) >= 3:
            df = pd.DataFrame(self.viewer.layers[ _index_layer].data, columns = ['frame','z','y','x'])
            print("3D",df)

        # b = self.viewer.layers.selection.active.properties
        #error is here somehow now
        # for key in b.keys():
        #     df[key] = b[key]


    
        return df
