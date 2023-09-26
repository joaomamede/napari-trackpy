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

if TYPE_CHECKING:
    import napari

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
        self.filename_edit.setText("/tmp/test2.csv")
        grid_layout = QGridLayout()
        grid_layout.addWidget(QLabel('File:'), 0, 0)
        grid_layout.addWidget(self.filename_edit, 0, 1)
        grid_layout.addWidget(file_browse, 0 ,2)
        self.layout().addLayout(grid_layout)
        self.layout().addWidget(save_btn)

        
        self._populate_layers()
        self.viewer.layers.events.removed.connect(self._populate_layers)
        self.viewer.layers.events.inserted.connect(self._populate_layers)
        self.viewer.layers.events.reordered.connect(self._populate_layers)
        # self._connect_layer()
    
   
    def _populate_layers(self):
        self.layersbox.clear()
        for layer in self.viewer.layers:
            if layer._type_string == 'image':
                self.layersbox.addItem(layer.name)

    # def _connect_layer(self):
    #     self.viewer.layers.events.changed.connect(self._populate_layers)

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


    def _select_layer(self,i):
        print("Layer to detect:", i, self.layersbox.currentIndex())
        # self.llayer.setText(

    # @thread_worker
    def _on_click(self):
        # print("napari has", len(self.viewer.layers), "layers")
        print("Detecting points on:",self.layersbox.currentIndex())
        if len(self.viewer.layers[0].data.shape) >= 3:
            if self.choice.isChecked():
                a = self.viewer.layers[self.layersbox.currentIndex()].data[self.min_timer.value():self.max_timer.value()]
                self.f = tp.batch(a,self.diameter_input.value(),minmass=self.mass_slider.value(),engine="numba")
                #TODO
                #if min is not 0 we have to adjust F to bump it up
            else:
                _time_locator = self.viewer.dims.current_step[0]
                self.f = tp.locate(self.viewer.layers[self.layersbox.currentIndex()].data[_time_locator],self.diameter_input.value(),minmass=self.mass_slider.value())
                self.f['frame'] = _time_locator
        elif len(self.viewer.layers[0].data.shape) == 2:
            self.f = tp.locate(self.viewer.layers[self.layersbox.currentIndex()].data,self.diameter_input.value(),minmass=self.mass_slider.value())
            self.f['frame'] = 0
                #TODO
        if self.ecc_tick.isChecked():
                self.f = self.f[ (self.f['ecc'] < self.ecc_input.value())
                   ]
        if self.size_filter_tick.isChecked():
                self.f = self.f[ (self.f['ecc'] < self.size_filter_input.value())
                   ]
        
        print(self.f)
        if len(self.viewer.layers[self.layersbox.currentIndex()].data.shape) <= 3:
            _points = self.f.loc[:,['frame','y','x']]
        elif len(self.viewer.layers[self.layersbox.currentIndex()].data.shape) > 3:
            _points = self.f.loc[:,['frame','z','y','x']]
        _metadata = self.f.loc[:,['mass','size','ecc']]

        #add the filters here if check size ..if ecc...
        self._points_layer = self.viewer.add_points(_points,properties=_metadata,**self.points_options)
        self._points_layer.scale = self.viewer.layers[self.layersbox.currentIndex()].scale

        self.btn2.setEnabled(True)

    # @thread_worker
    def _on_click2(self):
        # print("napari has", len(self.viewer.layers), "layers")
        # f = tp.locate(self.viewer.layers[2].data,5,minmass=500)
        f2 = self.f
        #add the filters here if check size ..if ecc...
        if self.ecc_tick.isChecked():
            f2 = f2[ (f2['ecc'] < self.ecc_input.value())
                   ]
        if self.size_filter_tick.isChecked():
            f2 = f2[ (f2['ecc'] < self.size_filter_input.value())
                   ]
        f2 = f2[(self.f['mass'] > self.mass_slider.value())    
                   ]

        
        if len(self.viewer.layers[self.layersbox.currentIndex()].data.shape) <= 3:
            _points = f2.loc[:,['frame','y','x']]
        elif len(self.viewer.layers[self.layersbox.currentIndex()].data.shape) > 3:
            _points = f2.loc[:,['frame','z','y','x']]

        _metadata = f2.loc[:,['mass','size','ecc']]
        self.f2 = f2
        self._points_layer_filter = self.viewer.add_points(_points,properties=_metadata,**self.points_options2)
        self._points_layer_filter.scale = self.viewer.layers[self.layersbox.currentIndex()].scale

    def _save_results(self):
        import pandas as pd
        ##TODO
        ##pull from points layer see example below
        if len(self.viewer.layers[self.layersbox.currentIndex()].data.shape) <= 3:
            df = pd.DataFrame(self.viewer.layers.selection.active.data, columns = ['frame','y','x'])
        elif len(self.viewer.layers[self.layersbox.currentIndex()].data.shape) > 3:
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
        self.filename_edit_links.setText("/tmp/test3.csv")
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