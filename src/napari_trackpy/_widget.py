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
from qtpy.QtWidgets import QLabel, QDoubleSpinBox, QWidget
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
        btn2 = QPushButton("Filter with new settings from already identified")
        btn2.clicked.connect(self._on_click2)

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
        self.layout().addWidget(btn2)
    
    def _populate_layers(self):
        for layer in self.viewer.layers:
            self.layersbox.addItem(layer.name)

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
        
        
        print(self.f)
        if len(self.viewer.layers[0].data.shape) <= 3:
            _points = self.f.loc[:,['frame','y','x']]
        elif len(self.viewer.layers[0].data.shape) > 3:
            _points = self.f.loc[:,['frame','z','y','x']]
        _metadata = self.f.loc[:,['mass','size','ecc']]
        #TODO
        if self.ecc_tick.isChecked():
                self.f = self.f[ (self.f['ecc'] < self.ecc_input.value())
                   ]
        if self.size_filter_tick.isChecked():
                self.f = self.f[ (self.f['ecc'] < self.size_filter_input.value())
                   ]
        #add the filters here if check size ..if ecc...
        self._points_layer = self.viewer.add_points(_points,properties=_metadata,**self.points_options)
        self._points_layer.scale = self.viewer.layers[self.layersbox.currentIndex()].scale

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

        if len(self.viewer.layers[0].data.shape) <= 3:
            _points = f2.loc[:,['frame','y','x']]
        elif len(self.viewer.layers[0].data.shape) > 3:
            _points = f2.loc[:,['frame','z','y','x']]

        _metadata = f2.loc[:,['mass','size','ecc']]
        self._points_layer_filter = self.viewer.add_points(_points,properties=_metadata,**self.points_options2)
        self._points_layer_filter.scale = self.viewer.layers[self.layersbox.currentIndex()].scale





# _frame = 1 #@param {type:"integer"}
# #@markdown ---
# #@markdown ### Enter the intensity cut
# _mass = 500#@param {type:"slider", min:0, max:1e5, step:100}

# #@markdown ---
# #@markdown ### Enter the diameter of the particle expected
# _diam = 5 #@param {type:"slider", min:3, max:19, step:2}

# f = tp.batch(a,_diam,minmass=_mass,
# #              processes=-1,
#              engine='numba'
#             )


# f = f[((f['mass'] > _mass) & (f['size'] < 2.25)
# #        & (f['mass'] < 4e6)
#              &          (f['ecc'] < 0.5)
#             )
#            ]


@magic_factory
def example_magic_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
def example_function_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")
