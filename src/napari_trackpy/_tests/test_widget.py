import importlib

import pytest


def _widget_module():
    pytest.importorskip("napari")
    pytest.importorskip("qtpy")
    return importlib.import_module("napari_trackpy._widget")


def test_add_points_compat_maps_edge_to_border():
    mod = _widget_module()

    class ViewerBorder:
        def add_points(self, data, *, border_width=None, border_color=None, **kwargs):
            return {
                "data": data,
                "border_width": border_width,
                "border_color": border_color,
                **kwargs,
            }

    out = mod._add_points_compat(
        ViewerBorder(),
        [[1, 2]],
        edge_width=0.5,
        edge_color="red",
        size=3,
    )
    assert out["border_width"] == 0.5
    assert out["border_color"] == "red"
    assert out["size"] == 3


def test_add_points_compat_maps_border_to_edge():
    mod = _widget_module()

    class ViewerEdge:
        def add_points(self, data, *, edge_width=None, edge_color=None, **kwargs):
            return {
                "data": data,
                "edge_width": edge_width,
                "edge_color": edge_color,
                **kwargs,
            }

    out = mod._add_points_compat(
        ViewerEdge(),
        [[1, 2]],
        border_width=0.25,
        border_color="green",
        size=4,
    )
    assert out["edge_width"] == 0.25
    assert out["edge_color"] == "green"
    assert out["size"] == 4


def test_get_choice_layer_returns_selected_layer():
    mod = _widget_module()

    class Layer:
        def __init__(self, name):
            self.name = name

    class Viewer:
        def __init__(self):
            self.layers = [Layer("A"), Layer("B")]

    class Combo:
        def currentText(self):
            return "B"

        def currentIndex(self):
            return 1

    class Obj:
        def __init__(self):
            self.viewer = Viewer()

    assert mod._get_choice_layer(Obj(), Combo()) == 1


def test_get_choice_layer_raises_for_invalid_selection():
    mod = _widget_module()

    class Layer:
        def __init__(self, name):
            self.name = name

    class Viewer:
        def __init__(self):
            self.layers = [Layer("A"), Layer("B")]

    class Combo:
        def currentText(self):
            return "missing"

        def currentIndex(self):
            return 99

    class Obj:
        def __init__(self):
            self.viewer = Viewer()

    with pytest.raises(ValueError):
        mod._get_choice_layer(Obj(), Combo())
