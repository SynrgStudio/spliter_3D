from __future__ import annotations

import numpy as np
import pyvista as pv
import vtk
from PyQt6.QtCore import Qt, pyqtSignal
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class VtkViewport(QVTKRenderWindowInteractor):
    facePicked = pyqtSignal(int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.renderer = vtk.vtkRenderer()
        self.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.GetRenderWindow().GetInteractor()
        self.renderer.SetBackground(0.12, 0.12, 0.12)
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.0005)
        self.mesh: pv.PolyData | None = None
        self.actor: vtk.vtkActor | None = None
        self.part_actors: list[vtk.vtkActor] = []
        self.Initialize()

    def set_mesh(self, mesh: pv.PolyData) -> None:
        self.mesh = mesh.copy()
        self.mesh.cell_data["selection"] = np.zeros(self.mesh.n_cells, dtype=np.uint8)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.mesh)
        mapper.SetScalarModeToUseCellData()
        mapper.SelectColorArray("selection")
        mapper.SetScalarRange(0, 2)
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(3)
        lut.SetTableValue(0, 0.78, 0.78, 0.78, 1.0)
        lut.SetTableValue(1, 1.0, 0.58, 0.0, 1.0)
        lut.SetTableValue(2, 0.0, 0.85, 1.0, 1.0)
        lut.Build()
        mapper.SetLookupTable(lut)
        mapper.SetUseLookupTableScalarRange(True)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToPhong()
        if self.actor:
            self.renderer.RemoveActor(self.actor)
        self.actor = actor
        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()
        self.render()

    def update_selection(self, selected: set[int]) -> None:
        if self.mesh is None:
            return
        arr = self.mesh.cell_data["selection"]
        arr[:] = 0
        valid = [i for i in selected if 0 <= i < self.mesh.n_cells]
        if valid:
            arr[valid] = 1
        self.mesh.GetCellData().GetArray("selection").Modified()
        self.render()

    def add_part(self, mesh: pv.PolyData, color=(0.0, 0.85, 1.0)) -> None:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetInterpolationToPhong()
        self.part_actors.append(actor)
        self.renderer.AddActor(actor)
        self.render()

    def clear_parts(self) -> None:
        for actor in self.part_actors:
            self.renderer.RemoveActor(actor)
        self.part_actors.clear()
        self.render()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton and self.actor is not None:
            x, y = event.position().x(), self.height() - event.position().y()
            self.picker.Pick(float(x), float(y), 0, self.renderer)
            if self.picker.GetActor() == self.actor:
                erase = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                self.facePicked.emit(int(self.picker.GetCellId()), erase)
                return
        super().mousePressEvent(event)

    def render(self) -> None:
        self.GetRenderWindow().Render()
