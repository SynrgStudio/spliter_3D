from __future__ import annotations

import math

import pyvista as pv
import vtk
from PyQt6.QtCore import Qt, pyqtSignal
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class VtkViewport(QVTKRenderWindowInteractor):
    """Small VTK viewport tailored for the rewrite app.

    Controls:
    - LMB click: smart-select face region
    - Ctrl + LMB click: erase face region
    - LMB drag: rotate camera
    - RMB drag: pan camera
    - Mouse wheel: zoom
    """

    facePicked = pyqtSignal(int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.renderer = vtk.vtkRenderer()
        self.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(0.12, 0.12, 0.12)
        self.renderer.SetTwoSidedLighting(True)
        self.renderer.UseFXAAOn()
        self._setup_lights()

        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.0005)
        self.mesh: pv.PolyData | None = None
        self.actor: vtk.vtkActor | None = None
        self.selection_actor: vtk.vtkActor | None = None
        self.part_actors: list[vtk.vtkActor] = []

        self._pressed_button: Qt.MouseButton | None = None
        self._press_pos: tuple[float, float] | None = None
        self._last_pos: tuple[float, float] | None = None
        self._press_modifiers = Qt.KeyboardModifier.NoModifier

        self.Initialize()

    def _setup_lights(self) -> None:
        self.renderer.RemoveAllLights()

        headlight = vtk.vtkLight()
        headlight.SetLightTypeToHeadlight()
        headlight.SetIntensity(0.9)
        self.renderer.AddLight(headlight)

        fill = vtk.vtkLight()
        fill.SetLightTypeToSceneLight()
        fill.SetPosition(1.0, -1.0, 1.0)
        fill.SetFocalPoint(0.0, 0.0, 0.0)
        fill.SetIntensity(0.45)
        self.renderer.AddLight(fill)

    def set_mesh(self, mesh: pv.PolyData) -> None:
        self.mesh = mesh.extract_surface(algorithm="dataset_surface").triangulate().clean()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.mesh)
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(0.78, 0.78, 0.74)
        prop.SetAmbient(0.28)
        prop.SetDiffuse(0.72)
        prop.SetSpecular(0.18)
        prop.SetSpecularPower(24.0)
        prop.SetInterpolationToPhong()
        prop.BackfaceCullingOff()
        prop.FrontfaceCullingOff()

        if self.actor:
            self.renderer.RemoveActor(self.actor)
        if self.selection_actor:
            self.renderer.RemoveActor(self.selection_actor)
            self.selection_actor = None
        self.actor = actor
        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self.render()

    def update_selection(self, selected: set[int]) -> None:
        if self.mesh is None:
            return
        if self.selection_actor:
            self.renderer.RemoveActor(self.selection_actor)
            self.selection_actor = None

        valid = [i for i in selected if 0 <= i < self.mesh.n_cells]
        if valid:
            selected_mesh = self.mesh.extract_cells(valid).extract_surface(algorithm="dataset_surface").triangulate()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(selected_mesh)
            mapper.ScalarVisibilityOff()

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.PickableOff()
            prop = actor.GetProperty()
            prop.SetColor(1.0, 0.55, 0.0)
            prop.SetAmbient(0.35)
            prop.SetDiffuse(0.80)
            prop.SetSpecular(0.08)
            prop.SetOpacity(0.92)
            prop.SetInterpolationToPhong()
            self.selection_actor = actor
            self.renderer.AddActor(actor)
        self.render()

    def add_part(self, mesh: pv.PolyData, color=(0.0, 0.85, 1.0)) -> None:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh.extract_surface(algorithm="dataset_surface").triangulate())
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(*color)
        prop.SetAmbient(0.25)
        prop.SetDiffuse(0.75)
        prop.SetSpecular(0.15)
        prop.SetInterpolationToPhong()
        self.part_actors.append(actor)
        self.renderer.AddActor(actor)
        self.render()

    def clear_parts(self) -> None:
        for actor in self.part_actors:
            self.renderer.RemoveActor(actor)
        self.part_actors.clear()
        self.render()

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            pos = (float(event.position().x()), float(event.position().y()))
            self._pressed_button = event.button()
            self._press_pos = pos
            self._last_pos = pos
            self._press_modifiers = event.modifiers()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pressed_button is None or self._last_pos is None:
            super().mouseMoveEvent(event)
            return

        pos = (float(event.position().x()), float(event.position().y()))
        dx = pos[0] - self._last_pos[0]
        dy = pos[1] - self._last_pos[1]
        self._last_pos = pos

        if self._pressed_button == Qt.MouseButton.LeftButton:
            self._rotate_camera(dx, dy)
        elif self._pressed_button == Qt.MouseButton.RightButton:
            self._pan_camera(dx, dy)
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._pressed_button == event.button():
            release_pos = (float(event.position().x()), float(event.position().y()))
            if self._pressed_button == Qt.MouseButton.LeftButton and self._is_click(release_pos):
                self._pick(release_pos, erase=bool(self._press_modifiers & Qt.KeyboardModifier.ControlModifier))
            self._pressed_button = None
            self._press_pos = None
            self._last_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.12 if delta > 0 else 1.0 / 1.12
        camera = self.renderer.GetActiveCamera()
        camera.Dolly(factor)
        self.renderer.ResetCameraClippingRange()
        self.render()
        event.accept()

    def _is_click(self, release_pos: tuple[float, float]) -> bool:
        if self._press_pos is None:
            return False
        dist = math.hypot(release_pos[0] - self._press_pos[0], release_pos[1] - self._press_pos[1])
        return dist <= 4.0

    def _pick(self, pos: tuple[float, float], erase: bool) -> None:
        if self.actor is None:
            return
        x, y = pos[0], self.height() - pos[1]
        self.picker.Pick(float(x), float(y), 0, self.renderer)
        if self.picker.GetActor() == self.actor:
            self.facePicked.emit(int(self.picker.GetCellId()), erase)

    def _rotate_camera(self, dx: float, dy: float) -> None:
        camera = self.renderer.GetActiveCamera()
        camera.Azimuth(-dx * 0.45)
        camera.Elevation(dy * 0.45)
        camera.OrthogonalizeViewUp()
        self.renderer.ResetCameraClippingRange()
        self.render()

    def _pan_camera(self, dx: float, dy: float) -> None:
        renderer = self.renderer
        camera = renderer.GetActiveCamera()
        focal = camera.GetFocalPoint()
        renderer.SetWorldPoint(focal[0], focal[1], focal[2], 1.0)
        renderer.WorldToDisplay()
        display_z = renderer.GetDisplayPoint()[2]

        width = max(self.width(), 1)
        height = max(self.height(), 1)
        last_x = width / 2.0
        last_y = height / 2.0

        renderer.SetDisplayPoint(last_x, last_y, display_z)
        renderer.DisplayToWorld()
        world_start = renderer.GetWorldPoint()
        world_start = [world_start[i] / world_start[3] for i in range(3)]

        renderer.SetDisplayPoint(last_x - dx, last_y + dy, display_z)
        renderer.DisplayToWorld()
        world_end = renderer.GetWorldPoint()
        world_end = [world_end[i] / world_end[3] for i in range(3)]

        motion = [world_end[i] - world_start[i] for i in range(3)]
        pos = camera.GetPosition()
        fp = camera.GetFocalPoint()
        camera.SetPosition(pos[0] + motion[0], pos[1] + motion[1], pos[2] + motion[2])
        camera.SetFocalPoint(fp[0] + motion[0], fp[1] + motion[1], fp[2] + motion[2])
        self.renderer.ResetCameraClippingRange()
        self.render()

    def render(self) -> None:
        self.GetRenderWindow().Render()
