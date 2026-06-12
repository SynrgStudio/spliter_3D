import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_API", "pyqt6")

# VTK/PyVista currently uses vtkXOpenGLRenderWindow, which is more reliable
# through XCB/XWayland than native Wayland. Some Linux desktops export
# QT_QPA_PLATFORM=wayland;xcb, so force xcb by default before Qt is imported.
# For troubleshooting, override with SPLIT3R_QT_QPA_PLATFORM=wayland/xcb/offscreen.
if sys.platform.startswith("linux"):
    os.environ["QT_QPA_PLATFORM"] = os.environ.get("SPLIT3R_QT_QPA_PLATFORM", "xcb")
else:
    os.environ.setdefault("QT_OPENGL", "desktop")

import numpy as np
import pyvista as pv
import vtk
from matplotlib.colors import ListedColormap
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDockWidget,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from scipy.spatial import KDTree

from app.extraction import extract_plug_socket
from app.mesh_io import load_trimesh, polydata_to_trimesh, trimesh_to_polydata, validate_polydata
from app.selection import build_adjacency_dict, compute_smart_shell_region

LOG_PATH = Path("app_log.txt")
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("split3r_clone")


class CustomQtInteractor(QtInteractor):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.main_window = parent
        self.setMouseTracking(True)
        self.drag_actor = None
        self.last_mouse_pos = None
        self.last_vtk_pos = (0, 0)
        self.last_update_time = 0

    def mousePressEvent(self, ev):
        vtk_y = self.height() - ev.pos().y()
        self.last_vtk_pos = (ev.pos().x(), vtk_y)

        if self.main_window and self.main_window.btn_move.isChecked() and ev.button() == Qt.MouseButton.LeftButton:
            _, _, picked_actor = self.main_window.get_ray_intersection_with_actor(self.last_vtk_pos)
            if picked_actor and picked_actor != self.main_window.main_actor_vtk:
                self.drag_actor = picked_actor
                self.last_mouse_pos = self.last_vtk_pos
                return

        if ev.button() == Qt.MouseButton.RightButton:
            if self.main_window and not self.main_window.btn_move.isChecked():
                self.main_window.is_painting = True
                self.main_window.is_erasing = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                self.main_window.on_mesh_interaction(self.last_vtk_pos)
            return
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self.drag_actor:
            self.drag_actor = None
            self.last_mouse_pos = None
            return
        if ev.button() == Qt.MouseButton.RightButton:
            if self.main_window:
                self.main_window.is_painting = False
                self.main_window.is_erasing = False
            return
        super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev):
        vtk_y = self.height() - ev.pos().y()
        current_pos = (ev.pos().x(), vtk_y)
        self.last_vtk_pos = current_pos

        if self.drag_actor and self.last_mouse_pos:
            renderer = self.main_window.plotter.renderer
            camera = renderer.GetActiveCamera()
            focal_point = camera.GetFocalPoint()
            renderer.SetWorldPoint(focal_point[0], focal_point[1], focal_point[2], 1.0)
            renderer.WorldToDisplay()
            display_z = renderer.GetDisplayPoint()[2]

            renderer.SetDisplayPoint(self.last_mouse_pos[0], self.last_mouse_pos[1], display_z)
            renderer.DisplayToWorld()
            world_last = np.array(renderer.GetWorldPoint()[:3])

            renderer.SetDisplayPoint(current_pos[0], current_pos[1], display_z)
            renderer.DisplayToWorld()
            world_curr = np.array(renderer.GetWorldPoint()[:3])

            translation = world_curr - world_last
            curr_pos = np.array(self.drag_actor.GetPosition())
            self.drag_actor.SetPosition(*(curr_pos + translation))
            self.last_mouse_pos = current_pos
            self.render()
            return

        now = time.time()
        if now - self.last_update_time > 0.033:
            if self.main_window and not self.main_window.btn_move.isChecked():
                self.main_window.is_erasing = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                self.main_window.on_mouse_move(current_pos)
            self.last_update_time = now
        super().mouseMoveEvent(ev)

    def wheelEvent(self, ev):
        if self.main_window and bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
            delta = ev.angleDelta().y()
            if self.main_window.radio_sphere.isChecked():
                current_val = self.main_window.slider_brush.value()
                self.main_window.slider_brush.setValue(min(100, current_val + 2) if delta > 0 else max(1, current_val - 2))
            elif self.main_window.radio_bucket.isChecked():
                current_val = self.main_window.slider_smart.value()
                self.main_window.slider_smart.setValue(min(90, current_val + 2) if delta > 0 else max(1, current_val - 2))
            return
        super().wheelEvent(ev)


class Split3rClone(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Split3r Clone - Plug & Socket")
        self.resize(1280, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        self.plotter = CustomQtInteractor(self)
        self.layout.addWidget(self.plotter.interactor)
        self.plotter.add_axes()
        self.plotter.set_background("#1e1e1e")
        self.plotter.enable_terrain_style()

        self.current_mesh = None
        self.t_mesh = None
        self.adj_dict = {}
        self.selected_cells = set()
        self.undo_stack = []

        self.brush_radius = 5.0
        self.smart_angle = 30.0
        self.extrude_depth = 2.0
        self.last_hover_face = -1

        self.main_actor_vtk = None
        self.cursor_actor = None
        self.cursor_mesh = pv.Sphere(radius=1.0)
        self.cell_picker = vtk.vtkCellPicker()
        self.extracted_parts = []

        self.init_menu_bar()
        self.init_ui_panel()
        logger.info("Application started")

    def init_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Archivo")

        import_action = QAction("Importar STL/OBJ/3MF...", self)
        import_action.setShortcut("Ctrl+O")
        import_action.triggered.connect(self.load_model)
        file_menu.addAction(import_action)

        export_plug_action = QAction("Exportar último plug...", self)
        export_plug_action.triggered.connect(self.export_last_plug)
        file_menu.addAction(export_plug_action)

        export_body_action = QAction("Exportar body/socket...", self)
        export_body_action.triggered.connect(self.export_body)
        file_menu.addAction(export_body_action)

        exit_action = QAction("Salir", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def init_ui_panel(self):
        self.dock = QDockWidget("Engineering Terminal", self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        self.panel_widget = QWidget()
        self.panel_layout = QVBoxLayout(self.panel_widget)

        self.lbl_info = QLabel("System Ready")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color: #ffffff; font-family: monospace; background-color: #333333; padding: 5px;")
        self.panel_layout.addWidget(self.lbl_info)

        self.lbl_selection = QLabel("Selected faces: 0")
        self.panel_layout.addWidget(self.lbl_selection)

        group_smart = QGroupBox("Smart Shell (Bambu Style)")
        layout_smart = QVBoxLayout(group_smart)
        self.radio_bucket = QRadioButton("Enable Smart Shell")
        self.radio_bucket.setChecked(True)
        layout_smart.addWidget(self.radio_bucket)
        self.lbl_smart = QLabel("Tolerance Angle: 30°")
        layout_smart.addWidget(self.lbl_smart)
        self.slider_smart = QSlider(Qt.Orientation.Horizontal)
        self.slider_smart.setMinimum(1)
        self.slider_smart.setMaximum(90)
        self.slider_smart.setValue(30)
        self.slider_smart.valueChanged.connect(self.update_smart_size)
        layout_smart.addWidget(self.slider_smart)
        self.panel_layout.addWidget(group_smart)

        group_brush = QGroupBox("Sphere Brush (Manual)")
        layout_brush = QVBoxLayout(group_brush)
        self.radio_sphere = QRadioButton("Enable Sphere Brush")
        layout_brush.addWidget(self.radio_sphere)
        self.lbl_brush = QLabel("Brush Size: 5")
        layout_brush.addWidget(self.lbl_brush)
        self.slider_brush = QSlider(Qt.Orientation.Horizontal)
        self.slider_brush.setMinimum(1)
        self.slider_brush.setMaximum(100)
        self.slider_brush.setValue(5)
        self.slider_brush.setEnabled(False)
        self.slider_brush.valueChanged.connect(self.update_brush_size)
        layout_brush.addWidget(self.slider_brush)
        self.panel_layout.addWidget(group_brush)

        group_3d = QGroupBox("3D PRINTING PARAMS")
        layout_3d = QVBoxLayout(group_3d)
        self.lbl_depth = QLabel("Grosor de Pieza: 2.0mm")
        layout_3d.addWidget(self.lbl_depth)
        self.slider_depth = QSlider(Qt.Orientation.Horizontal)
        self.slider_depth.setMinimum(1)
        self.slider_depth.setMaximum(100)
        self.slider_depth.setValue(20)
        self.slider_depth.valueChanged.connect(self.update_depth)
        layout_3d.addWidget(self.slider_depth)
        self.panel_layout.addWidget(group_3d)

        self.brush_group = QButtonGroup()
        self.brush_group.addButton(self.radio_bucket)
        self.brush_group.addButton(self.radio_sphere)
        self.radio_sphere.toggled.connect(lambda: self.slider_brush.setEnabled(self.radio_sphere.isChecked()))
        self.radio_bucket.toggled.connect(lambda: self.slider_smart.setEnabled(self.radio_bucket.isChecked()))

        self.btn_clear = QPushButton("Clear Selection")
        self.btn_clear.clicked.connect(self.clear_selection)
        self.panel_layout.addWidget(self.btn_clear)

        self.btn_invert = QPushButton("Invert Selection")
        self.btn_invert.clicked.connect(self.invert_selection)
        self.btn_invert.setEnabled(False)
        self.panel_layout.addWidget(self.btn_invert)

        self.btn_extract = QPushButton("EXTRACT PLUG & SOCKET")
        self.btn_extract.clicked.connect(self.extract_part)
        self.btn_extract.setEnabled(False)
        self.btn_extract.setMinimumHeight(50)
        self.btn_extract.setStyleSheet("background-color: #004444; color: white; font-weight: bold;")
        self.panel_layout.addWidget(self.btn_extract)

        self.btn_undo = QPushButton("Undo Last Extraction")
        self.btn_undo.clicked.connect(self.undo_last_extraction)
        self.btn_undo.setEnabled(False)
        self.panel_layout.addWidget(self.btn_undo)

        self.btn_move = QPushButton("Enable Move Mode (Drag)")
        self.btn_move.setCheckable(True)
        self.btn_move.clicked.connect(self.toggle_move_mode)
        self.btn_move.setEnabled(False)
        self.panel_layout.addWidget(self.btn_move)

        self.panel_layout.addStretch()
        self.dock.setWidget(self.panel_widget)

    def set_status(self, message, level=logging.INFO):
        self.lbl_info.setText(message)
        logger.log(level, message)

    def update_brush_size(self, val):
        self.brush_radius = float(val)
        self.lbl_brush.setText(f"Brush Size: {val}")
        if self.cursor_actor:
            self.cursor_actor.SetScale(self.brush_radius, self.brush_radius, self.brush_radius)
        self.refresh_hover()

    def update_smart_size(self, val):
        self.smart_angle = float(val)
        self.lbl_smart.setText(f"Tolerance Angle: {val}°")
        self.last_hover_face = -1
        self.refresh_hover()

    def update_depth(self, val):
        self.extrude_depth = float(val) / 10.0
        self.lbl_depth.setText(f"Grosor de Pieza: {self.extrude_depth:.1f}mm")

    def refresh_hover(self):
        if hasattr(self.plotter, "last_vtk_pos"):
            self.on_mouse_move(self.plotter.last_vtk_pos)

    def toggle_move_mode(self, checked):
        self.btn_move.setText("Disable Move Mode" if checked else "Enable Move Mode (Drag)")

    def load_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open 3D Model", "", "3D Files (*.stl *.obj *.3mf);;All Files (*.*)")
        if file_name:
            self.load_file(file_name)

    def load_file(self, filepath):
        try:
            self.set_status("Building topological map...")
            QApplication.processEvents()
            self.t_mesh = load_trimesh(filepath)
            self.current_mesh = trimesh_to_polydata(self.t_mesh)
            self.rebuild_topology()
            self.plotter.clear()
            self.extracted_parts = []
            self.undo_stack = []
            self.selected_cells = set()
            self.add_main_mesh_actor(reset_camera=True)
            self.set_status(f"SUCCESS: Model Loaded. Faces: {self.current_mesh.n_cells}")
        except Exception as e:
            logger.exception("Load failed")
            self.set_status(f"Load Error: {e}. Ver {LOG_PATH}", logging.ERROR)

    def rebuild_topology(self):
        validate_polydata(self.current_mesh)
        self.t_mesh = polydata_to_trimesh(self.current_mesh)
        self.adj_dict = build_adjacency_dict(self.t_mesh.face_adjacency, self.t_mesh.face_adjacency_angles)
        self.current_mesh.cell_data["Selection"] = np.zeros(self.current_mesh.n_cells)
        self.current_mesh.cell_data["Hover"] = np.zeros(self.current_mesh.n_cells)
        self.kdtree = KDTree(self.current_mesh.cell_centers().points)
        self.locator = vtk.vtkCellLocator()
        self.locator.SetDataSet(self.current_mesh)
        self.locator.BuildLocator()
        self.update_action_state()

    def add_main_mesh_actor(self, reset_camera=False):
        cmap = ListedColormap(["lightgray", "#55ff55", "orange", "orange"])
        if self.main_actor_vtk:
            self.plotter.remove_actor(self.main_actor_vtk)
        self.main_actor_vtk = self.plotter.add_mesh(
            self.current_mesh,
            scalars="Selection",
            cmap=cmap,
            clim=[0, 3],
            smooth_shading=True,
            pbr=True,
            show_scalar_bar=False,
            interpolate_before_map=False,
        )
        self.plotter.enable_eye_dome_lighting()
        if reset_camera:
            self.plotter.reset_camera()

    def get_ray_intersection_with_actor(self, pos):
        self.cell_picker.Pick(pos[0], pos[1], 0, self.plotter.renderer)
        picked_actor = self.cell_picker.GetActor()
        if picked_actor:
            return self.cell_picker.GetPickPosition(), self.cell_picker.GetCellId(), picked_actor
        return None, -1, None

    def get_ray_intersection(self, pos):
        hit_pos, cid, actor = self.get_ray_intersection_with_actor(pos)
        if actor == self.main_actor_vtk:
            return hit_pos, cid
        return None, -1

    def compute_smart_shell_region(self, seed_face):
        return compute_smart_shell_region(seed_face, self.adj_dict, self.smart_angle)

    def on_mouse_move(self, pos):
        if self.current_mesh is None:
            return
        hit_pos, cid = self.get_ray_intersection(pos)
        if self.radio_sphere.isChecked():
            if hit_pos:
                if not self.cursor_actor:
                    self.cursor_actor = self.plotter.add_mesh(self.cursor_mesh, color="red", style="wireframe", opacity=0.5, pickable=False)
                self.cursor_actor.SetVisibility(True)
                self.cursor_actor.SetPosition(hit_pos)
                self.cursor_actor.SetScale(self.brush_radius, self.brush_radius, self.brush_radius)
                self.current_mesh.cell_data["Hover"][:] = 0
                self.plotter.render()
                if getattr(self, "is_painting", False):
                    self.on_mesh_interaction(pos)
            else:
                if self.cursor_actor:
                    self.cursor_actor.SetVisibility(False)
                self.current_mesh.cell_data["Hover"][:] = 0
                self.update_visuals()
            return

        if self.cursor_actor:
            self.cursor_actor.SetVisibility(False)
        if hit_pos and cid != -1:
            if getattr(self, "is_painting", False):
                self.on_mesh_interaction(pos)
            elif cid != self.last_hover_face:
                self.last_hover_face = cid
                hover_indices = self.compute_smart_shell_region(cid)
                self.current_mesh.cell_data["Hover"][:] = 0
                self.current_mesh.cell_data["Hover"][hover_indices] = 1
                self.update_visuals()
        elif self.last_hover_face != -1:
            self.last_hover_face = -1
            self.current_mesh.cell_data["Hover"][:] = 0
            self.update_visuals()

    def on_mesh_interaction(self, pos):
        hit_pos, cid = self.get_ray_intersection(pos)
        if hit_pos and cid != -1:
            if self.radio_bucket.isChecked():
                indices = self.compute_smart_shell_region(cid)
            else:
                indices = self.kdtree.query_ball_point(hit_pos, r=self.brush_radius)
            if getattr(self, "is_erasing", False):
                self.selected_cells.difference_update(indices)
            else:
                self.selected_cells.update(indices)
            self.update_visuals()

    def update_visuals(self):
        if self.current_mesh is not None:
            selection_array = self.current_mesh.cell_data["Selection"]
            selection_array[:] = 0
            hover_array = self.current_mesh.cell_data["Hover"]
            selection_array[hover_array == 1] = 1
            valid_selected = [c for c in self.selected_cells if c < self.current_mesh.n_cells]
            if valid_selected:
                selection_array[valid_selected] += 2
            self.current_mesh.GetCellData().GetArray("Selection").Modified()
            self.plotter.render()
            self.update_action_state()

    def update_action_state(self):
        selected = len(self.selected_cells)
        has_mesh = self.current_mesh is not None
        self.lbl_selection.setText(f"Selected faces: {selected}")
        self.btn_extract.setEnabled(selected > 0 and has_mesh)
        self.btn_invert.setEnabled(has_mesh)
        self.btn_undo.setEnabled(bool(self.undo_stack))

    def clear_selection(self):
        self.selected_cells = set()
        self.update_visuals()

    def invert_selection(self):
        if self.current_mesh is None:
            return
        self.selected_cells = set(range(self.current_mesh.n_cells)) - self.selected_cells
        self.update_visuals()

    def extract_part(self):
        try:
            self.set_status(f"Building SOLID Plug & Socket ({self.extrude_depth}mm)...")
            QApplication.processEvents()
            self.undo_stack.append((self.current_mesh.copy(), list(self.extracted_parts)))
            plug_mesh, body_mesh = extract_plug_socket(self.current_mesh, self.selected_cells, self.extrude_depth)
            self.extracted_parts.append(plug_mesh)
            self.plotter.add_mesh(plug_mesh, color="cyan", pbr=True, name=f"Part_{len(self.extracted_parts)}")
            self.current_mesh = body_mesh
            self.selected_cells = set()
            self.rebuild_topology()
            self.add_main_mesh_actor(reset_camera=False)
            self.btn_move.setEnabled(True)
            self.set_status(f"SUCCESS: Solid Plug & Socket created ({self.extrude_depth}mm).")
        except Exception as e:
            if self.undo_stack:
                self.undo_stack.pop()
            logger.exception("Extraction failed")
            self.set_status(f"Extraction Error: {e}. Ver {LOG_PATH}", logging.ERROR)

    def undo_last_extraction(self):
        if not self.undo_stack:
            return
        self.current_mesh, self.extracted_parts = self.undo_stack.pop()
        self.selected_cells = set()
        self.rebuild_topology()
        self.plotter.clear()
        self.add_main_mesh_actor(reset_camera=False)
        for idx, part in enumerate(self.extracted_parts, start=1):
            self.plotter.add_mesh(part, color="cyan", pbr=True, name=f"Part_{idx}")
        self.set_status("Undo complete.")

    def export_mesh(self, mesh, title):
        if mesh is None:
            self.set_status("No hay malla para exportar.", logging.WARNING)
            return
        file_name, _ = QFileDialog.getSaveFileName(self, title, "", "STL (*.stl);;OBJ (*.obj);;PLY (*.ply);;All Files (*.*)")
        if not file_name:
            return
        try:
            mesh.save(file_name)
            self.set_status(f"Exportado: {file_name}")
        except Exception as e:
            logger.exception("Export failed")
            self.set_status(f"Export Error: {e}. Ver {LOG_PATH}", logging.ERROR)

    def export_last_plug(self):
        self.export_mesh(self.extracted_parts[-1] if self.extracted_parts else None, "Exportar último plug")

    def export_body(self):
        self.export_mesh(self.current_mesh, "Exportar body/socket")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Split3rClone()
    window.show()
    sys.exit(app.exec())
