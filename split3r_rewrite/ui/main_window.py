from __future__ import annotations

from pathlib import Path

import pyvista as pv
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..geometry.insert import InsertResult, create_insert
from ..geometry.mesh_io import load_polydata
from ..geometry.selection import SelectionGraph
from .vtk_viewport import VtkViewport


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Split3r Rewrite - Interlocking Inserts")
        self.resize(1280, 820)
        self.mesh: pv.PolyData | None = None
        self.graph: SelectionGraph | None = None
        self.selected: set[int] = set()
        self.last_result: InsertResult | None = None
        self.depth_mm = 3.0
        self.clearance_mm = 0.2
        self.angle_deg = 35.0

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        self.viewport = VtkViewport(self)
        self.viewport.facePicked.connect(self.pick_face)
        layout.addWidget(self.viewport, stretch=1)

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        layout.addWidget(panel)

        self.status = QLabel("Listo. Importá un modelo para empezar. LMB click selecciona, Ctrl+LMB borra, LMB drag rota, RMB drag panea. Drag sobre un insert extraído lo mueve.")
        self.status.setWordWrap(True)
        panel_layout.addWidget(self.status)

        btn_load = QPushButton("Import STL/OBJ/3MF")
        btn_load.clicked.connect(self.load_model)
        panel_layout.addWidget(btn_load)

        self.lbl_angle = QLabel("Smart angle: 35°")
        panel_layout.addWidget(self.lbl_angle)
        self.slider_angle = QSlider(Qt.Orientation.Horizontal)
        self.slider_angle.setRange(1, 90)
        self.slider_angle.setValue(35)
        self.slider_angle.valueChanged.connect(self.set_angle)
        panel_layout.addWidget(self.slider_angle)

        self.lbl_depth = QLabel("Insert depth: 3.0mm")
        panel_layout.addWidget(self.lbl_depth)
        self.slider_depth = QSlider(Qt.Orientation.Horizontal)
        self.slider_depth.setRange(1, 100)
        self.slider_depth.setValue(30)
        self.slider_depth.valueChanged.connect(self.set_depth)
        panel_layout.addWidget(self.slider_depth)

        self.lbl_clearance = QLabel("Slot clearance: 0.2mm")
        panel_layout.addWidget(self.lbl_clearance)
        self.slider_clearance = QSlider(Qt.Orientation.Horizontal)
        self.slider_clearance.setRange(0, 20)
        self.slider_clearance.setValue(2)
        self.slider_clearance.valueChanged.connect(self.set_clearance)
        panel_layout.addWidget(self.slider_clearance)

        self.lbl_selected = QLabel("Selected faces: 0")
        panel_layout.addWidget(self.lbl_selected)

        btn_clear = QPushButton("Clear selection")
        btn_clear.clicked.connect(self.clear_selection)
        panel_layout.addWidget(btn_clear)

        btn_extract = QPushButton("CREATE INTERLOCKING INSERT")
        btn_extract.clicked.connect(self.extract_insert)
        panel_layout.addWidget(btn_extract)

        btn_export_insert = QPushButton("Export insert STL")
        btn_export_insert.clicked.connect(self.export_insert)
        panel_layout.addWidget(btn_export_insert)

        btn_export_body = QPushButton("Export body/socket STL")
        btn_export_body.clicked.connect(self.export_body)
        panel_layout.addWidget(btn_export_body)

        panel_layout.addStretch()

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def set_angle(self, value: int) -> None:
        self.angle_deg = float(value)
        self.lbl_angle.setText(f"Smart angle: {value}°")

    def set_depth(self, value: int) -> None:
        self.depth_mm = value / 10.0
        self.lbl_depth.setText(f"Insert depth: {self.depth_mm:.1f}mm")

    def set_clearance(self, value: int) -> None:
        self.clearance_mm = value / 10.0
        self.lbl_clearance.setText(f"Slot clearance: {self.clearance_mm:.1f}mm")

    def load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import model", "", "3D files (*.stl *.obj *.3mf);;All files (*.*)")
        if not path:
            return
        try:
            self.set_status("Cargando modelo...")
            self.mesh = load_polydata(path)
            self.graph = SelectionGraph.from_polydata(self.mesh)
            self.selected.clear()
            self.last_result = None
            self.viewport.clear_parts()
            self.viewport.set_mesh(self.mesh)
            self.update_selection_ui()
            self.set_status(f"Modelo cargado: {Path(path).name} | faces: {self.mesh.n_cells}")
        except Exception as exc:
            self.set_status(f"Error al cargar: {exc}")

    def pick_face(self, face_id: int, erase: bool) -> None:
        if self.mesh is None or self.graph is None or face_id < 0:
            return
        region = self.graph.smart_shell(face_id, self.angle_deg)
        if erase:
            self.selected.difference_update(region)
        else:
            self.selected.update(region)
        self.update_selection_ui()

    def update_selection_ui(self) -> None:
        self.lbl_selected.setText(f"Selected faces: {len(self.selected)}")
        self.viewport.update_selection(self.selected)

    def clear_selection(self) -> None:
        self.selected.clear()
        self.update_selection_ui()

    def extract_insert(self) -> None:
        if self.mesh is None:
            self.set_status("No hay modelo cargado.")
            return
        try:
            self.set_status("Calculando insert + slot boolean...")
            result = create_insert(self.mesh, self.selected, self.depth_mm, self.clearance_mm, backend="auto")
            self.last_result = result
            self.mesh = result.body
            self.graph = SelectionGraph.from_polydata(self.mesh)
            self.selected.clear()
            self.viewport.set_mesh(self.mesh)
            self.viewport.clear_parts()
            self.viewport.add_part(result.insert)
            self.update_selection_ui()
            warn = " Con warnings." if result.warnings else ""
            self.set_status(f"Insert creado. Backend: {result.backend}.{warn}")
        except Exception as exc:
            self.set_status(f"Error en extracción: {exc}")

    def _save_mesh(self, mesh: pv.PolyData | None, title: str) -> None:
        if mesh is None:
            self.set_status("No hay malla para exportar.")
            return
        path, _ = QFileDialog.getSaveFileName(self, title, "", "STL (*.stl);;PLY (*.ply);;OBJ (*.obj)")
        if not path:
            return
        try:
            mesh.save(path)
            self.set_status(f"Exportado: {path}")
        except Exception as exc:
            self.set_status(f"Error exportando: {exc}")

    def export_insert(self) -> None:
        moved_insert = self.viewport.transformed_part_mesh(-1)
        mesh = moved_insert if moved_insert is not None else (self.last_result.insert if self.last_result else None)
        self._save_mesh(mesh, "Export insert")

    def export_body(self) -> None:
        self._save_mesh(self.mesh, "Export body/socket")
