from __future__ import annotations

import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pyvista as pv

from .mesh_io import validate_polydata

Backend = Literal["auto", "blender", "vtk"]


@dataclass(frozen=True)
class BooleanOutput:
    mesh: pv.PolyData
    backend: str
    warnings: tuple[str, ...] = ()


def _prep(mesh: pv.PolyData) -> pv.PolyData:
    validate_polydata(mesh)
    return mesh.extract_surface(algorithm="dataset_surface").triangulate().clean()


def _vtk_difference(base: pv.PolyData, cutter: pv.PolyData) -> BooleanOutput:
    out = _prep(base).boolean_difference(_prep(cutter), progress_bar=False).clean().triangulate()
    validate_polydata(out)
    if out.n_cells == 0:
        raise ValueError("VTK devolvió una malla vacía.")
    return BooleanOutput(out, "vtk")


def _blender_difference(base: pv.PolyData, cutter: pv.PolyData, blender: str) -> BooleanOutput:
    with tempfile.TemporaryDirectory(prefix="split3r_next_bool_") as tmp:
        tmp_path = Path(tmp)
        base_path = tmp_path / "base.stl"
        cutter_path = tmp_path / "cutter.stl"
        out_path = tmp_path / "out.stl"
        script = tmp_path / "bool.py"
        _prep(base).save(base_path)
        _prep(cutter).save(cutter_path)
        script.write_text(textwrap.dedent(f"""
            import bpy
            for obj in list(bpy.context.scene.objects):
                obj.select_set(True)
            bpy.ops.object.delete()
            bpy.ops.wm.stl_import(filepath={str(base_path)!r})
            base = bpy.context.object
            bpy.ops.wm.stl_import(filepath={str(cutter_path)!r})
            cutter = bpy.context.object
            bpy.context.view_layer.objects.active = base
            mod = base.modifiers.new('split3r_slot', 'BOOLEAN')
            mod.operation = 'DIFFERENCE'
            mod.object = cutter
            mod.solver = 'EXACT'
            bpy.ops.object.modifier_apply(modifier=mod.name)
            cutter.select_set(True)
            bpy.ops.object.delete()
            base.select_set(True)
            bpy.ops.wm.stl_export(filepath={str(out_path)!r}, export_selected_objects=True)
        """), encoding="utf-8")
        proc = subprocess.run([blender, "--background", "--factory-startup", "--python", str(script)], capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-2000:] or proc.stdout[-2000:])
        out = pv.read(out_path).extract_surface(algorithm="dataset_surface").triangulate().clean()
        validate_polydata(out)
        return BooleanOutput(out, "blender")


def difference(base: pv.PolyData, cutter: pv.PolyData, backend: Backend = "auto") -> BooleanOutput:
    warnings: list[str] = []
    if backend in ("auto", "blender"):
        blender = shutil.which("blender")
        if blender:
            try:
                return _blender_difference(base, cutter, blender)
            except Exception as exc:
                if backend == "blender":
                    raise
                warnings.append(f"Blender falló, usando VTK: {exc}")
        elif backend == "blender":
            raise RuntimeError("Blender no está en PATH.")
    out = _vtk_difference(base, cutter)
    return BooleanOutput(out.mesh, out.backend, tuple(warnings))
