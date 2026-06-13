from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pyvista as pv

from .booleans import Backend, difference
from .boundary import Boundary, compute_boundary
from .mesh_io import triangular_faces, validate_polydata


@dataclass(frozen=True)
class InsertResult:
    insert: pv.PolyData
    body: pv.PolyData
    slot_cutter: pv.PolyData
    boundary: Boundary
    backend: str
    warnings: tuple[str, ...]


def _norm(v: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(v, axis=1)
    out = np.zeros_like(v, dtype=float)
    mask = lengths > 1e-12
    out[mask] = v[mask] / lengths[mask, None]
    return out


def _vertex_normals(points: np.ndarray, faces: np.ndarray, selected_faces: Iterable[int]) -> np.ndarray:
    tris = points[faces]
    face_normals = _norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]))
    normals = np.zeros((len(points), 3), dtype=float)
    for fid in selected_faces:
        for vid in faces[int(fid)]:
            normals[int(vid)] += face_normals[int(fid)]
    normals = _norm(normals)
    normals[np.linalg.norm(normals, axis=1) <= 1e-12] = (0, 0, 1)
    return normals


def _lateral_dirs(points: np.ndarray, normals: np.ndarray, boundary: Boundary) -> dict[int, np.ndarray]:
    centroid = points[list(boundary.selected_vertices)].mean(axis=0)
    out: dict[int, np.ndarray] = {}
    for vid in {v for edge in boundary.boundary_edges for v in edge}:
        raw = points[vid] - centroid
        n = normals[vid]
        tangent = raw - n * float(np.dot(raw, n))
        length = float(np.linalg.norm(tangent))
        out[int(vid)] = tangent / length if length > 1e-12 else np.zeros(3)
    return out


def build_insert_solid(mesh: pv.PolyData, boundary: Boundary, depth: float, clearance: float = 0.0) -> pv.PolyData:
    """Build selected-surface insert or expanded slot cutter.

    ``clearance=0`` creates the printable male insert. Positive clearance
    expands the side boundary and bottom to create the female slot cutter.
    """
    if depth <= 0:
        raise ValueError("Depth debe ser mayor a cero.")
    if clearance < 0:
        raise ValueError("Clearance no puede ser negativo.")
    points = np.asarray(mesh.points, dtype=float)
    faces = triangular_faces(mesh)
    normals = _vertex_normals(points, faces, boundary.selected_faces)
    lateral = _lateral_dirs(points, normals, boundary) if clearance else {}
    verts = list(boundary.selected_vertices)
    top_i = {v: i for i, v in enumerate(verts)}
    bot_i = {v: i + len(verts) for i, v in enumerate(verts)}
    top = points[verts].copy()
    bottom = points[verts].copy()
    for i, vid in enumerate(verts):
        lat = lateral.get(vid, np.zeros(3)) * clearance
        top[i] = top[i] + normals[vid] * (clearance * 0.5) + lat
        bottom[i] = bottom[i] - normals[vid] * (depth + clearance) + lat
    out_points = np.vstack([top, bottom])
    out_faces: list[int] = []
    for fid in boundary.selected_faces:
        a, b, c = map(int, faces[int(fid)])
        out_faces.extend([3, top_i[a], top_i[b], top_i[c]])
        out_faces.extend([3, bot_i[c], bot_i[b], bot_i[a]])
    for a, b in boundary.boundary_edges:
        ta, tb, ba, bb = top_i[a], top_i[b], bot_i[a], bot_i[b]
        out_faces.extend([3, ta, ba, tb])
        out_faces.extend([3, tb, ba, bb])
    solid = pv.PolyData(out_points, np.asarray(out_faces, dtype=np.int64)).clean().triangulate()
    solid = solid.compute_normals(auto_orient_normals=True, consistent_normals=True)
    validate_polydata(solid)
    return solid


def surface_fallback_body(mesh: pv.PolyData, selected: Iterable[int], slot: pv.PolyData) -> pv.PolyData:
    selected_set = set(int(i) for i in selected)
    remaining = sorted(set(range(mesh.n_cells)) - selected_set)
    body = mesh.extract_cells(remaining).extract_surface(algorithm="dataset_surface").triangulate()
    body = body.append_polydata(slot.extract_surface(algorithm="dataset_surface").triangulate()).clean().triangulate()
    validate_polydata(body)
    return body


def create_insert(mesh: pv.PolyData, selected: Iterable[int], depth: float, clearance: float = 0.2, backend: Backend = "auto") -> InsertResult:
    validate_polydata(mesh)
    work = mesh.triangulate().clean()
    boundary = compute_boundary(work, selected)
    insert = build_insert_solid(work, boundary, depth, 0.0)
    slot = build_insert_solid(work, boundary, depth, clearance)
    warnings: list[str] = []
    try:
        bool_out = difference(work, slot, backend=backend)
        body = bool_out.mesh
        warnings.extend(bool_out.warnings)
        used_backend = bool_out.backend
    except Exception as exc:
        body = surface_fallback_body(work, boundary.selected_faces, slot)
        used_backend = "surface-fallback"
        warnings.append(f"Boolean falló; usando fallback de superficie: {exc}")
    return InsertResult(insert, body, slot, boundary, used_backend, tuple(warnings))
