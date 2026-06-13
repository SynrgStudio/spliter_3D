from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh


def scene_to_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    meshes: list[trimesh.Trimesh] = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        geom = scene.geometry.get(geometry_name)
        if isinstance(geom, trimesh.Trimesh):
            mesh = geom.copy()
            mesh.apply_transform(transform)
            meshes.append(mesh)
    if not meshes:
        raise ValueError("La escena no contiene mallas compatibles.")
    return trimesh.util.concatenate(meshes)


def load_polydata(path: str | Path) -> pv.PolyData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    loaded = trimesh.load(path, force=None)
    mesh = scene_to_mesh(loaded) if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Formato no soportado: {type(mesh).__name__}")
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError("La malla está vacía.")
    mesh.remove_unreferenced_vertices()
    if mesh.faces.shape[1] != 3:
        raise ValueError("Solo se soportan mallas trianguladas.")
    faces = np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)).ravel()
    return pv.PolyData(mesh.vertices, faces).triangulate().clean()


def validate_polydata(mesh: pv.PolyData | None) -> None:
    if mesh is None:
        raise ValueError("No hay malla cargada.")
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError("La malla está vacía.")
    faces = np.asarray(mesh.faces)
    if len(faces) % 4 != 0:
        raise ValueError("La tabla de caras es inválida.")
    reshaped = faces.reshape((-1, 4))
    if np.any(reshaped[:, 0] != 3):
        raise ValueError("La malla debe estar triangulada.")


def triangular_faces(mesh: pv.PolyData) -> np.ndarray:
    validate_polydata(mesh)
    return np.asarray(mesh.faces, dtype=np.int64).reshape((-1, 4))[:, 1:]
