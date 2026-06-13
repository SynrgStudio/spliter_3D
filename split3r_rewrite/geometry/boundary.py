from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pyvista as pv

from .mesh_io import triangular_faces, validate_polydata


@dataclass(frozen=True)
class Boundary:
    selected_faces: tuple[int, ...]
    selected_vertices: tuple[int, ...]
    boundary_edges: tuple[tuple[int, int], ...]
    boundary_loops: tuple[tuple[int, ...], ...]


def valid_selection(mesh: pv.PolyData, selected: Iterable[int]) -> tuple[int, ...]:
    validate_polydata(mesh)
    out = tuple(sorted({int(i) for i in selected if 0 <= int(i) < mesh.n_cells}))
    if not out:
        raise ValueError("No hay caras seleccionadas.")
    if len(out) >= mesh.n_cells:
        raise ValueError("No se puede extraer el 100% del modelo.")
    return out


def _edges(face: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    a, b, c = map(int, face)
    return tuple(tuple(sorted(e)) for e in ((a, b), (b, c), (c, a)))  # type: ignore[return-value]


def chain_loops(edges: Iterable[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    remaining = {tuple(sorted((int(a), int(b)))) for a, b in edges}
    if not remaining:
        raise ValueError("La selección no tiene borde abierto/cerrado detectable.")
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in remaining:
        adj[a].append(b)
        adj[b].append(a)
    if any(len(n) != 2 for n in adj.values()):
        raise ValueError("El boundary está abierto o tiene ramificaciones.")
    loops: list[tuple[int, ...]] = []
    while remaining:
        start, cur = next(iter(remaining))
        prev = start
        loop = [start]
        while True:
            loop.append(cur)
            remaining.discard(tuple(sorted((prev, cur))))
            n0, n1 = adj[cur]
            nxt = n0 if n1 == prev else n1
            prev, cur = cur, nxt
            if cur == start:
                remaining.discard(tuple(sorted((prev, cur))))
                break
            if len(loop) > len(adj) + 2:
                raise ValueError("No se pudo cerrar el loop del boundary.")
        if len(loop) < 3:
            raise ValueError("Boundary demasiado chico.")
        loops.append(tuple(loop))
    return tuple(loops)


def compute_boundary(mesh: pv.PolyData, selected: Iterable[int]) -> Boundary:
    faces = triangular_faces(mesh)
    selected_faces = valid_selection(mesh, selected)
    counts: Counter[tuple[int, int]] = Counter()
    vertices: set[int] = set()
    for face_id in selected_faces:
        face = faces[face_id]
        vertices.update(map(int, face))
        counts.update(_edges(face))
    boundary_edges = tuple(sorted(edge for edge, count in counts.items() if count == 1))
    loops = chain_loops(boundary_edges)
    return Boundary(selected_faces, tuple(sorted(vertices)), boundary_edges, loops)
