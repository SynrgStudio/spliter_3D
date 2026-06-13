from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pyvista as pv
import trimesh

from .mesh_io import triangular_faces


@dataclass
class SelectionGraph:
    adjacency: dict[int, list[tuple[int, float]]]

    @classmethod
    def from_polydata(cls, mesh: pv.PolyData) -> "SelectionGraph":
        faces = triangular_faces(mesh)
        tm = trimesh.Trimesh(vertices=np.asarray(mesh.points), faces=faces, process=False)
        adjacency: dict[int, list[tuple[int, float]]] = {}
        for edge, angle in zip(tm.face_adjacency, tm.face_adjacency_angles):
            a, b = int(edge[0]), int(edge[1])
            adjacency.setdefault(a, []).append((b, float(angle)))
            adjacency.setdefault(b, []).append((a, float(angle)))
        return cls(adjacency)

    def smart_shell(self, seed: int, angle_degrees: float) -> set[int]:
        if seed < 0:
            return set()
        threshold = float(np.radians(angle_degrees))
        visited = {int(seed)}
        queue: deque[int] = deque([int(seed)])
        while queue:
            current = queue.popleft()
            for neighbor, angle in self.adjacency.get(current, []):
                if neighbor not in visited and angle <= threshold:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited
