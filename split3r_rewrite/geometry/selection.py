from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
import pyvista as pv
import trimesh

from .mesh_io import triangular_faces


@dataclass(frozen=True)
class FaceNeighbor:
    face: int
    angle: float
    edge: tuple[int, int]


@dataclass
class SelectionGraph:
    """Connectivity graph used by the interactive selector.

    The important bit is that selection is *boundary aware*.  A plain
    normal-angle flood-fill treats every coplanar triangulation edge as a
    path, so a click on one star point can leak through the whole coplanar
    fan.  Here we keep the shared edge and topological boundary vertices so
    ``smart_shell`` can refuse to cross coplanar lobe/base edges whose two
    endpoints are on a real boundary.
    """

    adjacency: dict[int, list[FaceNeighbor]]
    boundary_vertices: set[int]

    @classmethod
    def from_polydata(cls, mesh: pv.PolyData) -> "SelectionGraph":
        faces = triangular_faces(mesh)
        tm = trimesh.Trimesh(vertices=np.asarray(mesh.points), faces=faces, process=False)

        edge_counts: Counter[tuple[int, int]] = Counter()
        for tri in faces:
            a, b, c = map(int, tri)
            edge_counts[_edge(a, b)] += 1
            edge_counts[_edge(b, c)] += 1
            edge_counts[_edge(c, a)] += 1
        boundary_vertices = {v for edge, count in edge_counts.items() if count == 1 for v in edge}

        adjacency: dict[int, list[FaceNeighbor]] = {}
        for (a, b), angle in zip(tm.face_adjacency, tm.face_adjacency_angles):
            fa, fb = int(a), int(b)
            shared = _shared_edge(faces[fa], faces[fb])
            link_ab = FaceNeighbor(fb, float(angle), shared)
            link_ba = FaceNeighbor(fa, float(angle), shared)
            adjacency.setdefault(fa, []).append(link_ab)
            adjacency.setdefault(fb, []).append(link_ba)
        return cls(adjacency, boundary_vertices)

    def smart_shell(self, seed: int, angle_degrees: float) -> set[int]:
        if seed < 0:
            return set()
        threshold = float(np.radians(angle_degrees))
        coplanar_eps = float(np.radians(2.0))
        visited = {int(seed)}
        queue: deque[int] = deque([int(seed)])
        while queue:
            current = queue.popleft()
            for neighbor in self.adjacency.get(current, []):
                if neighbor.face in visited:
                    continue
                if neighbor.angle > threshold:
                    continue
                if self._is_real_coplanar_boundary(neighbor.edge, neighbor.angle, coplanar_eps):
                    continue
                visited.add(neighbor.face)
                queue.append(neighbor.face)
        return visited

    def _is_real_coplanar_boundary(self, edge: tuple[int, int], angle: float, coplanar_eps: float) -> bool:
        """Return true for coplanar seams that are likely feature boundaries.

        In open/flat artwork (stars, logos, eye patches) the actual lobe
        boundary is often a coplanar internal edge connecting two perimeter
        vertices.  A pure coplanarity flood-fill crosses it and selects the
        wrong lobe.  If both endpoints are topological boundary vertices, we
        treat that coplanar edge as a hard stop.
        """
        return angle <= coplanar_eps and edge[0] in self.boundary_vertices and edge[1] in self.boundary_vertices


def _edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _shared_edge(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    common = sorted(set(map(int, a)).intersection(map(int, b)))
    if len(common) != 2:
        raise ValueError("Adjacent faces do not share one full edge.")
    return int(common[0]), int(common[1])
