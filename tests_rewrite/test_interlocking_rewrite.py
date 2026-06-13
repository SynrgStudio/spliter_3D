import pyvista as pv
import pytest

from split3r_rewrite.geometry.boundary import chain_loops, compute_boundary
from split3r_rewrite.geometry.insert import create_insert
from split3r_rewrite.geometry.selection import SelectionGraph


def cube_top():
    mesh = pv.Cube().triangulate().clean()
    centers = mesh.cell_centers().points
    z = centers[:, 2].max()
    selected = {i for i, p in enumerate(centers) if p[2] == z}
    return mesh, selected


def test_boundary_for_cube_top_is_closed():
    mesh, selected = cube_top()
    boundary = compute_boundary(mesh, selected)
    assert len(boundary.boundary_edges) == 4
    assert len(boundary.boundary_loops) == 1


def test_open_boundary_rejected():
    with pytest.raises(ValueError):
        chain_loops([(0, 1), (1, 2), (2, 3)])


def test_insert_workflow_generates_insert_slot_body():
    mesh, selected = cube_top()
    result = create_insert(mesh, selected, depth=0.25, clearance=0.05, backend="vtk")
    assert result.insert.n_cells > 0
    assert result.slot_cutter.volume > result.insert.volume
    assert result.body.n_cells > 0
    assert result.backend in {"vtk", "surface-fallback"}


def test_smart_selection_stops_at_coplanar_real_boundary():
    # Two coplanar triangles share an internal edge. Both endpoints of that
    # shared edge are on the real/open perimeter, so it is a lobe/base boundary
    # rather than a triangulation edge to flood through.
    points = [
        (0.0, 1.0, 0.0),   # 0 upper notch
        (0.0, -1.0, 0.0),  # 1 lower notch
        (2.0, 0.0, 0.0),   # 2 star tip
        (-1.0, 0.0, 0.0),  # 3 center/body side
    ]
    faces = [3, 0, 1, 2, 3, 1, 0, 3]
    mesh = pv.PolyData(points, faces).triangulate().clean()
    graph = SelectionGraph.from_polydata(mesh)

    assert graph.smart_shell(0, angle_degrees=35.0) == {0}
