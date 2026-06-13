import pyvista as pv
import pytest

from split3r_rewrite.geometry.boundary import chain_loops, compute_boundary
from split3r_rewrite.geometry.insert import create_insert


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
