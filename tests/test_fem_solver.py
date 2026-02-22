"""Unit tests for src.solver.fem_solver.FEMSolver."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from src.models.node import Node
from src.models.spring import Spring
from src.models.structure import Structure
from src.solver.fem_solver import FEMSolver


@pytest.fixture
def solver() -> FEMSolver:
    return FEMSolver()


@pytest.fixture
def two_node_fixed() -> Structure:
    """Two-node system: left node pinned, right node loaded downward.

    n0(0,0) --- n1(1,0)
    pinned       Fz=1
    """
    s = Structure()
    n0 = s.add_node(0.0, 0.0)
    n1 = s.add_node(1.0, 0.0)
    n0.fixed_x = True
    n0.fixed_z = True
    n1.fz = 1.0
    s.add_spring(n0, n1)
    s.renumber_dofs()
    return s


@pytest.fixture
def triangle_structure() -> Structure:
    """Three-node triangle: bottom-left pinned, bottom-right roller, top loaded.

    Forces a well-conditioned problem with multiple springs at angles.
    """
    s = Structure()
    n0 = s.add_node(0.0, 0.0)   # bottom-left
    n1 = s.add_node(1.0, 0.0)   # bottom-right
    n2 = s.add_node(0.5, 1.0)   # top
    n0.fixed_x = True
    n0.fixed_z = True
    n1.fixed_z = True
    n2.fz = -1.0  # load upward (negative z)
    s.add_spring(n0, n1)
    s.add_spring(n0, n2)
    s.add_spring(n1, n2)
    s.renumber_dofs()
    return s


class TestAssembly:
    """Test stiffness matrix and force vector assembly."""

    def test_global_stiffness_shape(self, solver, two_node_fixed):
        K = solver._assemble_global_stiffness(two_node_fixed)
        assert K.shape == (4, 4)

    def test_global_stiffness_symmetry(self, solver, two_node_fixed):
        K = solver._assemble_global_stiffness(two_node_fixed)
        K_dense = K.toarray()
        np.testing.assert_array_almost_equal(K_dense, K_dense.T)

    def test_force_vector(self, solver, two_node_fixed):
        F = solver._build_force_vector(two_node_fixed)
        assert F.shape == (4,)
        # n0: no force -> F[0]=0, F[1]=0; n1: Fz=1 -> F[3]=1
        assert F[0] == 0.0
        assert F[1] == 0.0
        assert F[2] == 0.0
        assert F[3] == 1.0

    def test_boundary_conditions_penalty(self, solver, two_node_fixed):
        K = solver._assemble_global_stiffness(two_node_fixed)
        F = solver._build_force_vector(two_node_fixed)
        K, F = solver._apply_boundary_conditions(K, F, two_node_fixed)
        K_dense = K.toarray()
        # Fixed DOFs (0, 1) should have PENALTY on diagonal
        assert K_dense[0, 0] == FEMSolver.PENALTY
        assert K_dense[1, 1] == FEMSolver.PENALTY
        # Force on fixed DOFs should be zero
        assert F[0] == 0.0
        assert F[1] == 0.0


class TestSolve:
    """Test the full solve pipeline."""

    def test_solve_returns_displacement(self, solver, triangle_structure):
        u = solver.solve(triangle_structure)
        assert u.shape == (6,)  # 3 nodes x 2 DOFs

    def test_fixed_dofs_near_zero(self, solver, triangle_structure):
        u = solver.solve(triangle_structure)
        # n0 is pinned: DOFs 0, 1 should be ~0
        assert abs(u[0]) < 1e-6
        assert abs(u[1]) < 1e-6
        # n1 fixed_z: DOF 3 should be ~0
        assert abs(u[3]) < 1e-6

    def test_loaded_node_displaces(self, solver, triangle_structure):
        u = solver.solve(triangle_structure)
        # n2 loaded with Fz=-1 (upward), should have negative z-displacement
        # DOF index for n2.z = 5
        assert u[5] < 0  # moves upward (negative z)

    def test_no_nan_in_result(self, solver, triangle_structure):
        u = solver.solve(triangle_structure)
        assert not np.any(np.isnan(u))
        assert not np.any(np.isinf(u))

    def test_solve_cantilever(self, solver, cantilever_beam):
        u = solver.solve(cantilever_beam)
        assert u.shape == (cantilever_beam.num_dofs,)
        assert not np.any(np.isnan(u))

    def test_larger_structure(self, solver, rect_3x2):
        """Solve a structure with BCs applied."""
        # Pin bottom-left, roller bottom-right, load top-center
        nodes = rect_3x2.get_nodes()
        # Find bottom-left (x=0, z=max)
        max_z = max(n.z for n in nodes)
        for n in nodes:
            if n.x == 0.0 and n.z == max_z:
                n.fixed_x = True
                n.fixed_z = True
            if n.x == 3.0 and n.z == max_z:
                n.fixed_z = True
        # Load top-center
        for n in nodes:
            if n.x == 1.0 and n.z == 0.0:
                n.fz = 1.0
                break

        rect_3x2.renumber_dofs()
        u = solver.solve(rect_3x2)
        assert len(u) == rect_3x2.num_dofs
        assert not np.any(np.isnan(u))
