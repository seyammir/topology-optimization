"""Unit tests for src.models.spring.Spring."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.models.node import Node
from src.models.spring import Spring


class TestSpringCreation:
    """Test Spring construction."""

    def test_horizontal_spring_auto_k(self, horizontal_spring):
        assert horizontal_spring.k == 1.0
        assert horizontal_spring.length == pytest.approx(1.0)
        assert horizontal_spring.angle == pytest.approx(0.0)

    def test_vertical_spring_auto_k(self):
        n0 = Node(id=0, x=0.0, z=0.0)
        n1 = Node(id=1, x=0.0, z=1.0)
        sp = Spring(n0, n1)
        assert sp.k == 1.0
        assert sp.length == pytest.approx(1.0)
        assert sp.angle == pytest.approx(math.pi / 2)

    def test_diagonal_spring_auto_k(self, diagonal_spring):
        assert diagonal_spring.k == pytest.approx(1.0 / math.sqrt(2))
        assert diagonal_spring.length == pytest.approx(math.sqrt(2))
        assert diagonal_spring.angle == pytest.approx(math.pi / 4)

    def test_custom_k(self):
        n0 = Node(id=0, x=0.0, z=0.0)
        n1 = Node(id=1, x=1.0, z=0.0)
        sp = Spring(n0, n1, k=5.0)
        assert sp.k == 5.0

    def test_node_ids(self, horizontal_spring):
        assert horizontal_spring.node_ids == (0, 1)


class TestSpringDOFs:
    """Test DOF index retrieval."""

    def test_dof_indices(self):
        n0 = Node(id=0, x=0.0, z=0.0)
        n1 = Node(id=1, x=1.0, z=0.0)
        n0.dof_indices = (0, 1)
        n1.dof_indices = (2, 3)
        sp = Spring(n0, n1)
        assert sp.dof_indices == [0, 1, 2, 3]


class TestStiffnessMatrix:
    """Test the local (global-coordinate) stiffness matrix."""

    def test_horizontal_spring_matrix(self, horizontal_spring):
        """For θ=0: c=1, s=0 → only the x-x entries are non-zero."""
        n0 = horizontal_spring.node_i
        n1 = horizontal_spring.node_j
        n0.dof_indices = (0, 1)
        n1.dof_indices = (2, 3)

        ke = horizontal_spring.local_stiffness_matrix()
        assert ke.shape == (4, 4)

        k = horizontal_spring.k
        expected = k * np.array([
            [ 1,  0, -1,  0],
            [ 0,  0,  0,  0],
            [-1,  0,  1,  0],
            [ 0,  0,  0,  0],
        ], dtype=float)
        np.testing.assert_array_almost_equal(ke, expected)

    def test_vertical_spring_matrix(self):
        """For θ=π/2: c=0, s=1 → only the z-z entries are non-zero."""
        n0 = Node(id=0, x=0.0, z=0.0)
        n1 = Node(id=1, x=0.0, z=1.0)
        sp = Spring(n0, n1)

        ke = sp.local_stiffness_matrix()
        k = sp.k
        expected = k * np.array([
            [0,  0,  0,  0],
            [0,  1,  0, -1],
            [0,  0,  0,  0],
            [0, -1,  0,  1],
        ], dtype=float)
        np.testing.assert_array_almost_equal(ke, expected)

    def test_stiffness_matrix_symmetry(self, diagonal_spring):
        ke = diagonal_spring.local_stiffness_matrix()
        np.testing.assert_array_almost_equal(ke, ke.T)

    def test_stiffness_matrix_singular(self, horizontal_spring):
        """The element stiffness matrix must be singular (rigid-body modes)."""
        ke = horizontal_spring.local_stiffness_matrix()
        assert np.linalg.matrix_rank(ke) < 4


class TestStrainEnergy:
    """Test strain energy computation."""

    def test_zero_displacement(self, horizontal_spring):
        n0 = horizontal_spring.node_i
        n1 = horizontal_spring.node_j
        n0.dof_indices = (0, 1)
        n1.dof_indices = (2, 3)
        u = np.zeros(4)
        assert horizontal_spring.strain_energy(u) == 0.0

    def test_nonzero_displacement(self, horizontal_spring):
        n0 = horizontal_spring.node_i
        n1 = horizontal_spring.node_j
        n0.dof_indices = (0, 1)
        n1.dof_indices = (2, 3)
        # Stretch the spring by 1 unit in x
        u = np.array([0.0, 0.0, 1.0, 0.0])
        se = horizontal_spring.strain_energy(u)
        # E = 0.5 * k * δ² = 0.5 * 1.0 * 1² = 0.5
        assert se == pytest.approx(0.5)

    def test_rigid_body_translation(self, horizontal_spring):
        """Rigid-body translation produces zero strain energy."""
        n0 = horizontal_spring.node_i
        n1 = horizontal_spring.node_j
        n0.dof_indices = (0, 1)
        n1.dof_indices = (2, 3)
        u = np.array([1.0, 2.0, 1.0, 2.0])
        assert horizontal_spring.strain_energy(u) == pytest.approx(0.0)


class TestSpringSerialization:
    """Test to_dict."""

    def test_to_dict(self, horizontal_spring):
        d = horizontal_spring.to_dict()
        assert d["node_i"] == 0
        assert d["node_j"] == 1
        assert d["k"] == 1.0

    def test_repr(self, horizontal_spring):
        r = repr(horizontal_spring)
        assert "Spring" in r
        assert "0" in r
        assert "1" in r
