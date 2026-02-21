"""Unit tests for src.models.node.Node."""

from __future__ import annotations

import pytest

from src.models.node import Node


class TestNodeCreation:
    """Test Node construction and default values."""

    def test_default_values(self):
        n = Node(id=0, x=1.0, z=2.0)
        assert n.id == 0
        assert n.x == 1.0
        assert n.z == 2.0
        assert n.mass == 1.0
        assert n.fixed_x is False
        assert n.fixed_z is False
        assert n.fx == 0.0
        assert n.fz == 0.0

    def test_custom_values(self):
        n = Node(id=5, x=3.0, z=4.0, mass=2.5, fixed_x=True, fx=10.0)
        assert n.mass == 2.5
        assert n.fixed_x is True
        assert n.fixed_z is False
        assert n.fx == 10.0


class TestNodeProperties:
    """Test computed properties on Node."""

    def test_is_fixed_partial(self):
        n = Node(id=0, x=0.0, z=0.0, fixed_x=True)
        assert n.is_fixed is True
        assert n.is_pinned is False

    def test_is_pinned(self, fixed_node):
        assert fixed_node.is_fixed is True
        assert fixed_node.is_pinned is True

    def test_not_fixed(self, simple_node):
        assert simple_node.is_fixed is False
        assert simple_node.is_pinned is False

    def test_has_load(self, loaded_node):
        assert loaded_node.has_load is True

    def test_no_load(self, simple_node):
        assert simple_node.has_load is False

    def test_is_protected_fixed(self, fixed_node):
        assert fixed_node.is_protected is True

    def test_is_protected_loaded(self, loaded_node):
        assert loaded_node.is_protected is True

    def test_not_protected(self, simple_node):
        assert simple_node.is_protected is False


class TestNodeDOFs:
    """Test DOF index assignment."""

    def test_default_dof_indices(self):
        n = Node(id=0, x=0.0, z=0.0)
        assert n.dof_indices == (-1, -1)

    def test_set_dof_indices(self):
        n = Node(id=0, x=0.0, z=0.0)
        n.dof_indices = (4, 5)
        assert n.dof_indices == (4, 5)
        assert n._dof_x == 4
        assert n._dof_z == 5


class TestNodeSerialization:
    """Test to_dict / from_dict round-trip."""

    def test_to_dict(self, simple_node):
        d = simple_node.to_dict()
        assert d["id"] == 0
        assert d["x"] == 1.0
        assert d["z"] == 2.0
        assert d["mass"] == 1.0
        assert d["fixed_x"] is False
        assert d["fx"] == 0.0

    def test_round_trip(self):
        original = Node(id=7, x=3.5, z=1.2, mass=2.0, fixed_x=True, fz=-5.0)
        d = original.to_dict()
        restored = Node.from_dict(d)
        assert restored.id == original.id
        assert restored.x == original.x
        assert restored.z == original.z
        assert restored.mass == original.mass
        assert restored.fixed_x == original.fixed_x
        assert restored.fz == original.fz

    def test_from_dict_defaults(self):
        d = {"id": 0, "x": 0.0, "z": 0.0}
        n = Node.from_dict(d)
        assert n.mass == 1.0
        assert n.fixed_x is False
        assert n.fx == 0.0


class TestNodeEquality:
    """Test __eq__ and __hash__."""

    def test_equal_by_id(self):
        a = Node(id=1, x=0.0, z=0.0)
        b = Node(id=1, x=5.0, z=5.0)
        assert a == b

    def test_not_equal_different_id(self):
        a = Node(id=1, x=0.0, z=0.0)
        b = Node(id=2, x=0.0, z=0.0)
        assert a != b

    def test_hash_by_id(self):
        a = Node(id=3, x=0.0, z=0.0)
        b = Node(id=3, x=1.0, z=1.0)
        assert hash(a) == hash(b)

    def test_not_equal_other_type(self):
        n = Node(id=0, x=0.0, z=0.0)
        assert n != "not_a_node"
