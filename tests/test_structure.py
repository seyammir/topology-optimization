"""Unit tests for src.models.structure.Structure."""

from __future__ import annotations

import pytest

from src.models.node import Node
from src.models.spring import Spring
from src.models.structure import Structure


class TestStructureCreation:
    """Test factory and manual construction."""

    def test_empty_structure(self):
        s = Structure()
        assert s.num_nodes == 0
        assert s.num_dofs == 0

    def test_rectangular_node_count(self, rect_2x1):
        # 2x1 grid -> (2+1)x(1+1) = 6 nodes
        assert rect_2x1.num_nodes == 6

    def test_rectangular_dof_count(self, rect_2x1):
        assert rect_2x1.num_dofs == 12

    def test_rectangular_spring_count(self, rect_2x1):
        # 2x1 grid: 2 cells
        # Horizontal: 2*2 = 4, Vertical: 3*1 = 3, Diag: 2*2 = 4
        # Actually: horiz per row = 2, rows = 2 -> 4; vert per col = 1, cols = 3 -> 3
        # Diags: 2 cells x 2 diags = 4
        # Total = 4 + 3 + 4 = 11
        springs = rect_2x1.get_springs()
        assert len(springs) == 11

    def test_rectangular_larger(self, rect_3x2):
        # 3x2 -> (4)x(3) = 12 nodes
        assert rect_3x2.num_nodes == 12


class TestNodeOperations:
    """Test add, remove, get operations on nodes."""

    def test_add_node(self):
        s = Structure()
        n = s.add_node(1.0, 2.0)
        assert n.x == 1.0
        assert n.z == 2.0
        assert s.num_nodes == 1

    def test_add_multiple_nodes(self):
        s = Structure()
        n0 = s.add_node(0.0, 0.0)
        n1 = s.add_node(1.0, 0.0)
        assert n0.id != n1.id
        assert s.num_nodes == 2

    def test_get_node(self):
        s = Structure()
        n = s.add_node(3.0, 4.0)
        retrieved = s.get_node(n.id)
        assert retrieved.x == 3.0
        assert retrieved.z == 4.0

    def test_get_nodes(self, rect_2x1):
        nodes = rect_2x1.get_nodes()
        assert len(nodes) == 6

    def test_get_node_ids(self, rect_2x1):
        ids = rect_2x1.get_node_ids()
        assert len(ids) == 6

    def test_remove_node(self, rect_2x1):
        initial_count = rect_2x1.num_nodes
        # Pick a non-corner node to remove
        rect_2x1.remove_node(0)
        assert rect_2x1.num_nodes == initial_count - 1

    def test_remove_nonexistent_node(self, rect_2x1):
        with pytest.raises(KeyError):
            rect_2x1.remove_node(9999)


class TestSpringOperations:
    """Test spring add / get / for_node."""

    def test_add_spring(self):
        s = Structure()
        n0 = s.add_node(0.0, 0.0)
        n1 = s.add_node(1.0, 0.0)
        sp = s.add_spring(n0, n1)
        assert sp.k == 1.0
        assert s.graph.number_of_edges() == 1

    def test_get_spring(self):
        s = Structure()
        n0 = s.add_node(0.0, 0.0)
        n1 = s.add_node(1.0, 0.0)
        s.add_spring(n0, n1, k=3.0)
        sp = s.get_spring(n0.id, n1.id)
        assert sp.k == 3.0

    def test_get_springs(self, rect_2x1):
        springs = rect_2x1.get_springs()
        assert len(springs) > 0

    def test_get_springs_for_node(self, rect_2x1):
        springs = rect_2x1.get_springs_for_node(0)
        assert len(springs) > 0


class TestDOFManagement:
    """Test DOF numbering."""

    def test_renumber_dofs(self):
        s = Structure()
        n0 = s.add_node(0.0, 0.0)
        n1 = s.add_node(1.0, 0.0)
        s.renumber_dofs()
        # Nodes sorted by id: 0 -> (0,1), 1 -> (2,3)
        assert n0.dof_indices == (0, 1)
        assert n1.dof_indices == (2, 3)

    def test_renumber_after_removal(self):
        s = Structure()
        n0 = s.add_node(0.0, 0.0)
        n1 = s.add_node(1.0, 0.0)
        n2 = s.add_node(2.0, 0.0)
        s.add_spring(n0, n1)
        s.add_spring(n1, n2)
        s.remove_node(n1.id)
        s.renumber_dofs()
        # Only n0 and n2 remain, sorted -> (0,1) and (2,3)
        assert n0.dof_indices == (0, 1)
        assert n2.dof_indices == (2, 3)


class TestTopologyQueries:
    """Test connectivity, mechanism, and mass queries."""

    def test_is_connected_full_grid(self, rect_2x1):
        assert rect_2x1.is_connected() is True

    def test_is_connected_empty(self):
        s = Structure()
        assert s.is_connected() is False

    def test_total_mass(self, rect_2x1):
        # 6 nodes x default mass 1.0 = 6.0
        assert rect_2x1.total_mass() == pytest.approx(6.0)

    def test_protected_node_ids(self, cantilever_beam):
        protected = cantilever_beam.get_protected_node_ids()
        # Should include fixed and loaded nodes
        assert len(protected) > 0

    def test_supports_reachable_from_loads(self, cantilever_beam):
        assert cantilever_beam.supports_reachable_from_loads() is True

    def test_supports_unreachable_when_disconnected(self):
        """When load and support are in different components, should fail."""
        s = Structure()
        n0 = s.add_node(0.0, 0.0)  # fixed
        n1 = s.add_node(10.0, 0.0)  # loaded, no connection
        n0.fixed_x = True
        n0.fixed_z = True
        n1.fz = -1.0
        assert s.supports_reachable_from_loads() is False


class TestMechanismDetection:
    """Test has_mechanism and related checks."""

    def test_no_mechanism_in_grid(self, rect_2x1):
        assert rect_2x1.has_mechanism() is False

    def test_single_spring_is_mechanism(self):
        """A node with only one spring (collinear) is a mechanism."""
        s = Structure()
        n0 = s.add_node(0.0, 0.0)
        n1 = s.add_node(1.0, 0.0)
        s.add_spring(n0, n1)
        s.renumber_dofs()
        assert s.has_mechanism() is True

    def test_collinear_springs_mechanism(self):
        """Three nodes in a line - middle node is a mechanism."""
        s = Structure()
        n0 = s.add_node(0.0, 0.0)
        n1 = s.add_node(1.0, 0.0)
        n2 = s.add_node(2.0, 0.0)
        s.add_spring(n0, n1)
        s.add_spring(n1, n2)
        s.renumber_dofs()
        assert s.has_mechanism() is True

    def test_pinned_node_not_mechanism(self):
        """A pinned node with collinear springs is NOT a mechanism."""
        s = Structure()
        n0 = s.add_node(0.0, 0.0)
        n0.fixed_x = True
        n0.fixed_z = True
        n1 = s.add_node(1.0, 0.0)
        s.add_spring(n0, n1)
        s.renumber_dofs()
        # n0 is pinned, so not a mechanism. n1 has 1 spring -> mechanism
        assert s.node_is_mechanism(n0.id) is False
        assert s.node_is_mechanism(n1.id) is True


class TestSnapshot:
    """Test deep-copy snapshot."""

    def test_snapshot_independent(self, rect_2x1):
        snap = rect_2x1.snapshot()
        # Modify original
        original_count = rect_2x1.num_nodes
        some_id = rect_2x1.get_node_ids()[0]
        rect_2x1.remove_node(some_id)
        # Snapshot should be unaffected
        assert snap.num_nodes == original_count
        assert rect_2x1.num_nodes == original_count - 1


class TestRemoveDanglingNodes:
    """Test iterative dangling-node removal."""

    def test_removes_leaf_node(self):
        """A degree-1 non-protected node should be removed."""
        s = Structure()
        n0 = s.add_node(0.0, 0.0)
        n1 = s.add_node(1.0, 0.0)
        n2 = s.add_node(2.0, 0.0)  # dangling leaf
        n0.fixed_x = True
        n0.fixed_z = True
        n1.fz = -1.0  # load on n1
        s.add_spring(n0, n1)
        s.add_spring(n1, n2)
        removed = s.remove_dangling_nodes()
        assert removed == 1
        assert n2.id not in s.graph

    def test_cascading_removal(self):
        """Removing one leaf should expose and remove the next."""
        s = Structure()
        n0 = s.add_node(0.0, 0.0)
        n1 = s.add_node(1.0, 0.0)
        n2 = s.add_node(2.0, 0.0)
        n3 = s.add_node(3.0, 0.0)  # tail of dangling chain
        n0.fixed_x = True
        n0.fixed_z = True
        n0.fz = -1.0
        s.add_spring(n0, n1)
        s.add_spring(n1, n2)
        s.add_spring(n2, n3)
        # n3 is leaf -> removed -> n2 becomes leaf -> removed -> n1 too
        removed = s.remove_dangling_nodes()
        assert removed == 3
        assert n1.id not in s.graph
        assert n2.id not in s.graph
        assert n3.id not in s.graph
        # Only the protected node remains
        assert s.num_nodes == 1

    def test_protected_leaf_kept(self):
        """A degree-1 node with a load should NOT be removed."""
        s = Structure()
        n0 = s.add_node(0.0, 0.0)
        n1 = s.add_node(1.0, 0.0)
        n0.fixed_x = True
        n0.fixed_z = True
        n1.fz = -1.0  # load -> protected
        s.add_spring(n0, n1)
        removed = s.remove_dangling_nodes()
        assert removed == 0
        assert s.num_nodes == 2

    def test_no_dangling_in_grid(self, rect_2x1):
        """A full rectangular grid has no dangling nodes."""
        removed = rect_2x1.remove_dangling_nodes()
        assert removed == 0

    def test_branch_with_loop_at_tip(self):
        """Branch 1-2-3-4-5-6-5 (loop at tip) should be removed.

        Main body is a triangle (A-B-C) with supports/loads.
        Branch hangs off node A: A-1-2-3-4-5-6, with 6 looping back to 5.
        """
        s = Structure()
        # Main body triangle
        a = s.add_node(0.0, 0.0)
        b = s.add_node(2.0, 0.0)
        c = s.add_node(1.0, 1.0)
        a.fixed_x = True
        a.fixed_z = True
        b.fixed_x = True
        b.fixed_z = True
        c.fz = -1.0
        s.add_spring(a, b)
        s.add_spring(b, c)
        s.add_spring(c, a)

        # Dangling branch off node a: a-n1-n2-n3-n4-n5-n6, n6 back to n5
        n1 = s.add_node(-1.0, 0.0)
        n2 = s.add_node(-2.0, 0.0)
        n3 = s.add_node(-3.0, 0.0)
        n4 = s.add_node(-4.0, 0.0)
        n5 = s.add_node(-5.0, 0.0)
        n6 = s.add_node(-5.0, 1.0)
        s.add_spring(a, n1)
        s.add_spring(n1, n2)
        s.add_spring(n2, n3)
        s.add_spring(n3, n4)
        s.add_spring(n4, n5)
        s.add_spring(n5, n6)
        s.add_spring(n6, n5)  # loop back

        removed = s.remove_dangling_nodes()
        assert removed == 6
        # Main triangle should remain intact
        assert a.id in s.graph
        assert b.id in s.graph
        assert c.id in s.graph
        assert s.num_nodes == 3

    def test_branch_with_larger_loop(self):
        """Branch 1-2-3-4-5-6-7-5 (triangle loop at tip)."""
        s = Structure()
        # Main body
        a = s.add_node(0.0, 0.0)
        b = s.add_node(2.0, 0.0)
        a.fixed_x = True
        a.fixed_z = True
        b.fixed_x = True
        b.fixed_z = True
        b.fz = -1.0
        s.add_spring(a, b)

        # Branch off a
        n1 = s.add_node(-1.0, 0.0)
        n2 = s.add_node(-2.0, 0.0)
        n3 = s.add_node(-3.0, 0.0)
        n4 = s.add_node(-4.0, 0.0)
        n5 = s.add_node(-5.0, 0.0)
        n6 = s.add_node(-5.0, 1.0)
        n7 = s.add_node(-4.0, 1.0)
        s.add_spring(a, n1)
        s.add_spring(n1, n2)
        s.add_spring(n2, n3)
        s.add_spring(n3, n4)
        s.add_spring(n4, n5)
        s.add_spring(n5, n6)
        s.add_spring(n6, n7)
        s.add_spring(n7, n5)  # loop back to n5

        removed = s.remove_dangling_nodes()
        assert removed == 7
        assert s.num_nodes == 2  # only a, b remain


class TestSerialization:
    """Test to_dict / from_dict round-trip."""

    def test_round_trip(self, rect_2x1):
        d = rect_2x1.to_dict()
        restored = Structure.from_dict(d)
        assert restored.num_nodes == rect_2x1.num_nodes
        assert restored.graph.number_of_edges() == rect_2x1.graph.number_of_edges()

    def test_serialized_structure_fields(self, rect_2x1):
        d = rect_2x1.to_dict()
        assert "nodes" in d
        assert "springs" in d
        assert len(d["nodes"]) == 6
        assert len(d["springs"]) == 11

    def test_repr(self, rect_2x1):
        r = repr(rect_2x1)
        assert "Structure" in r
        assert "6" in r  # nodes
