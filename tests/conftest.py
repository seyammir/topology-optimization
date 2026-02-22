"""Shared fixtures for unit tests."""

from __future__ import annotations

import pytest

from src.models.node import Node
from src.models.spring import Spring
from src.models.structure import Structure


@pytest.fixture
def simple_node() -> Node:
    """A plain node at (1, 2) with default mass."""
    return Node(id=0, x=1.0, z=2.0)


@pytest.fixture
def fixed_node() -> Node:
    """A pinned node (both DOFs constrained)."""
    return Node(id=1, x=0.0, z=0.0, fixed_x=True, fixed_z=True)


@pytest.fixture
def loaded_node() -> Node:
    """A node with an external force applied."""
    return Node(id=2, x=5.0, z=0.0, fx=0.0, fz=-1.0)


@pytest.fixture
def horizontal_spring() -> Spring:
    """A horizontal spring of length 1 between (0,0) and (1,0)."""
    n0 = Node(id=0, x=0.0, z=0.0)
    n1 = Node(id=1, x=1.0, z=0.0)
    return Spring(n0, n1)


@pytest.fixture
def diagonal_spring() -> Spring:
    """A diagonal spring between (0,0) and (1,1)."""
    n0 = Node(id=0, x=0.0, z=0.0)
    n1 = Node(id=1, x=1.0, z=1.0)
    return Spring(n0, n1)


@pytest.fixture
def rect_2x1() -> Structure:
    """A small 2x1 rectangular structure (3x2 = 6 nodes)."""
    return Structure.create_rectangular(2, 1)


@pytest.fixture
def rect_3x2() -> Structure:
    """A 3x2 rectangular structure (4x3 = 12 nodes)."""
    return Structure.create_rectangular(3, 2)


@pytest.fixture
def cantilever_beam() -> Structure:
    """A 4x2 structure with left edge pinned, top-right loaded downward."""
    struct = Structure.create_rectangular(4, 2)
    # Pin left edge nodes (ids 0, 5, 10 for a 5-column grid)
    cols = 5
    rows = 3
    for iz in range(rows):
        nid = iz * cols
        node = struct.get_node(nid)
        node.fixed_x = True
        node.fixed_z = True
    # Load top-right corner
    top_right = struct.get_node(cols - 1)
    top_right.fz = 1.0
    return struct
