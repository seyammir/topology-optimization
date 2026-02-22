"""Unit tests for src.presets.mbb_beam."""

from __future__ import annotations

import pytest

from src.presets.mbb_beam import create_mbb_beam
from src.models.structure import Structure


class TestMBBBeamFull:
    """Test the full MBB beam preset."""

    def test_creates_structure(self):
        s = create_mbb_beam(nx=6, nz=3)
        assert isinstance(s, Structure)

    def test_correct_node_count(self):
        nx, nz = 6, 3
        s = create_mbb_beam(nx=nx, nz=nz)
        expected_nodes = (nx + 1) * (nz + 1)
        assert s.num_nodes == expected_nodes

    def test_bottom_left_pinned(self):
        nx, nz = 6, 3
        s = create_mbb_beam(nx=nx, nz=nz)
        cols = nx + 1
        bl_id = nz * cols  # bottom-left
        bl = s.get_node(bl_id)
        assert bl.fixed_x is True
        assert bl.fixed_z is True

    def test_bottom_right_roller(self):
        nx, nz = 6, 3
        s = create_mbb_beam(nx=nx, nz=nz)
        cols = nx + 1
        br_id = nz * cols + nx
        br = s.get_node(br_id)
        assert br.fixed_z is True
        # Roller: x is free
        assert br.fixed_x is False

    def test_top_center_loaded(self):
        nx, nz = 6, 3
        s = create_mbb_beam(nx=nx, nz=nz)
        mid_x = nx // 2
        tc_id = mid_x  # top row, middle column
        tc = s.get_node(tc_id)
        assert tc.fz != 0.0

    def test_connected(self):
        s = create_mbb_beam(nx=6, nz=3)
        assert s.is_connected()

    def test_supports_reachable(self):
        s = create_mbb_beam(nx=6, nz=3)
        assert s.supports_reachable_from_loads()

    def test_custom_load(self):
        s = create_mbb_beam(nx=4, nz=2, load=-2.0)
        mid_x = 4 // 2
        tc = s.get_node(mid_x)
        assert tc.fz == pytest.approx(2.0)  # abs(load)


class TestMBBBeamHalf:
    """Test the half-symmetry MBB beam preset."""

    def test_creates_structure(self):
        s = create_mbb_beam(nx=6, nz=3, half=True)
        assert isinstance(s, Structure)

    def test_left_edge_fixed_x(self):
        nx, nz = 6, 3
        s = create_mbb_beam(nx=nx, nz=nz, half=True)
        cols = nx + 1
        rows = nz + 1
        for iz in range(rows):
            nid = iz * cols
            node = s.get_node(nid)
            assert node.fixed_x is True, f"Node {nid} at left edge should have fixed_x"

    def test_bottom_right_roller_z(self):
        nx, nz = 6, 3
        s = create_mbb_beam(nx=nx, nz=nz, half=True)
        cols = nx + 1
        br_id = nz * cols + nx
        br = s.get_node(br_id)
        assert br.fixed_z is True

    def test_top_left_loaded(self):
        nx, nz = 6, 3
        s = create_mbb_beam(nx=nx, nz=nz, half=True)
        tl = s.get_node(0)
        assert tl.fz != 0.0

    def test_connected(self):
        s = create_mbb_beam(nx=6, nz=3, half=True)
        assert s.is_connected()

    def test_supports_reachable(self):
        s = create_mbb_beam(nx=6, nz=3, half=True)
        assert s.supports_reachable_from_loads()
