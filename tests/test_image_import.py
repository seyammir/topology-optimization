"""Tests for src.utils.image_import — image-to-structure conversion."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src.utils.image_import import structure_from_image


# Helpers
def _make_bw_image(width: int, height: int, *, black: bool = True) -> io.BytesIO:
    """Create a solid B&W image and return it as a BytesIO PNG."""
    colour = 0 if black else 255
    img = Image.new("L", (width, height), color=colour)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _make_checkerboard_image(cols: int, rows: int) -> io.BytesIO:
    """Create a checkerboard pattern (alternating black/white pixels)."""
    arr = np.zeros((rows, cols), dtype=np.uint8)
    for iz in range(rows):
        for ix in range(cols):
            if (ix + iz) % 2 == 0:
                arr[iz, ix] = 0    # black
            else:
                arr[iz, ix] = 255  # white
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _make_half_image(cols: int, rows: int) -> io.BytesIO:
    """Left half black, right half white."""
    arr = np.full((rows, cols), 255, dtype=np.uint8)
    arr[:, : cols // 2] = 0
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# Tests
class TestStructureFromImage:
    """Tests for structure_from_image()."""

    def test_all_black_equals_rectangular(self):
        """A fully black image should create the same number of nodes as
        Structure.create_rectangular (all grid points present)."""
        w, h = 4, 3
        buf = _make_bw_image(200, 200, black=True)
        struct = structure_from_image(buf, width=w, height=h)
        expected_nodes = (w + 1) * (h + 1)
        assert struct.num_nodes == expected_nodes

    def test_all_white_raises(self):
        """An all-white image should raise ValueError (no material)."""
        buf = _make_bw_image(50, 50, black=False)
        with pytest.raises(ValueError, match="no material"):
            structure_from_image(buf, width=4, height=3)

    def test_half_image_fewer_nodes(self):
        """A half-black image should produce fewer nodes than full grid."""
        w, h = 10, 5
        buf = _make_half_image(200, 200)
        struct = structure_from_image(buf, width=w, height=h)
        full_nodes = (w + 1) * (h + 1)
        assert 0 < struct.num_nodes < full_nodes

    def test_structure_has_springs(self):
        """Material nodes should be connected by springs."""
        buf = _make_bw_image(100, 100, black=True)
        struct = structure_from_image(buf, width=5, height=5)
        assert struct.graph.number_of_edges() > 0

    def test_dofs_renumbered(self):
        """All nodes should have valid DOF indices after import."""
        buf = _make_bw_image(100, 100, black=True)
        struct = structure_from_image(buf, width=3, height=3)
        for node in struct.get_nodes():
            dof_x, dof_z = node.dof_indices
            assert dof_x >= 0
            assert dof_z >= 0

    def test_invalid_dimensions(self):
        """width/height < 1 should raise ValueError."""
        buf = _make_bw_image(50, 50, black=True)
        with pytest.raises(ValueError, match="width and height"):
            structure_from_image(buf, width=0, height=3)
        buf.seek(0)
        with pytest.raises(ValueError, match="width and height"):
            structure_from_image(buf, width=3, height=0)

    def test_checkerboard_pattern(self):
        """Checkerboard should create roughly half the nodes."""
        w, h = 6, 6
        cols, rows = w + 1, h + 1
        buf = _make_checkerboard_image(cols, rows)
        struct = structure_from_image(buf, width=w, height=h, threshold=128)
        total = cols * rows
        # Expect roughly half (±1) of the grid positions are material
        assert total // 2 - 1 <= struct.num_nodes <= total // 2 + 1

    def test_threshold_low_means_more_void(self):
        """A very low threshold should classify most grey pixels as void."""
        # Create a medium-grey image (128)
        arr = np.full((50, 50), 128, dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        # threshold=128 -> 128 < 128 is False -> all void
        buf.seek(0)
        with pytest.raises(ValueError, match="no material"):
            structure_from_image(buf, width=4, height=4, threshold=128)

        # threshold=129 -> 128 < 129 is True -> all material
        buf.seek(0)
        struct = structure_from_image(buf, width=4, height=4, threshold=129)
        assert struct.num_nodes == 25

    def test_rgb_image_works(self):
        """An RGB image should be auto-converted to greyscale."""
        img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        struct = structure_from_image(buf, width=3, height=3)
        assert struct.num_nodes == 16  # 4x4 grid, all material

    def test_serialisation_roundtrip(self):
        """Import from image, serialise to dict, and reconstruct."""
        from src.models.structure import Structure as S
        buf = _make_bw_image(100, 100, black=True)
        original = structure_from_image(buf, width=4, height=3)
        data = original.to_dict()
        restored = S.from_dict(data)
        assert restored.num_nodes == original.num_nodes
        assert restored.graph.number_of_edges() == original.graph.number_of_edges()
