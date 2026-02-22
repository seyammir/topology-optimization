"""Unit tests for src.utils.io_handler."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.models.structure import Structure
from src.utils.io_handler import (
    save_state,
    load_state,
    state_to_json_string,
    structure_from_json_string,
)


@pytest.fixture
def sample_structure() -> Structure:
    """A small structure with BCs for serialization testing."""
    s = Structure.create_rectangular(2, 1)
    n = s.get_node(0)
    n.fixed_x = True
    n.fixed_z = True
    n5 = s.get_node(5)
    n5.fz = -1.0
    return s


class TestSaveLoad:
    """Test file-based save/load."""

    def test_save_creates_file(self, sample_structure):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            save_state(sample_structure, path)
            assert path.exists()

    def test_load_restores_structure(self, sample_structure):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            save_state(sample_structure, path)
            loaded = load_state(path)
            assert loaded.num_nodes == sample_structure.num_nodes
            assert loaded.graph.number_of_edges() == sample_structure.graph.number_of_edges()

    def test_round_trip_preserves_bcs(self, sample_structure):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            save_state(sample_structure, path)
            loaded = load_state(path)
            n0 = loaded.get_node(0)
            assert n0.fixed_x is True
            assert n0.fixed_z is True
            n5 = loaded.get_node(5)
            assert n5.fz == pytest.approx(-1.0)

    def test_save_creates_directories(self, sample_structure):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "state.json"
            save_state(sample_structure, path)
            assert path.exists()

    def test_saved_file_is_valid_json(self, sample_structure):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            save_state(sample_structure, path)
            with open(path, "r") as f:
                data = json.load(f)
            assert "nodes" in data
            assert "springs" in data


class TestStringSerialisation:
    """Test JSON-string based serialisation."""

    def test_to_json_string(self, sample_structure):
        s = state_to_json_string(sample_structure)
        assert isinstance(s, str)
        data = json.loads(s)
        assert "nodes" in data

    def test_from_json_string(self, sample_structure):
        s = state_to_json_string(sample_structure)
        restored = structure_from_json_string(s)
        assert restored.num_nodes == sample_structure.num_nodes

    def test_round_trip_string(self, sample_structure):
        json_str = state_to_json_string(sample_structure)
        restored = structure_from_json_string(json_str)
        assert restored.num_nodes == sample_structure.num_nodes
        assert restored.graph.number_of_edges() == sample_structure.graph.number_of_edges()

    def test_string_round_trip_preserves_bcs(self, sample_structure):
        json_str = state_to_json_string(sample_structure)
        restored = structure_from_json_string(json_str)
        n0 = restored.get_node(0)
        assert n0.fixed_x is True
        assert n0.fixed_z is True
