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
        restored, result, initial = structure_from_json_string(s)
        assert restored.num_nodes == sample_structure.num_nodes
        assert result is None
        assert initial is None

    def test_round_trip_string(self, sample_structure):
        json_str = state_to_json_string(sample_structure)
        restored, _, _ = structure_from_json_string(json_str)
        assert restored.num_nodes == sample_structure.num_nodes
        assert restored.graph.number_of_edges() == sample_structure.graph.number_of_edges()

    def test_string_round_trip_preserves_bcs(self, sample_structure):
        json_str = state_to_json_string(sample_structure)
        restored, _, _ = structure_from_json_string(json_str)
        n0 = restored.get_node(0)
        assert n0.fixed_x is True
        assert n0.fixed_z is True


class TestSIMPResultSerialisation:
    """Test that SIMP optimisation results survive a save/load cycle."""

    def test_simp_result_round_trip(self, sample_structure):
        from src.solver.optimizer_base import OptimizationResult

        result = OptimizationResult(algorithm="SIMP")
        result.iterations = 5
        result.penalization = 3.0
        result.compliance_history = [100.0, 80.0, 60.0]
        springs = sample_structure.get_springs()
        result.densities = {sp.node_ids: 0.5 for sp in springs}
        result.density_history = [dict(result.densities)]

        json_str = state_to_json_string(sample_structure, result=result)
        loaded_struct, loaded_result, _ = structure_from_json_string(json_str)

        assert loaded_struct.num_nodes == sample_structure.num_nodes
        assert loaded_result is not None
        assert loaded_result.algorithm == "SIMP"
        assert loaded_result.iterations == 5
        assert loaded_result.compliance_history == pytest.approx([100.0, 80.0, 60.0])
        assert loaded_result.densities is not None
        assert len(loaded_result.densities) == len(result.densities)
        assert len(loaded_result.density_history) == 1

    def test_simp_densities_on_structure_round_trip(self, sample_structure):
        """SIMP density data attached to the structure itself is preserved."""
        springs = sample_structure.get_springs()
        sample_structure._simp_densities = {
            sp.node_ids: 0.7 for sp in springs
        }
        sample_structure._simp_spring_volumes = {
            sp.node_ids: sp.length for sp in springs
        }
        json_str = state_to_json_string(sample_structure)
        loaded, _, _ = structure_from_json_string(json_str)
        assert hasattr(loaded, "_simp_densities")
        assert len(loaded._simp_densities) == len(springs)


class TestInitialStructureSerialisation:
    """Test that the initial (pre-optimisation) structure is preserved."""

    def test_initial_structure_round_trip(self, sample_structure):
        """Saving with an initial_structure restores it on load."""
        # Simulate INR: remove a node from the current structure.
        optimized = sample_structure.snapshot()
        removable = [
            n.id for n in optimized.get_nodes()
            if not n.is_fixed and not n.has_load
        ]
        if removable:
            optimized.remove_node(removable[0])

        json_str = state_to_json_string(
            optimized,
            initial_structure=sample_structure,
        )
        loaded_current, _, loaded_initial = structure_from_json_string(json_str)

        assert loaded_current.num_nodes == optimized.num_nodes
        assert loaded_initial is not None
        assert loaded_initial.num_nodes == sample_structure.num_nodes

    def test_no_initial_structure_returns_none(self, sample_structure):
        json_str = state_to_json_string(sample_structure)
        _, _, loaded_initial = structure_from_json_string(json_str)
        assert loaded_initial is None
