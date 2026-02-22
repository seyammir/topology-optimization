"""Unit tests for src.solver.optimizer.TopologyOptimizer."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.structure import Structure
from src.solver.fem_solver import FEMSolver
from src.solver.optimizer import TopologyOptimizer, OptimizationResult
from src.presets.mbb_beam import create_mbb_beam


class TestOptimizerCreation:
    """Test optimizer construction and validation."""

    def test_default_values(self):
        opt = TopologyOptimizer()
        assert opt.target_mass_fraction == 0.5
        assert opt.removal_per_iteration == 1
        assert opt.filter_radius == 1.5

    def test_custom_values(self):
        opt = TopologyOptimizer(
            target_mass_fraction=0.3,
            removal_per_iteration=5,
            filter_radius=2.0,
        )
        assert opt.target_mass_fraction == 0.3
        assert opt.removal_per_iteration == 5
        assert opt.filter_radius == 2.0

    def test_invalid_mass_fraction_zero(self):
        with pytest.raises(ValueError):
            TopologyOptimizer(target_mass_fraction=0.0)

    def test_invalid_mass_fraction_one(self):
        with pytest.raises(ValueError):
            TopologyOptimizer(target_mass_fraction=1.0)

    def test_invalid_mass_fraction_negative(self):
        with pytest.raises(ValueError):
            TopologyOptimizer(target_mass_fraction=-0.1)

    def test_removal_rate_clamped(self):
        opt = TopologyOptimizer(removal_per_iteration=0)
        assert opt.removal_per_iteration == 1

    def test_filter_radius_clamped(self):
        opt = TopologyOptimizer(filter_radius=-1.0)
        assert opt.filter_radius == 0.0


class TestComputeNodeEnergies:
    """Test the static _compute_node_energies method."""

    def test_zero_displacement(self):
        s = Structure()
        n0 = s.add_node(0.0, 0.0)
        n1 = s.add_node(1.0, 0.0)
        s.add_spring(n0, n1)
        s.renumber_dofs()
        u = np.zeros(4)
        energies = TopologyOptimizer._compute_node_energies(s, u)
        assert energies[n0.id] == 0.0
        assert energies[n1.id] == 0.0

    def test_nonzero_displacement(self):
        s = Structure()
        n0 = s.add_node(0.0, 0.0)
        n1 = s.add_node(1.0, 0.0)
        s.add_spring(n0, n1)
        s.renumber_dofs()
        u = np.array([0.0, 0.0, 1.0, 0.0])
        energies = TopologyOptimizer._compute_node_energies(s, u)
        # Both nodes share the spring -> each gets strain_energy / 1
        assert energies[n0.id] > 0
        assert energies[n1.id] > 0

    def test_energies_for_all_nodes(self, cantilever_beam):
        solver = FEMSolver()
        cantilever_beam.renumber_dofs()
        u = solver.solve(cantilever_beam)
        energies = TopologyOptimizer._compute_node_energies(cantilever_beam, u)
        assert len(energies) == cantilever_beam.num_nodes
        # All energies should be non-negative
        for e in energies.values():
            assert e >= 0.0


class TestFilterEnergies:
    """Test the spatial sensitivity filter."""

    def test_filter_with_very_small_radius(self):
        """With a tiny radius only the self-weight is non-zero -> values unchanged."""
        s = Structure()
        n0 = s.add_node(0.0, 0.0)
        n1 = s.add_node(1.0, 0.0)
        s.add_spring(n0, n1)
        raw = {n0.id: 1.0, n1.id: 2.0}
        # radius=0 gives all-zero weights (self-distance is 0, max(0,0-0)=0).
        # The optimizer skips the filter when radius <= 0, so test with a
        # tiny positive radius that only covers each node itself.
        filtered = TopologyOptimizer._filter_energies(s, raw, filter_radius=0.01)
        assert filtered[n0.id] == pytest.approx(1.0)
        assert filtered[n1.id] == pytest.approx(2.0)

    def test_filter_smooths_energies(self):
        s = Structure()
        n0 = s.add_node(0.0, 0.0)
        n1 = s.add_node(1.0, 0.0)
        n2 = s.add_node(2.0, 0.0)
        s.add_spring(n0, n1)
        s.add_spring(n1, n2)
        raw = {n0.id: 0.0, n1.id: 10.0, n2.id: 0.0}
        filtered = TopologyOptimizer._filter_energies(s, raw, filter_radius=1.5)
        # Middle node's value should decrease, end nodes should increase
        assert filtered[n1.id] < 10.0
        assert filtered[n0.id] > 0.0
        assert filtered[n2.id] > 0.0

    def test_filter_empty_structure(self):
        s = Structure()
        result = TopologyOptimizer._filter_energies(s, {}, filter_radius=3.0)
        assert result == {}


class TestRestoreNode:
    """Test node restoration after removal."""

    def test_restore_maintains_structure(self):
        s = Structure()
        n0 = s.add_node(0.0, 0.0)
        n1 = s.add_node(1.0, 0.0)
        n2 = s.add_node(0.5, 1.0)
        s.add_spring(n0, n1)
        s.add_spring(n0, n2)
        s.add_spring(n1, n2)

        initial_nodes = s.num_nodes
        initial_edges = s.graph.number_of_edges()

        # Backup and remove n2
        node_backup = s.get_node(n2.id)
        springs_backup = s.get_springs_for_node(n2.id)
        s.remove_node(n2.id)
        assert s.num_nodes == initial_nodes - 1

        # Restore
        TopologyOptimizer._restore_node(s, node_backup, springs_backup)
        assert s.num_nodes == initial_nodes
        assert s.graph.number_of_edges() == initial_edges


class TestSingleStep:
    """Test the step() method."""

    def test_step_returns_correct_types(self, cantilever_beam):
        opt = TopologyOptimizer(
            target_mass_fraction=0.5,
            removal_per_iteration=1,
            filter_radius=1.5,
        )
        u, energies, removed = opt.step(cantilever_beam)
        assert isinstance(u, np.ndarray)
        assert isinstance(energies, dict)
        assert isinstance(removed, int)

    def test_step_removes_at_most_n_nodes(self, cantilever_beam):
        n = 2
        opt = TopologyOptimizer(
            target_mass_fraction=0.5,
            removal_per_iteration=n,
            filter_radius=1.5,
        )
        initial_count = cantilever_beam.num_nodes
        _, _, removed = opt.step(cantilever_beam)
        assert removed <= n
        assert cantilever_beam.num_nodes >= initial_count - n

    def test_step_does_not_remove_protected(self, cantilever_beam):
        opt = TopologyOptimizer(
            target_mass_fraction=0.5,
            removal_per_iteration=1,
            filter_radius=1.5,
        )
        protected_before = cantilever_beam.get_protected_node_ids()
        opt.step(cantilever_beam)
        # All protected nodes should still exist
        for nid in protected_before:
            assert nid in cantilever_beam.graph


class TestFullOptimization:
    """Test the optimize() method on a small structure."""

    def test_optimize_reaches_target(self):
        struct = create_mbb_beam(nx=6, nz=3, half=True)
        initial_mass = struct.total_mass()
        target_frac = 0.6
        opt = TopologyOptimizer(
            target_mass_fraction=target_frac,
            removal_per_iteration=2,
            filter_radius=1.5,
        )
        result = opt.optimize(struct)
        assert isinstance(result, OptimizationResult)
        assert result.iterations > 0
        assert struct.total_mass() <= target_frac * initial_mass + 1.0

    def test_optimize_result_has_history(self):
        struct = create_mbb_beam(nx=4, nz=2, half=True)
        opt = TopologyOptimizer(
            target_mass_fraction=0.7,
            removal_per_iteration=1,
            filter_radius=1.5,
        )
        result = opt.optimize(struct)
        assert len(result.history) >= 2  # at least initial + final
        assert len(result.compliance_history) > 0

    def test_callback_called(self):
        struct = create_mbb_beam(nx=4, nz=2, half=True)
        opt = TopologyOptimizer(
            target_mass_fraction=0.7,
            removal_per_iteration=1,
            filter_radius=1.5,
        )
        calls = []

        def cb(s, it, ne):
            calls.append(it)

        opt.optimize(struct, callback=cb)
        assert len(calls) > 0

    def test_structure_stays_connected(self):
        struct = create_mbb_beam(nx=6, nz=3, half=True)
        opt = TopologyOptimizer(
            target_mass_fraction=0.5,
            removal_per_iteration=2,
            filter_radius=1.5,
        )
        opt.optimize(struct)
        assert struct.is_connected()

    def test_supports_still_reachable(self):
        struct = create_mbb_beam(nx=6, nz=3, half=True)
        opt = TopologyOptimizer(
            target_mass_fraction=0.5,
            removal_per_iteration=2,
            filter_radius=1.5,
        )
        opt.optimize(struct)
        assert struct.supports_reachable_from_loads()
