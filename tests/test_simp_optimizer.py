"""Unit tests for src.solver.simp_optimizer.SIMPOptimizer."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.structure import Structure
from src.solver.fem_solver import FEMSolver
from src.solver.simp_optimizer import SIMPOptimizer
from src.solver.optimizer_base import OptimizationResult
from src.presets.mbb_beam import create_mbb_beam


class TestSIMPCreation:
    """Test SIMP optimizer construction and validation."""

    def test_default_values(self):
        opt = SIMPOptimizer()
        assert opt.target_mass_fraction == 0.5
        assert opt.penalization == 3.0
        assert opt.move_limit == 0.2
        assert opt.x_min == 1e-3
        assert opt.convergence_tol == 0.01

    def test_custom_values(self):
        opt = SIMPOptimizer(
            target_mass_fraction=0.4,
            penalization=4.0,
            move_limit=0.1,
            x_min=0.01,
            convergence_tol=0.005,
        )
        assert opt.target_mass_fraction == 0.4
        assert opt.penalization == 4.0
        assert opt.move_limit == 0.1

    def test_invalid_mass_fraction(self):
        with pytest.raises(ValueError):
            SIMPOptimizer(target_mass_fraction=0.0)
        with pytest.raises(ValueError):
            SIMPOptimizer(target_mass_fraction=1.0)


class TestSIMPSingleStep:
    """Test the SIMP step() method."""

    def test_step_returns_correct_types(self, cantilever_beam):
        opt = SIMPOptimizer(target_mass_fraction=0.5, filter_radius=1.5)
        u, energies, count = opt.step(cantilever_beam)
        assert isinstance(u, np.ndarray)
        assert isinstance(energies, dict)
        assert count == 0  # SIMP never removes nodes

    def test_step_preserves_structure(self, cantilever_beam):
        initial_nodes = cantilever_beam.num_nodes
        initial_springs = cantilever_beam.graph.number_of_edges()
        opt = SIMPOptimizer(target_mass_fraction=0.5)
        opt.step(cantilever_beam)
        assert cantilever_beam.num_nodes == initial_nodes
        assert cantilever_beam.graph.number_of_edges() == initial_springs


class TestSIMPFullOptimization:
    """Test the SIMP optimize() method."""

    def test_optimize_returns_result(self):
        struct = create_mbb_beam(nx=4, nz=2, half=True)
        opt = SIMPOptimizer(
            target_mass_fraction=0.5,
            filter_radius=1.5,
            max_iterations=20,
        )
        result = opt.optimize(struct)
        assert isinstance(result, OptimizationResult)
        assert result.iterations > 0
        assert result.algorithm == "SIMP"

    def test_densities_returned(self):
        struct = create_mbb_beam(nx=4, nz=2, half=True)
        opt = SIMPOptimizer(
            target_mass_fraction=0.5,
            max_iterations=10,
        )
        result = opt.optimize(struct)
        assert result.densities is not None
        assert len(result.densities) > 0

    def test_densities_within_bounds(self):
        struct = create_mbb_beam(nx=4, nz=2, half=True)
        opt = SIMPOptimizer(
            target_mass_fraction=0.5,
            x_min=0.001,
            max_iterations=10,
        )
        result = opt.optimize(struct)
        for xe in result.densities.values():
            assert xe >= opt.x_min - 1e-12
            assert xe <= 1.0 + 1e-12

    def test_compliance_history_populated(self):
        struct = create_mbb_beam(nx=4, nz=2, half=True)
        opt = SIMPOptimizer(
            target_mass_fraction=0.5,
            max_iterations=10,
        )
        result = opt.optimize(struct)
        assert len(result.compliance_history) > 0
        # All compliance values >= 0
        for c in result.compliance_history:
            assert c >= 0.0

    def test_callback_called(self):
        struct = create_mbb_beam(nx=4, nz=2, half=True)
        opt = SIMPOptimizer(
            target_mass_fraction=0.5,
            max_iterations=5,
        )
        calls = []

        def cb(s, it, ne):
            calls.append(it)

        opt.optimize(struct, callback=cb)
        assert len(calls) > 0

    def test_structure_unchanged_after_simp(self):
        """SIMP does not modify the structure topology."""
        struct = create_mbb_beam(nx=4, nz=2, half=True)
        initial_nodes = struct.num_nodes
        initial_springs = struct.graph.number_of_edges()
        opt = SIMPOptimizer(
            target_mass_fraction=0.5,
            max_iterations=10,
        )
        opt.optimize(struct)
        assert struct.num_nodes == initial_nodes
        assert struct.graph.number_of_edges() == initial_springs

    def test_density_history_populated(self):
        """optimize() records one density snapshot per iteration."""
        struct = create_mbb_beam(nx=4, nz=2, half=True)
        opt = SIMPOptimizer(
            target_mass_fraction=0.5,
            max_iterations=10,
        )
        result = opt.optimize(struct)
        assert len(result.density_history) == result.iterations
        # Each snapshot should have the same keys as the final densities
        for snap in result.density_history:
            assert set(snap.keys()) == set(result.densities.keys())
            for xe in snap.values():
                assert 0.0 <= xe <= 1.0 + 1e-12