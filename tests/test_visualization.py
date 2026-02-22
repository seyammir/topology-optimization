"""Unit tests for src.utils.visualization.Visualizer."""

from __future__ import annotations

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.models.structure import Structure
from src.solver.fem_solver import FEMSolver
from src.utils.visualization import Visualizer


@pytest.fixture
def small_struct() -> Structure:
    """A 3x2 structure with BCs for visualisation tests."""
    s = Structure.create_rectangular(3, 2)
    n0 = s.get_node(0)
    n0.fixed_x = True
    n0.fixed_z = True
    # Load a node
    n = s.get_node(3)
    n.fz = 1.0
    return s


class TestPlotStructure:
    """Test the main structure plotting function."""

    def test_returns_figure(self, small_struct):
        fig = Visualizer.plot_structure(small_struct, title="Test")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_displacement(self, small_struct):
        solver = FEMSolver()
        small_struct.renumber_dofs()
        u = solver.solve(small_struct)
        fig = Visualizer.plot_structure(small_struct, u=u, scale=10.0)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_auto_scale(self, small_struct):
        solver = FEMSolver()
        small_struct.renumber_dofs()
        u = solver.solve(small_struct)
        fig = Visualizer.plot_structure(small_struct, u=u, scale=0)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_node_ids(self, small_struct):
        fig = Visualizer.plot_structure(small_struct, show_node_ids=True)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_on_existing_axes(self, small_struct):
        fig, ax = plt.subplots()
        result = Visualizer.plot_structure(small_struct, ax=ax)
        assert result is fig
        plt.close(fig)


class TestPlotEnergyHeatmap:
    """Test the heatmap plotting."""

    def test_returns_figure(self, small_struct):
        energies = {nid: float(i) for i, nid in enumerate(small_struct.get_node_ids())}
        fig = Visualizer.plot_energy_heatmap(small_struct, energies)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_empty_energies(self, small_struct):
        fig = Visualizer.plot_energy_heatmap(small_struct, {})
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_uniform_energies(self, small_struct):
        energies = {nid: 1.0 for nid in small_struct.get_node_ids()}
        fig = Visualizer.plot_energy_heatmap(small_struct, energies)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_on_existing_axes(self, small_struct):
        energies = {nid: float(i) for i, nid in enumerate(small_struct.get_node_ids())}
        fig, ax = plt.subplots()
        result = Visualizer.plot_energy_heatmap(small_struct, energies, ax=ax)
        assert result is fig
        plt.close(fig)


class TestExportHelpers:
    """Test figure export utilities."""

    def test_fig_to_png_bytes(self, small_struct):
        fig = Visualizer.plot_structure(small_struct)
        png = Visualizer.fig_to_png_bytes(fig)
        assert isinstance(png, bytes)
        assert len(png) > 0
        # PNG magic bytes
        assert png[:4] == b"\x89PNG"
        plt.close(fig)

    def test_export_image(self, small_struct, tmp_path):
        fig = Visualizer.plot_structure(small_struct)
        path = str(tmp_path / "test_output.png")
        Visualizer.export_image(fig, path)
        import os
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
        plt.close(fig)
