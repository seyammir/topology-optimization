"""Tests for the report_generator module."""

from __future__ import annotations

import pytest
import numpy as np

from src.models.structure import Structure
from src.solver.fem_solver import FEMSolver
from src.solver.optimizer_base import OptimizationResult
from src.utils.report_generator import generate_report


@pytest.fixture
def cantilever_with_result():
    """Return a solved cantilever beam with an OptimizationResult."""
    struct = Structure.create_rectangular(4, 2)
    cols = 5
    rows = 3
    for iz in range(rows):
        nid = iz * cols
        node = struct.get_node(nid)
        node.fixed_x = True
        node.fixed_z = True
    top_right = struct.get_node(cols - 1)
    top_right.fz = -1.0

    initial = struct.snapshot()

    struct.renumber_dofs()
    solver = FEMSolver()
    u = solver.solve(struct)

    result = OptimizationResult(algorithm="Node Removal (INR)")
    result.history.append(initial)
    result.history.append(struct.snapshot())
    result.compliance_history = [1.0, 0.8, 0.75]
    result.iterations = 3

    # Compute node energies (simple proxy)
    node_energies = {}
    for n in struct.get_nodes():
        ix, iz = n.dof_indices
        if ix >= 0 and iz >= 0 and ix < len(u) and iz < len(u):
            node_energies[n.id] = float(u[ix] ** 2 + u[iz] ** 2)
        else:
            node_energies[n.id] = 0.0

    return {
        "initial_structure": initial,
        "final_structure": struct,
        "result": result,
        "displacement": u,
        "node_energies": node_energies,
    }


class TestGenerateReport:
    """Tests for generate_report()."""

    def test_returns_html_string(self, cantilever_with_result):
        html = generate_report(
            initial_structure=cantilever_with_result["initial_structure"],
            final_structure=cantilever_with_result["final_structure"],
            result=cantilever_with_result["result"],
            displacement=cantilever_with_result["displacement"],
            node_energies=cantilever_with_result["node_energies"],
            algorithm="Node Removal (INR)",
            parameters={
                "target_mass_fraction": 0.5,
                "filter_radius": 1.5,
                "removal_per_iteration": 3,
            },
            version="1.0.0",
        )
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "Topology Optimisation" in html

    def test_contains_key_sections(self, cantilever_with_result):
        html = generate_report(
            initial_structure=cantilever_with_result["initial_structure"],
            final_structure=cantilever_with_result["final_structure"],
            result=cantilever_with_result["result"],
            displacement=cantilever_with_result["displacement"],
            node_energies=cantilever_with_result["node_energies"],
            algorithm="Node Removal (INR)",
            parameters={"target_mass_fraction": 0.5, "filter_radius": 1.5},
            version="1.0.0",
        )
        assert "Problem Setup" in html
        assert "Initial Structure" in html
        assert "Optimised Structure" in html
        assert "Deformation" in html
        assert "Strain Energy" in html
        assert "Compliance History" in html
        assert "Black" in html  # B/W Density section

    def test_contains_embedded_images(self, cantilever_with_result):
        html = generate_report(
            initial_structure=cantilever_with_result["initial_structure"],
            final_structure=cantilever_with_result["final_structure"],
            result=cantilever_with_result["result"],
            displacement=cantilever_with_result["displacement"],
            node_energies=cantilever_with_result["node_energies"],
            algorithm="Node Removal (INR)",
            parameters={"target_mass_fraction": 0.5, "filter_radius": 1.5},
        )
        assert "data:image/png;base64," in html

    def test_parameters_in_report(self, cantilever_with_result):
        html = generate_report(
            initial_structure=cantilever_with_result["initial_structure"],
            final_structure=cantilever_with_result["final_structure"],
            result=cantilever_with_result["result"],
            displacement=cantilever_with_result["displacement"],
            node_energies=cantilever_with_result["node_energies"],
            algorithm="Node Removal (INR)",
            parameters={
                "target_mass_fraction": 0.5,
                "filter_radius": 1.5,
                "removal_per_iteration": 3,
            },
        )
        assert "50%" in html  # target mass fraction
        assert "1.5" in html  # filter radius
        assert "Node Removal" in html

    def test_no_result_minimal_report(self):
        struct = Structure.create_rectangular(3, 2)
        struct.renumber_dofs()
        solver = FEMSolver()

        # Pin bottom-left
        n0 = struct.get_node(0)
        n0.fixed_x = True
        n0.fixed_z = True
        # Pin bottom-right
        n3 = struct.get_node(3)
        n3.fixed_x = True
        n3.fixed_z = True
        # Load top-middle
        n7 = struct.get_node(7)
        n7.fz = -1.0

        struct.renumber_dofs()
        u = solver.solve(struct)

        html = generate_report(
            initial_structure=struct.snapshot(),
            final_structure=struct,
            result=None,
            displacement=u,
            node_energies=None,
            algorithm="Node Removal (INR)",
            parameters={"target_mass_fraction": 0.5, "filter_radius": 1.5},
        )
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        # Should still have basic sections
        assert "Optimised Structure" in html

    def test_simp_report(self, cantilever_with_result):
        """Report with SIMP-style densities."""
        struct = cantilever_with_result["final_structure"]
        springs = struct.get_springs()
        densities = {sp.node_ids: 0.8 for sp in springs}

        result = OptimizationResult(algorithm="SIMP")
        result.densities = densities
        result.compliance_history = [2.0, 1.5, 1.2]
        result.iterations = 3
        result.history.append(struct.snapshot())

        html = generate_report(
            initial_structure=cantilever_with_result["initial_structure"],
            final_structure=struct,
            result=result,
            displacement=cantilever_with_result["displacement"],
            node_energies=cantilever_with_result["node_energies"],
            algorithm="SIMP",
            parameters={
                "target_mass_fraction": 0.5,
                "filter_radius": 1.5,
                "penalization": 3.0,
                "move_limit": 0.2,
                "convergence_tol": 0.01,
                "max_iterations": 200,
            },
        )
        assert "SIMP" in html
        assert "Density Field" in html

    def test_comparison_report(self, cantilever_with_result):
        """Report with comparison results."""
        struct = cantilever_with_result["final_structure"]

        res_inr = OptimizationResult(algorithm="Node Removal (INR)")
        res_inr.compliance_history = [1.0, 0.8]
        res_inr.iterations = 2
        res_inr.history.append(struct.snapshot())

        res_simp = OptimizationResult(algorithm="SIMP")
        res_simp.compliance_history = [1.0, 0.7]
        res_simp.iterations = 5
        res_simp.densities = {sp.node_ids: 0.5 for sp in struct.get_springs()}
        res_simp.history.append(struct.snapshot())

        comp = {
            "Node Removal (INR)": res_inr,
            "SIMP": res_simp,
        }

        html = generate_report(
            initial_structure=cantilever_with_result["initial_structure"],
            final_structure=struct,
            result=res_inr,
            displacement=cantilever_with_result["displacement"],
            node_energies=cantilever_with_result["node_energies"],
            algorithm="Node Removal (INR)",
            parameters={"target_mass_fraction": 0.5, "filter_radius": 1.5},
            comparison_results=comp,
        )
        assert "Algorithm Comparison" in html
        assert "Node Removal" in html
        assert "SIMP" in html

    def test_version_in_report(self, cantilever_with_result):
        html = generate_report(
            initial_structure=cantilever_with_result["initial_structure"],
            final_structure=cantilever_with_result["final_structure"],
            result=cantilever_with_result["result"],
            displacement=cantilever_with_result["displacement"],
            node_energies=cantilever_with_result["node_energies"],
            algorithm="Node Removal (INR)",
            parameters={"target_mass_fraction": 0.5, "filter_radius": 1.5},
            version="1.0.0",
        )
        assert "v1.0.0" in html
