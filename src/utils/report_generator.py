"""Final report generator for topology optimisation results.

Produces a self-contained HTML report with embedded images (base64)
that can be downloaded from the Streamlit UI.  No external
dependencies beyond *matplotlib* and the standard library are
required.
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..models.structure import Structure
from ..solver.optimizer_base import OptimizationResult
from ..solver.fem_solver import FEMSolver
from .visualization import Visualizer

logger = logging.getLogger(__name__)


def _fig_to_base64(fig: Figure, dpi: int = 150) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return encoded


def _embed_img(b64: str, alt: str = "", width: str = "100%") -> str:
    """Return an <img> tag with embedded base64 data."""
    return (
        f'<img src="data:image/png;base64,{b64}" '
        f'alt="{alt}" style="max-width:{width}; height:auto;" />'
    )


# Public API


def generate_report(
    *,
    initial_structure: Structure | None,
    final_structure: Structure,
    result: OptimizationResult | None,
    displacement: np.ndarray | None,
    node_energies: dict[int, float] | None,
    algorithm: str,
    parameters: dict[str, Any],
    comparison_results: dict[str, OptimizationResult] | None = None,
    version: str = "",
) -> str:
    """Build a self-contained HTML report string.

    Parameters
    ----------
    initial_structure : Structure | None
        The structure before any optimisation.
    final_structure : Structure
        The structure after optimisation (or the current state).
    result : OptimizationResult | None
        Full optimisation result container.
    displacement : ndarray | None
        Final displacement vector.
    node_energies : dict | None
        Per-node strain energies.
    algorithm : str
        Name of the algorithm used (``"Node Removal (INR)"`` / ``"SIMP"``).
    parameters : dict
        Sidebar parameter snapshot (target fraction, filter radius, ...).
    comparison_results : dict | None
        If a comparison run was executed, the dict ``{algo_name: result}``.
    version : str
        Application version string.

    Returns
    -------
    str
        Complete HTML document as a string.
    """
    logger.info("Generating final report ...")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    is_simp = (
        algorithm == "SIMP"
        and result is not None
        and result.densities is not None
    )
    densities = result.densities if is_simp else None

    sections: list[str] = []

    # 1. Header / meta
    sections.append(_section_header(now, version))

    # 2. Problem setup
    sections.append(_section_parameters(
        algorithm, parameters, initial_structure, final_structure, is_simp, result,
    ))

    # 3. Initial structure
    if initial_structure is not None:
        try:
            fig = Visualizer.plot_structure(initial_structure, title="Initial Structure")
            sections.append(_wrap_section(
                "Initial Structure", _embed_img(_fig_to_base64(fig), "Initial Structure"),
            ))
        except Exception:
            logger.exception("Report: failed to render initial structure")

    # 4. Optimised structure
    try:
        fig = Visualizer.plot_structure(
            final_structure,
            title="Optimised Structure",
            densities=densities,
        )
        sections.append(_wrap_section(
            "Optimised Structure", _embed_img(_fig_to_base64(fig), "Optimised Structure"),
        ))
    except Exception:
        logger.exception("Report: failed to render optimised structure")

    # 5. Deformed shape
    if displacement is not None:
        try:
            fig = Visualizer.plot_structure(
                final_structure,
                u=displacement,
                scale=0,
                title="Deformed Structure (auto-scaled)",
                densities=densities,
            )
            sections.append(_wrap_section(
                "Deformation", _embed_img(_fig_to_base64(fig), "Deformed Structure"),
            ))
        except Exception:
            logger.exception("Report: failed to render deformation plot")

    # 6. Strain-energy heatmap
    if node_energies is not None:
        try:
            fig = Visualizer.plot_energy_heatmap(
                final_structure, node_energies,
                title="Strain Energy Heatmap",
                densities=densities,
            )
            sections.append(_wrap_section(
                "Strain Energy Heatmap",
                _embed_img(_fig_to_base64(fig), "Strain Energy"),
            ))
        except Exception:
            logger.exception("Report: failed to render energy heatmap")

    # 7. Internal forcess
    if displacement is not None:
        try:
            final_structure.renumber_dofs()
            pen = result.penalization if result else 3.0
            spring_forces = FEMSolver.compute_internal_forces(
                final_structure, displacement,
                densities=densities, penalization=pen,
            )
            fig = Visualizer.plot_internal_forces(
                final_structure, spring_forces,
                title="Tension & Compression",
                densities=densities,
            )
            sections.append(_wrap_section(
                "Internal Forces",
                _embed_img(_fig_to_base64(fig), "Internal Forces"),
            ))
        except Exception:
            logger.exception("Report: failed to render internal forces plot")

    # 8. Black & white density
    try:
        if is_simp and densities is not None:
            fig = Visualizer.plot_bw_density_from_springs(
                final_structure, densities,
                initial_structure=initial_structure,
                title="Topology (B/W) - SIMP",
            )
        else:
            fig = Visualizer.plot_bw_density(
                final_structure,
                initial_structure=initial_structure,
                title="Topology (B/W)",
            )
        sections.append(_wrap_section(
            "Black and White Density",
            _embed_img(_fig_to_base64(fig), "B/W Density"),
        ))
    except Exception:
        logger.exception("Report: failed to render B/W density plot")

    # 9. SIMP density field
    if is_simp and densities is not None:
        try:
            fig = Visualizer.plot_density_field(
                final_structure, densities,
                title="SIMP Density Field",
            )
            sections.append(_wrap_section(
                "Density Field (SIMP)",
                _embed_img(_fig_to_base64(fig), "Density Field"),
            ))
        except Exception:
            logger.exception("Report: failed to render density field plot")

    # 10. Compliance history
    if result is not None and result.compliance_history:
        sections.append(_section_compliance_history(result))

    # 11. Algorithm comparison
    if comparison_results and len(comparison_results) >= 2:
        sections.append(_section_comparison(
            comparison_results, initial_structure,
        ))

    # 12. Footer
    sections.append(_section_footer())

    html = _HTML_TEMPLATE.replace("{{BODY}}", "\n".join(sections))
    logger.info("Report generated successfully (%d characters)", len(html))
    return html


# Private section builders


def _wrap_section(title: str, body: str) -> str:
    return f'<div class="section"><h2>{title}</h2>{body}</div>'


def _section_header(timestamp: str, version: str) -> str:
    ver = f" v{version}" if version else ""
    return (
        '<div class="header">'
        f"<h1>Topology Optimisation - Final Report</h1>"
        f'<p class="subtitle">Generated {timestamp} | '
        f"Topology Optimization{ver}</p>"
        "</div>"
    )


def _section_parameters(
    algorithm: str,
    parameters: dict[str, Any],
    initial_structure: Structure | None,
    final_structure: Structure,
    is_simp: bool,
    result: OptimizationResult | None,
) -> str:
    rows: list[str] = []

    def _row(label: str, value: Any) -> None:
        rows.append(f"<tr><td><strong>{label}</strong></td><td>{value}</td></tr>")

    _row("Algorithm", algorithm)
    _row("Target Mass Fraction", f"{parameters.get('target_mass_fraction', '-'):.0%}"
         if isinstance(parameters.get('target_mass_fraction'), (int, float))
         else parameters.get('target_mass_fraction', '-'))
    _row("Filter Radius", parameters.get("filter_radius", "-"))

    if algorithm == "SIMP":
        _row("Penalization (p)", parameters.get("penalization", "-"))
        _row("Move Limit", parameters.get("move_limit", "-"))
        _row("Convergence Tol.", parameters.get("convergence_tol", "-"))
        _row("Max Iterations", parameters.get("max_iterations", "-"))
    else:
        _row("Removal Rate / Iteration", parameters.get("removal_per_iteration", "-"))

    if initial_structure is not None:
        _row("Initial Nodes", initial_structure.num_nodes)
        _row("Initial Springs", initial_structure.graph.number_of_edges())
        _row("Initial Mass", f"{initial_structure.total_mass():.1f}")

    # Final metrics
    if is_simp and result and result.densities:
        dens = result.densities
        threshold = 0.1
        springs = final_structure.get_springs()
        active_spr = sum(1 for d in dens.values() if d >= threshold)
        eff_mass = sum(
            dens.get(sp.node_ids, dens.get((sp.node_ids[1], sp.node_ids[0]), 1.0)) * sp.length
            for sp in springs
        )
        total_mass = sum(sp.length for sp in springs)
        active_nodes: set[int] = set()
        for sp in springs:
            key = sp.node_ids
            rkey = (key[1], key[0])
            xe = dens.get(key, dens.get(rkey, 1.0))
            if xe >= threshold:
                active_nodes.add(key[0])
                active_nodes.add(key[1])
        _row("Final Active Nodes", len(active_nodes))
        _row("Final Active Springs", active_spr)
        _row("Effective Mass", f"{eff_mass:.1f}")
        _row("Final Mass Fraction", f"{eff_mass / total_mass:.1%}" if total_mass else "-")
    else:
        _row("Final Nodes", final_structure.num_nodes)
        _row("Final Springs", final_structure.graph.number_of_edges())
        _row("Final Mass", f"{final_structure.total_mass():.1f}")
        if initial_structure is not None:
            init_m = initial_structure.total_mass()
            frac = final_structure.total_mass() / init_m if init_m else 0
            _row("Final Mass Fraction", f"{frac:.1%}")

    if result is not None:
        _row("Iterations", result.iterations)
        if result.compliance_history:
            _row("Initial Compliance", f"{result.compliance_history[0]:.6g}")
            _row("Final Compliance", f"{result.compliance_history[-1]:.6g}")
            c0 = result.compliance_history[0]
            c1 = result.compliance_history[-1]
            if c0:
                _row("Compliance Change", f"{(c1 - c0) / c0 * 100:+.1f}%")

    table = (
        '<table class="params-table"><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )
    return _wrap_section("Problem Setup and Results Summary", table)


def _section_compliance_history(result: OptimizationResult) -> str:
    ch = result.compliance_history
    fig, ax = plt.subplots(figsize=(10, 4))
    iterations = list(range(1, len(ch) + 1))
    ax.plot(iterations, ch, color="#2563eb", lw=2, marker="o", markersize=3)
    ax.fill_between(iterations, ch, alpha=0.08, color="#2563eb")

    i_min = int(np.argmin(ch))
    i_max = int(np.argmax(ch))
    ax.annotate(
        f"Min: {ch[i_min]:.4g}",
        xy=(i_min + 1, ch[i_min]),
        xytext=(0, -18), textcoords="offset points",
        fontsize=8, color="#16a34a", fontweight="bold", ha="center",
        arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1),
    )
    if i_max != i_min:
        ax.annotate(
            f"Max: {ch[i_max]:.4g}",
            xy=(i_max + 1, ch[i_max]),
            xytext=(0, 16), textcoords="offset points",
            fontsize=8, color="#dc2626", fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1),
        )

    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("Compliance (total strain energy)", fontsize=10)
    ax.set_title("Compliance vs. Iteration", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(1, max(len(ch), 2))
    fig.tight_layout()

    img = _embed_img(_fig_to_base64(fig), "Compliance History")
    return _wrap_section("Compliance History", img)


def _section_comparison(
    comparison_results: dict[str, OptimizationResult],
    initial_structure: Structure | None,
) -> str:
    parts: list[str] = []

    # Structure comparison plot
    try:
        fig = Visualizer.plot_comparison_structures(comparison_results)
        parts.append(_embed_img(_fig_to_base64(fig), "Structure Comparison"))
    except Exception:
        logger.exception("Report: comparison structure plot failed")

    # Compliance convergence overlay
    try:
        fig = Visualizer.plot_compliance_comparison(comparison_results)
        parts.append(_embed_img(_fig_to_base64(fig), "Compliance Comparison"))
    except Exception:
        logger.exception("Report: compliance comparison chart failed")

    # Summary table
    init_mass = (
        initial_structure.total_mass() if initial_structure else 1.0
    )
    rows: list[str] = []
    header = "<tr><th>Metric</th>"
    for name in comparison_results:
        header += f"<th>{name}</th>"
    header += "</tr>"
    rows.append(header)

    # Gather metrics
    metrics: dict[str, dict[str, str]] = {}
    for algo, res in comparison_results.items():
        m: dict[str, str] = {}
        m["Iterations"] = str(res.iterations)
        c_final = res.compliance_history[-1] if res.compliance_history else 0
        m["Final Compliance"] = f"{c_final:.4g}"
        if res.compliance_history:
            c0 = res.compliance_history[0]
            change = ((c_final - c0) / c0 * 100) if c0 else 0
            m["Compliance Change"] = f"{change:+.1f}%"
        else:
            m["Compliance Change"] = "-"
        final_s = res.history[-1] if res.history else None
        if res.densities and final_s:
            dens = res.densities
            sps = final_s.get_springs()
            active = sum(1 for d in dens.values() if d >= 0.1)
            eff = sum(
                dens.get(sp.node_ids, dens.get((sp.node_ids[1], sp.node_ids[0]), 1.0)) * sp.length
                for sp in sps
            )
            m["Active Springs"] = str(active)
            m["Eff. Mass"] = f"{eff:.1f}"
            m["Mass Fraction"] = f"{eff / init_mass:.1%}" if init_mass else "-"
        elif final_s:
            m["Active Springs"] = str(final_s.graph.number_of_edges())
            m["Eff. Mass"] = f"{final_s.total_mass():.1f}"
            mf = final_s.total_mass() / init_mass if init_mass else 0
            m["Mass Fraction"] = f"{mf:.1%}"
        metrics[algo] = m

    all_keys = list(next(iter(metrics.values())).keys()) if metrics else []
    for key in all_keys:
        row = f"<tr><td><strong>{key}</strong></td>"
        for algo in comparison_results:
            row += f"<td>{metrics[algo].get(key, '-')}</td>"
        row += "</tr>"
        rows.append(row)

    table = '<table class="params-table">' + "".join(rows) + "</table>"
    parts.append(table)

    return _wrap_section("Algorithm Comparison", "\n".join(parts))


def _section_footer() -> str:
    return (
        '<div class="footer">'
        "<p>This report was automatically generated by the "
        "Topology Optimization application.  "
        "All plots use the same rendering pipeline as the interactive UI.</p>"
        "</div>"
    )


# HTML template

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Topology Optimisation - Final Report</title>
<style>
  :root {
    --primary: #2563eb;
    --bg: #ffffff;
    --text: #1e293b;
    --muted: #64748b;
    --border: #e2e8f0;
    --section-bg: #f8fafc;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    max-width: 1100px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
  }
  .header {
    text-align: center;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 3px solid var(--primary);
  }
  .header h1 {
    font-size: 2rem;
    color: var(--primary);
    margin-bottom: 0.3rem;
  }
  .header .subtitle {
    color: var(--muted);
    font-size: 0.95rem;
  }
  .section {
    background: var(--section-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.8rem;
    page-break-inside: avoid;
  }
  .section h2 {
    font-size: 1.3rem;
    color: var(--primary);
    margin-bottom: 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid var(--border);
  }
  .section img {
    display: block;
    margin: 0.8rem auto;
    border-radius: 6px;
    border: 1px solid var(--border);
  }
  .params-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92rem;
  }
  .params-table th,
  .params-table td {
    text-align: left;
    padding: 0.45rem 0.8rem;
    border-bottom: 1px solid var(--border);
  }
  .params-table th {
    background: var(--primary);
    color: #fff;
  }
  .params-table tr:nth-child(even) td {
    background: rgba(37, 99, 235, 0.04);
  }
  .footer {
    margin-top: 2rem;
    text-align: center;
    font-size: 0.85rem;
    color: var(--muted);
    border-top: 1px solid var(--border);
    padding-top: 1rem;
  }
  @media print {
    body { padding: 0; }
    .section { break-inside: avoid; }
  }
</style>
</head>
<body>
{{BODY}}
</body>
</html>
"""
