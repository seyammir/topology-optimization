"""Streamlit web application for 2-D topology optimisation.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.models.node import Node
from src.models.structure import Structure
from src.solver.fem_solver import FEMSolver
from src.solver.optimizer import OptimizationResult, TopologyOptimizer
from src.utils.io_handler import state_to_json_string, structure_from_json_string
from src.utils.visualization import Visualizer
from src.presets.mbb_beam import create_mbb_beam

# Page config
st.set_page_config(
    page_title="Topology Optimization",
    page_icon="🏗️",
    layout="wide",
)

# Session-state initialisation
if "structure" not in st.session_state:
    st.session_state.structure = None
if "initial_structure" not in st.session_state:
    st.session_state.initial_structure = None
if "result" not in st.session_state:
    st.session_state.result = None
if "displacement" not in st.session_state:
    st.session_state.displacement = None
if "node_energies" not in st.session_state:
    st.session_state.node_energies = None
if "iteration" not in st.session_state:
    st.session_state.iteration = 0
if "editor_gen" not in st.session_state:
    st.session_state.editor_gen = 0


def _reset_state() -> None:
    """Clear all optimisation-related state."""
    st.session_state.result = None
    st.session_state.displacement = None
    st.session_state.node_energies = None
    st.session_state.iteration = 0


# SIDEBAR
with st.sidebar:
    st.header("⚙️ Configuration")

    # Preset
    preset = st.selectbox(
        "Preset",
        ["Custom", "MBB Beam", "MBB Beam (half)"],
    )

    st.subheader("Design Space")
    col_w, col_h = st.columns(2)
    with col_w:
        width = st.number_input("Width (cells)", 2, 100, 30, key="width")
    with col_h:
        height = st.number_input("Height (cells)", 2, 60, 10, key="height")

    st.subheader("Optimization")
    target_frac = st.slider(
        "Target Mass Fraction", 0.1, 0.9, 0.5, 0.05,
        help="Fraction of the original mass to be retained.",
    )
    removal_rate = st.slider(
        "Removal Rate per Iteration", 1, 20, 3,
        help="Maximum number of nodes removed per step. Lower values = better topology, but slower.",
    )
    filter_radius = st.slider(
        "Filter Radius", 0.0, 6.0, 1.5, 0.5,
        help="Spatial sensitivity filter: smooths the energy distribution "
             "across neighboring nodes. Best range: 1.5-3.0 (≈ 1.5-3x cell size).",
    )

    # Create / preset buttons
    if st.button("🔨 Create Structure", use_container_width=True):
        _reset_state()
        st.session_state.editor_gen += 1  # invalidate old editor widget keys
        if preset == "MBB Beam":
            struct = create_mbb_beam(nx=width, nz=height, half=False)
        elif preset == "MBB Beam (half)":
            struct = create_mbb_beam(nx=width, nz=height, half=True)
        else:
            struct = Structure.create_rectangular(width, height)
        st.session_state.structure = struct
        st.session_state.initial_structure = struct.snapshot()

    # Save / Load
    st.divider()
    st.subheader("💾 Save / Load")

    # Download
    if st.session_state.structure is not None:
        json_str = state_to_json_string(st.session_state.structure)
        st.download_button(
            "⬇️ Download State (JSON)",
            data=json_str,
            file_name="topology_state.json",
            mime="application/json",
            use_container_width=True,
        )

    # Upload
    uploaded = st.file_uploader("Load State (JSON)", type=["json"])
    if uploaded is not None:
        try:
            content = uploaded.read().decode("utf-8")
            st.session_state.structure = structure_from_json_string(content)
            st.session_state.initial_structure = st.session_state.structure.snapshot()
            _reset_state()
            st.success("State loaded!")
        except Exception as exc:
            st.error(f"Error loading: {exc}")


# MAIN AREA
st.title("2-D Topology Optimization")
st.caption("Mass-Spring Model  ·  Iterative Node Removal  ·  FEM Solver")

struct: Structure | None = st.session_state.structure

if struct is None:
    st.info(
        "First create a structure via the sidebar "
        "(select a preset -> **Create Structure**)."
    )
    st.stop()

# Boundary-condition & force editor (expandable)
with st.expander("🔧 Edit Boundary Conditions & Forces", expanded=False):
    st.markdown(
        "Select a **node** by its grid position and "
        "set boundary conditions or forces."
    )

    # Determine grid bounds from node coordinates.
    all_nodes = struct.get_nodes()
    xs_set = sorted({n.x for n in all_nodes})
    zs_set = sorted({n.z for n in all_nodes})

    _g = st.session_state.editor_gen  # unique prefix per structure creation
    col_a, col_b = st.columns(2)
    with col_a:
        sel_x = st.selectbox("x Position", xs_set, key=f"sel_x_{_g}")
    with col_b:
        sel_z = st.selectbox("z Position", zs_set, key=f"sel_z_{_g}")

    # Find the node at that position.
    target_node: Node | None = None
    for n in all_nodes:
        if n.x == sel_x and n.z == sel_z:
            target_node = n
            break

    if target_node is not None:
        st.write(f"**Node {target_node.id}** at position ({target_node.x}, {target_node.z})")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            fx = target_node.fixed_x
            target_node.fixed_x = st.checkbox("Fixed x", value=fx, key=f"fix_x_{_g}_{target_node.id}")
        with col2:
            fz = target_node.fixed_z
            target_node.fixed_z = st.checkbox("Fixed z", value=fz, key=f"fix_z_{_g}_{target_node.id}")
        with col3:
            target_node.fx = st.number_input(
                "Force Fx [N]", value=float(target_node.fx),
                step=0.1, format="%.2f", key=f"fx_{_g}_{target_node.id}",
            )
        with col4:
            target_node.fz = st.number_input(
                "Force Fz [N]", value=float(target_node.fz),
                step=0.1, format="%.2f", key=f"fz_{_g}_{target_node.id}",
            )
    else:
        st.warning("No node found at this position.")

# Summary metrics
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Nodes", struct.num_nodes)
col_m2.metric("Springs", struct.graph.number_of_edges())
col_m3.metric("Mass [kg]", f"{struct.total_mass():.0f}")
init_mass = (
    st.session_state.initial_structure.total_mass()
    if st.session_state.initial_structure else struct.total_mass()
)
frac = struct.total_mass() / init_mass if init_mass else 0
col_m4.metric("Mass Fraction", f"{frac:.1%}")

# Optimisation controls
st.divider()
ctrl_cols = st.columns([1, 1, 1, 1])
with ctrl_cols[0]:
    run_full = st.button("▶️ Start Optimization", use_container_width=True)
with ctrl_cols[1]:
    run_step = st.button("⏩ Single Step", use_container_width=True)
with ctrl_cols[2]:
    do_reset = st.button("🔄 Reset", use_container_width=True)
with ctrl_cols[3]:
    do_cleanup = st.button("🧹 Remove Dangling", use_container_width=True,
                           help="Iteratively remove dead-end nodes that don't carry load.")

if do_reset and st.session_state.initial_structure is not None:
    st.session_state.structure = st.session_state.initial_structure.snapshot()
    _reset_state()
    st.rerun()

if do_cleanup:
    removed = struct.remove_dangling_nodes()
    if removed > 0:
        st.success(f"Removed {removed} dangling nodes.")
        # Re-solve for updated displacements / energies
        struct.renumber_dofs()
        solver = FEMSolver()
        u = solver.solve(struct)
        st.session_state.displacement = u
        st.session_state.node_energies = TopologyOptimizer._compute_node_energies(struct, u)
    else:
        st.info("No dangling nodes found.")
    st.rerun()

# Full optimisation run
if run_full:
    optimizer = TopologyOptimizer(
        target_mass_fraction=target_frac,
        removal_per_iteration=removal_rate,
        filter_radius=filter_radius,
    )
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    initial_mass = struct.total_mass()
    target_mass = target_frac * initial_mass

    def _cb(s: Structure, it: int, ne: dict) -> None:
        frac_done = 1.0 - (s.total_mass() - target_mass) / (initial_mass - target_mass)
        frac_done = max(0.0, min(1.0, frac_done))
        progress_bar.progress(frac_done)
        status_text.text(
            f"Iteration {it} — Nodes: {s.num_nodes}, "
            f"Mass: {s.total_mass():.0f}/{target_mass:.0f}"
        )

    result = optimizer.optimize(struct, callback=_cb)
    progress_bar.progress(1.0)
    status_text.text(
        f"✅ Done after {result.iterations} iterations — "
        f"{struct.num_nodes} nodes remaining."
    )
    st.session_state.result = result

    # Solve once more for final displacement.
    struct.renumber_dofs()
    solver = FEMSolver()
    u = solver.solve(struct)
    st.session_state.displacement = u

    # Final node energies.
    st.session_state.node_energies = TopologyOptimizer._compute_node_energies(struct, u)

    st.rerun()

# Single step
if run_step:
    optimizer = TopologyOptimizer(
        target_mass_fraction=target_frac,
        removal_per_iteration=removal_rate,
        filter_radius=filter_radius,
    )
    u, ne, removed = optimizer.step(struct)
    st.session_state.displacement = u
    st.session_state.node_energies = ne
    st.session_state.iteration += 1
    if removed == 0:
        st.warning("No node could be removed.")
    else:
        st.success(f"Step {st.session_state.iteration}: {removed} nodes removed.")
    st.rerun()

# Visualisation tabs
st.divider()
tab_init, tab_current, tab_deformed, tab_heatmap, tab_bw = st.tabs(
    [
        "📐 Initial Structure",
        "🏗️ Current Structure",
        "📏 Deformation",
        "🌡️ Strain Energy",
        "⬛ B/W Density",
    ]
)

with tab_init:
    if st.session_state.initial_structure is not None:
        fig_init = Visualizer.plot_structure(
            st.session_state.initial_structure, title="Initial Structure"
        )
        st.pyplot(fig_init)

        st.write("")
        png_init = Visualizer.fig_to_png_bytes(fig_init)
        st.download_button(
            "⬇️ Download Image (PNG)",
            data=png_init,
            file_name="initial_structure.png",
            mime="image/png",
            key="dl_init",
        )
        plt.close(fig_init)

with tab_current:
    fig_cur = Visualizer.plot_structure(struct, title="Current Structure")
    st.pyplot(fig_cur)

    st.write("")
    png_bytes = Visualizer.fig_to_png_bytes(fig_cur)
    st.download_button(
        "⬇️ Download Image (PNG)",
        data=png_bytes,
        file_name="topology_result.png",
        mime="image/png",
        key="dl_cur",
    )
    plt.close(fig_cur)

with tab_deformed:
    u = st.session_state.displacement
    if u is not None:
        auto_scale = st.checkbox("Auto Scaling", value=True, key="auto_scale")
        if auto_scale:
            fig_def = Visualizer.plot_structure(
                struct, u=u, scale=0, title="Deformed Structure (auto-scaled)"
            )
        else:
            scale = st.slider("Magnification Factor", 1.0, 500.0, 50.0, 1.0, key="def_scale")
            fig_def = Visualizer.plot_structure(
                struct, u=u, scale=scale, title="Deformed Structure"
            )
        st.pyplot(fig_def)

        st.write("")
        png_def = Visualizer.fig_to_png_bytes(fig_def)
        st.download_button(
            "⬇️ Download Image (PNG)",
            data=png_def,
            file_name="deformed_structure.png",
            mime="image/png",
            key="dl_def",
        )
        plt.close(fig_def)
    else:
        st.info("Run an optimization or a single step first.")

with tab_heatmap:
    ne = st.session_state.node_energies
    if ne is not None:
        fig_hm = Visualizer.plot_energy_heatmap(struct, ne, title="Strain Energy")
        st.pyplot(fig_hm)

        st.write("")
        png_hm = Visualizer.fig_to_png_bytes(fig_hm)
        st.download_button(
            "⬇️ Download Image (PNG)",
            data=png_hm,
            file_name="strain_energy_heatmap.png",
            mime="image/png",
            key="dl_hm",
        )
        plt.close(fig_hm)
    else:
        st.info("Run an optimization or a single step first.")

with tab_bw:
    fig_bw = Visualizer.plot_bw_density(
        struct,
        initial_structure=st.session_state.initial_structure,
        title="Topology (Black & White)",
    )
    st.pyplot(fig_bw)

    st.write("")
    png_bw = Visualizer.fig_to_png_bytes(fig_bw)
    st.download_button(
        "⬇️ Download Image (PNG)",
        data=png_bw,
        file_name="topology_bw.png",
        mime="image/png",
        key="dl_bw",
    )
    plt.close(fig_bw)

# Compliance history chart (if full optimisation was run)
result: OptimizationResult | None = st.session_state.result
if result is not None and result.compliance_history:
    st.divider()
    st.subheader("📈 Compliance History")

    ch = result.compliance_history

    # Key metrics
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    col_c1.metric("Iterations", result.iterations)
    col_c2.metric("Initial Compliance", f"{ch[0]:.4g}")
    col_c3.metric("Final Compliance", f"{ch[-1]:.4g}")
    change_pct = ((ch[-1] - ch[0]) / ch[0] * 100) if ch[0] != 0 else 0
    col_c4.metric("Change", f"{change_pct:+.1f}%")

    # Matplotlib chart with proper styling
    fig_ch, ax_ch = plt.subplots(figsize=(10, 4))
    iterations = list(range(1, len(ch) + 1))
    ax_ch.plot(iterations, ch, color="#2563eb", lw=2, marker="o", markersize=3)
    ax_ch.fill_between(iterations, ch, alpha=0.08, color="#2563eb")

    # Min / max markers
    i_min = int(np.argmin(ch))
    i_max = int(np.argmax(ch))
    ax_ch.annotate(
        f"Min: {ch[i_min]:.4g}",
        xy=(i_min + 1, ch[i_min]),
        xytext=(0, -18), textcoords="offset points",
        fontsize=8, color="#16a34a", fontweight="bold", ha="center",
        arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1),
    )
    if i_max != i_min:
        ax_ch.annotate(
            f"Max: {ch[i_max]:.4g}",
            xy=(i_max + 1, ch[i_max]),
            xytext=(0, 16), textcoords="offset points",
            fontsize=8, color="#dc2626", fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1),
        )

    ax_ch.set_xlabel("Iteration", fontsize=10)
    ax_ch.set_ylabel("Compliance (total strain energy)", fontsize=10)
    ax_ch.set_title("Compliance vs. Iteration", fontsize=13, fontweight="bold")
    ax_ch.grid(True, linestyle="--", alpha=0.4)
    ax_ch.set_xlim(1, max(len(ch), 2))
    fig_ch.tight_layout()

    st.pyplot(fig_ch)

    st.write("")
    png_ch = Visualizer.fig_to_png_bytes(fig_ch)
    st.download_button(
        "⬇️ Download Chart (PNG)",
        data=png_ch,
        file_name="compliance_history.png",
        mime="image/png",
        key="dl_compliance",
    )
    plt.close(fig_ch)
