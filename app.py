"""Streamlit web application for 2-D topology optimisation.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src import __version__
from src.models.node import Node
from src.models.structure import Structure
from src.solver.fem_solver import FEMSolver
from src.solver.optimizer import TopologyOptimizer
from src.solver.optimizer_base import OptimizationResult
from src.solver.simp_optimizer import SIMPOptimizer
from src.utils.io_handler import state_to_json_string, structure_from_json_string
from src.utils.image_import import structure_from_image
from src.utils.visualization import Visualizer
from src.presets.mbb_beam import create_mbb_beam
from streamlit_drawable_canvas import st_canvas


# Logging configuration
_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FMT = "%(asctime)s  %(levelname)-8s  %(name)s - %(message)s"
_LOG_DATE = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FMT,
    datefmt=_LOG_DATE,
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.handlers.RotatingFileHandler(
            _LOG_DIR / "app.log",
            maxBytes=5 * 1024 * 1024,  # 5 MB per file
            backupCount=3,
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Topology Optimization v" + __version__,
    page_icon="🏗️",
    layout="wide",
)

# Custom CSS for a cleaner look
st.markdown(
    """
    <style>
    /* Tighten metrics row spacing (theme-aware) */
    div[data-testid="stMetric"] {
        background: color-mix(in srgb, var(--default-textColor) 6%, transparent);
        border: 1px solid rgba(128, 128, 128, 0.45);
        border-radius: 8px;
        padding: 10px 14px;
    }
    /* Tab content padding */
    div[data-testid="stTab"] > div { padding-top: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
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
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "show_upload_success" not in st.session_state:
    st.session_state.show_upload_success = False
if "img_upload_key" not in st.session_state:
    st.session_state.img_upload_key = 0
if "show_img_upload_success" not in st.session_state:
    st.session_state.show_img_upload_success = False
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0
if "show_canvas_success" not in st.session_state:
    st.session_state.show_canvas_success = False
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = {}
if "algorithm" not in st.session_state:
    st.session_state.algorithm = "Node Removal (INR)"
if "optimization_running" not in st.session_state:
    st.session_state.optimization_running = False
if "show_stop_toast" not in st.session_state:
    st.session_state.show_stop_toast = False

# Show stop toast from previous rerun
if st.session_state.show_stop_toast:
    st.toast("Optimization stopped.", icon="🔴")
    st.session_state.show_stop_toast = False


def _reset_state() -> None:
    """Clear all optimisation-related state."""
    st.session_state.result = None
    st.session_state.displacement = None
    st.session_state.node_energies = None
    st.session_state.iteration = 0
    st.session_state.comparison_results = {}
    st.session_state.optimization_running = False


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
    algorithm = st.selectbox(
        "Algorithm",
        ["Node Removal (INR)", "SIMP"],
        key="algorithm",
        help="Select the topology optimisation algorithm to use.",
    )
    target_frac = st.slider(
        "Target Mass Fraction", 0.1, 0.9, 0.5, 0.05,
        help="Fraction of the original mass to be retained.",
    )

    # Algorithm-specific parameters
    if algorithm == "Node Removal (INR)":
        removal_rate = st.slider(
            "Removal Rate per Iteration", 1, 20, 3,
            help="Maximum number of nodes removed per step. Lower values = better topology, but slower.",
        )
    else:  # SIMP
        simp_penalization = st.slider(
            "Penalization Power (p)", 1.0, 5.0, 3.0, 0.5,
            help="SIMP penalization exponent. Higher values push densities towards 0/1.",
        )
        simp_move_limit = st.slider(
            "Move Limit", 0.05, 0.5, 0.2, 0.05,
            help="Maximum density change per element per iteration.",
        )
        simp_convergence_tol = st.slider(
            "Convergence Tolerance", 0.001, 0.1, 0.01, 0.001,
            format="%.3f",
            help="Stop when max density change falls below this value.",
        )
        simp_max_iterations = st.number_input(
            "Max Iterations", min_value=10, max_value=1000, value=200, step=10,
            help="Maximum number of SIMP iterations. The algorithm may stop earlier if convergence is reached.",
        )

    filter_radius = st.slider(
        "Filter Radius", 0.0, 6.0, 1.5, 0.5,
        help="Spatial sensitivity filter: smooths the energy distribution "
             "across neighboring nodes. Best range: 1.5-3.0 (≈ 1.5-3x cell size).",
    )

    # Create / preset buttons
    if st.button("🔨 Create Structure", width='stretch'):
        _reset_state()
        st.session_state.editor_gen += 1  # invalidate old editor widget keys
        try:
            if preset == "MBB Beam":
                struct = create_mbb_beam(nx=width, nz=height, half=False)
            elif preset == "MBB Beam (half)":
                struct = create_mbb_beam(nx=width, nz=height, half=True)
            else:
                struct = Structure.create_rectangular(width, height)
            st.session_state.structure = struct
            st.session_state.initial_structure = struct.snapshot()
            logger.info("Created structure: preset=%s, %dx%d", preset, width, height)
        except (ValueError, RuntimeError) as exc:
            logger.exception("Failed to create structure")
            st.error(f"Could not create structure: {exc}")

    # Save / Load
    st.divider()
    st.subheader("💾 Save / Load")

    # Download
    if st.session_state.structure is not None:
        try:
            json_str = state_to_json_string(st.session_state.structure)
            st.download_button(
                "⬇️ Download State (JSON)",
                data=json_str,
                file_name="topology_state.json",
                mime="application/json",
                width='stretch',
            )
        except Exception:
            logger.exception("Failed to serialise structure for download")
            st.error("Could not prepare the download. See logs for details.")

    # Show success toast from previous upload
    if st.session_state.show_upload_success:
        st.toast("State loaded successfully!", icon="✅")
        st.session_state.show_upload_success = False

    # Upload
    uploaded = st.file_uploader(
        "Load State (JSON)",
        type=["json"],
        key=f"uploader_{st.session_state.upload_key}",
    )
    if uploaded is not None:
        try:
            content = uploaded.read().decode("utf-8")
            st.session_state.structure = structure_from_json_string(content)
            st.session_state.initial_structure = st.session_state.structure.snapshot()
            _reset_state()
            logger.info("State loaded from uploaded file")
            # Set flag to show success message after rerun
            st.session_state.show_upload_success = True
            # Clear the uploader so the old file does not get
            # re-applied on every subsequent rerun.
            st.session_state.upload_key += 1
            st.rerun()
        except json.JSONDecodeError:
            logger.exception("Uploaded file is not valid JSON")
            st.error("The uploaded file is not valid JSON. Please check the file format.")
        except KeyError as exc:
            logger.exception("Uploaded JSON is missing required fields")
            st.error(f"Invalid state file - missing data: {exc}")
        except Exception as exc:
            logger.exception("Unexpected error loading uploaded state")
            st.error(f"Could not load state: {exc}")

    # Image import
    st.divider()
    st.subheader("🖼️ Import from Image")
    st.caption(
        "Upload a **black & white** image.  "
        "Black pixels -> material, white pixels -> void.  "
        "The image is resized to the grid dimensions above."
    )
    if st.session_state.show_img_upload_success:
        st.toast("Structure imported from image!", icon="✅")
        st.session_state.show_img_upload_success = False

    img_threshold = st.slider(
        "BW Threshold", 0, 255, 128,
        help="Grey-value cut-off (0-255). Pixels darker than this are material.",
    )
    img_file = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg", "bmp", "gif", "tiff"],
        key=f"img_uploader_{st.session_state.img_upload_key}",
    )
    if img_file is not None:
        try:
            struct_img = structure_from_image(
                img_file, width=width, height=height, threshold=img_threshold,
            )
            st.session_state.structure = struct_img
            st.session_state.initial_structure = struct_img.snapshot()
            _reset_state()
            st.session_state.editor_gen += 1
            logger.info(
                "Structure imported from image: %dx%d, %d nodes",
                width, height, struct_img.num_nodes,
            )
            st.session_state.show_img_upload_success = True
            st.session_state.img_upload_key += 1
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            logger.exception("Failed to import structure from image")
            st.error(f"Could not import image: {exc}")

    # Draw structure
    st.divider()
    st.subheader("✏️ Draw Structure")
    st.caption(
        "Draw your shape below (black = material).  "
        "Click **Apply Drawing** when done."
    )
    if st.session_state.show_canvas_success:
        st.toast("Structure created from drawing!", icon="✅")
        st.session_state.show_canvas_success = False

    drawing_mode = st.selectbox(
        "Drawing Tool",
        ["freedraw", "rect", "circle", "line", "transform"],
        format_func=lambda m: {
            "freedraw": "✏️ Freehand",
            "rect": "⬜ Rectangle",
            "circle": "⭕ Circle",
            "line": "📏 Line",
            "transform": "🔄 Move / Resize",
        }[m],
    )

    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=st.slider("Brush Size", 1, 30, 10, key="brush_size"),
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=300,
        height=200,
        drawing_mode=drawing_mode,
        key=f"canvas_{st.session_state.canvas_key}",
    )

    if st.button("🎨 Apply Drawing", width='stretch'):
        if canvas_result.image_data is not None:
            try:
                from PIL import Image as PILImage
                import io
                # canvas returns RGBA numpy array; convert to greyscale PNG
                rgba = canvas_result.image_data.astype(np.uint8)
                img = PILImage.fromarray(rgba, "RGBA").convert("L")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                struct_drawn = structure_from_image(
                    buf, width=width, height=height, threshold=img_threshold,
                )
                st.session_state.structure = struct_drawn
                st.session_state.initial_structure = struct_drawn.snapshot()
                _reset_state()
                st.session_state.editor_gen += 1
                st.session_state.canvas_key += 1
                logger.info(
                    "Structure created from canvas drawing: %dx%d, %d nodes",
                    width, height, struct_drawn.num_nodes,
                )
                st.session_state.show_canvas_success = True
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                logger.exception("Failed to create structure from drawing")
                st.error(f"Could not create structure from drawing: {exc}")
        else:
            st.warning("Please draw something on the canvas first.")

    # About
    st.divider()
    st.caption(f"**Topology Optimization** v{__version__}  \nMIT License")


# MAIN AREA
st.title("2-D Topology Optimization")
st.caption(
    "Mass-Spring Model | INR / SIMP Algorithms | FEM Solver | "
    f"v{__version__}"
)

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
st.divider()
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

_res = st.session_state.result
_is_simp_result = (
    st.session_state.algorithm == "SIMP"
    and _res is not None
    and _res.densities is not None
)

if _is_simp_result:
    # For SIMP: show effective (density-weighted) metrics
    _dens = _res.densities
    _threshold = 0.1  # springs below this density are considered void
    _active_springs = sum(1 for d in _dens.values() if d >= _threshold)
    # Effective mass = sum of density * spring_length (= node mass proxy)
    _springs_list = struct.get_springs()
    _eff_mass = sum(
        _dens.get(sp.node_ids, _dens.get((sp.node_ids[1], sp.node_ids[0]), 1.0)) * sp.length
        for sp in _springs_list
    )
    _total_mass = sum(sp.length for sp in _springs_list)
    # Active nodes: nodes connected to at least one active spring
    _active_nodes = set()
    for sp in _springs_list:
        key = sp.node_ids
        rkey = (key[1], key[0])
        xe = _dens.get(key, _dens.get(rkey, 1.0))
        if xe >= _threshold:
            _active_nodes.add(key[0])
            _active_nodes.add(key[1])
    col_m1.metric("Active Nodes", len(_active_nodes), help="Nodes with density >= 0.1")
    col_m2.metric("Active Springs", _active_springs, help="Springs with density >= 0.1")
    col_m3.metric("Eff. Mass", f"{_eff_mass:.0f}", help="Density-weighted total spring length")
    _frac_simp = _eff_mass / _total_mass if _total_mass > 0 else 0
    col_m4.metric("Mass Fraction", f"{_frac_simp:.1%}")
else:
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
ctrl_cols = st.columns([1, 1, 1, 1, 1, 1])
with ctrl_cols[0]:
    run_full = st.button(
        "▶️ Start", width='stretch', 
        help="Run the optimization until completion.")
with ctrl_cols[1]:
    run_step = st.button("⏩ Single Step", width='stretch')
with ctrl_cols[2]:
    run_compare = st.button(
        "🔬 Compare Both", width='stretch',
        help="Run both INR and SIMP on the same structure and compare results side-by-side.",
    )
with ctrl_cols[3]:
    force_stop = st.button(
        "⏹️ Stop", width='stretch',
        help="Force stop the running optimization.",
    )
with ctrl_cols[4]:
    do_reset = st.button("🔄 Reset", width='stretch')
with ctrl_cols[5]:
    do_cleanup = st.button("🧹 Cleanup", width='stretch',
                           help="Iteratively remove dead-end (dangling) nodes that don't carry load.")

# Handle force-stop: discard progress, reset to initial structure
if force_stop:
    st.session_state.optimization_running = False
    if st.session_state.initial_structure is not None:
        st.session_state.structure = st.session_state.initial_structure.snapshot()
    _reset_state()
    logger.info("Optimization force-stopped by user.")
    st.session_state.show_stop_toast = True
    st.rerun()

if do_reset and st.session_state.initial_structure is not None:
    st.session_state.structure = st.session_state.initial_structure.snapshot()
    _reset_state()
    st.rerun()

if do_cleanup:
    try:
        removed = struct.remove_dangling_nodes()
        if removed > 0:
            st.success(f"Removed {removed} dangling nodes.")
            logger.info("Removed %d dangling nodes", removed)
            # Re-solve for updated displacements / energies
            struct.renumber_dofs()
            solver = FEMSolver()
            u = solver.solve(struct)
            st.session_state.displacement = u
            st.session_state.node_energies = TopologyOptimizer._compute_node_energies(struct, u)
        else:
            st.info("No dangling nodes found.")
    except Exception:
        logger.exception("Error during dangling-node removal")
        st.error("An error occurred while removing dangling nodes. Check logs for details.")
    st.rerun()

# Helper: create the right optimizer based on sidebar selection
def _create_optimizer():
    """Instantiate the optimizer selected in the sidebar."""
    algo = st.session_state.algorithm
    if algo == "SIMP":
        return SIMPOptimizer(
            target_mass_fraction=target_frac,
            filter_radius=filter_radius,
            penalization=simp_penalization,
            move_limit=simp_move_limit,
            convergence_tol=simp_convergence_tol,
            max_iterations=simp_max_iterations,
        )
    else:  # Node Removal (INR)
        return TopologyOptimizer(
            target_mass_fraction=target_frac,
            removal_per_iteration=removal_rate,
            filter_radius=filter_radius,
        )

# Full optimisation run
if run_full:
    try:
        # Always reset to initial structure so that repeated /
        # comparative runs are independent of each other.
        algo = st.session_state.algorithm
        if st.session_state.initial_structure is not None:
            struct = st.session_state.initial_structure.snapshot()
            st.session_state.structure = struct

        st.session_state.optimization_running = True
        optimizer = _create_optimizer()
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        initial_mass = struct.total_mass()
        target_mass = target_frac * initial_mass

        def _cb(s: Structure, it: int, ne: dict[int, float]) -> None:
            if algo == "SIMP":
                max_it = optimizer.max_iterations
                frac_done = min(1.0, it / max_it)
                progress_bar.progress(frac_done)
                # Compute current volume fraction from densities
                dens = getattr(s, "_simp_densities", None)
                vols = getattr(s, "_simp_spring_volumes", None)
                if dens and vols:
                    cur_vol = sum(dens[k] * vols[k] for k in dens)
                    tot_vol = sum(vols.values())
                    active = sum(1 for d in dens.values() if d >= 0.1)
                    status_text.text(
                        f"[SIMP] Iteration {it}/{max_it} - "
                        f"Active Springs: {active}, "
                        f"Mass: {cur_vol:.0f}/{target_mass:.0f}"
                    )
                else:
                    status_text.text(f"[SIMP] Iteration {it}/{max_it}")
            else:
                frac_done = 1.0 - (s.total_mass() - target_mass) / (initial_mass - target_mass)
                frac_done = max(0.0, min(1.0, frac_done))
                progress_bar.progress(frac_done)
                status_text.text(
                    f"[{algo}] Iteration {it} - "
                    f"Nodes: {s.num_nodes}, "
                    f"Mass: {s.total_mass():.0f}/{target_mass:.0f}"
                )

        logger.info(
            "Starting optimization: algo=%s, target=%.0f%%, filter=%.1f",
            algo, target_frac * 100, filter_radius,
        )
        result = optimizer.optimize(struct, callback=_cb)
        progress_bar.progress(1.0)
        if algo == "SIMP" and result.densities:
            _active_final = sum(1 for d in result.densities.values() if d >= 0.1)
            _c_final = result.compliance_history[-1] if result.compliance_history else 0
            status_text.text(
                f"Done! [SIMP] after {result.iterations} iterations - "
                f"Active Springs: {_active_final}/{len(result.densities)}, "
                f"Compliance: {_c_final:.4g}"
            )
        else:
            status_text.text(
                f"Done! [{algo}] after {result.iterations} iterations - "
                f"{struct.num_nodes} nodes remaining."
            )
        st.session_state.result = result

        # Store in comparison dict
        st.session_state.comparison_results[algo] = result

        # Solve once more for final displacement.
        struct.renumber_dofs()
        solver = FEMSolver()
        if algo == "SIMP" and result.densities:
            u = solver.solve_with_densities(struct, result.densities)
        else:
            u = solver.solve(struct)
        st.session_state.displacement = u

        # Final node energies.
        if algo == "SIMP" and result.densities:
            st.session_state.node_energies = SIMPOptimizer._compute_node_energies_from_densities(
                struct, u, result.densities,
            )
        else:
            st.session_state.node_energies = TopologyOptimizer._compute_node_energies(struct, u)
        st.session_state.optimization_running = False
        logger.info(
            "Optimization complete: algo=%s, %d iterations, %d nodes remaining",
            algo, result.iterations, struct.num_nodes,
        )
    except ValueError as exc:
        st.session_state.optimization_running = False
        logger.exception("Invalid optimization parameters")
        st.error(f"Invalid optimization parameters: {exc}")
    except Exception:
        st.session_state.optimization_running = False
        logger.exception("Optimization failed")
        st.error(
            "An unexpected error occurred during optimization. "
            "Please check the structure setup and try again."
        )
    st.rerun()

# Single step
if run_step:
    try:
        optimizer = _create_optimizer()
        u, ne, removed = optimizer.step(struct)
        st.session_state.displacement = u
        st.session_state.node_energies = ne
        st.session_state.iteration += 1
        algo = st.session_state.algorithm
        if removed == 0 and algo != "SIMP":
            st.warning("No element could be removed - the structure may already be at its minimum.")
        elif algo == "SIMP":
            dens = getattr(struct, "_simp_densities", {})
            active = sum(1 for d in dens.values() if d >= 0.1)
            st.success(
                f"[SIMP] Step {st.session_state.iteration}: "
                f"Active Springs: {active}/{len(dens)}"
            )
            logger.info("[SIMP] Single step %d: active springs %d/%d",
                        st.session_state.iteration, active, len(dens))
        else:
            st.success(f"[{algo}] Step {st.session_state.iteration}: {removed} elements removed.")
            logger.info("[%s] Single step %d: removed %d elements", algo, st.session_state.iteration, removed)
    except ValueError as exc:
        logger.exception("Invalid parameters for single step")
        st.error(f"Invalid parameters: {exc}")
    except Exception:
        logger.exception("Single-step optimization failed")
        st.error("An error occurred during the optimization step.")
    st.rerun()

# Compare Both algorithms
if run_compare:
    if st.session_state.initial_structure is None:
        st.error("Create a structure first before comparing algorithms.")
    else:
        st.session_state.optimization_running = True
        # Safe defaults for algorithm-specific parameters
        _inr_removal = st.session_state.get("removal_rate", 3)
        # Try to read sidebar slider values; fall back to defaults
        _simp_pen = st.session_state.get("simp_penalization", 3.0)
        _simp_ml = st.session_state.get("simp_move_limit", 0.2)
        _simp_ct = st.session_state.get("simp_convergence_tol", 0.01)
        _simp_mi = st.session_state.get("simp_max_iterations", 200)

        st.session_state.comparison_results = {}
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        algo_specs = [
            (
                "Node Removal (INR)",
                lambda: TopologyOptimizer(
                    target_mass_fraction=target_frac,
                    removal_per_iteration=_inr_removal,
                    filter_radius=filter_radius,
                ),
            ),
            (
                "SIMP",
                lambda: SIMPOptimizer(
                    target_mass_fraction=target_frac,
                    filter_radius=filter_radius,
                    penalization=_simp_pen,
                    move_limit=_simp_ml,
                    convergence_tol=_simp_ct,
                    max_iterations=_simp_mi,
                ),
            ),
        ]

        try:
            for step_idx, (algo_name, make_opt) in enumerate(algo_specs):
                status_text.text(f"Running {algo_name}...")
                progress_bar.progress(step_idx / 2)

                # Fresh copy of the initial structure each time
                run_struct = st.session_state.initial_structure.snapshot()
                optimizer = make_opt()

                initial_mass = run_struct.total_mass()
                target_mass_val = target_frac * initial_mass

                def _cmp_cb(s, it, ne, _algo=algo_name, _opt=optimizer, _si=step_idx):
                    if _algo == "SIMP":
                        max_it = _opt.max_iterations
                        frac_done = min(1.0, it / max_it) if max_it else 0.0
                        progress_bar.progress((_si + frac_done) / 2)
                        status_text.text(f"[{_algo}] Iteration {it}/{max_it}")
                    else:
                        im = initial_mass
                        tm = target_mass_val
                        frac_done = 1.0 - (s.total_mass() - tm) / (im - tm) if im != tm else 1.0
                        frac_done = max(0.0, min(1.0, frac_done))
                        progress_bar.progress((_si + frac_done) / 2)
                        status_text.text(
                            f"[{_algo}] Iteration {it} - "
                            f"Nodes: {s.num_nodes}, "
                            f"Mass: {s.total_mass():.0f}/{tm:.0f}"
                        )

                result = optimizer.optimize(run_struct, callback=_cmp_cb)
                st.session_state.comparison_results[algo_name] = result
                logger.info(
                    "Comparison run complete: algo=%s, %d iterations",
                    algo_name, result.iterations,
                )

            progress_bar.progress(1.0)
            status_text.text(
                "Comparison complete! Switch to the "
                "🔬 Algorithm Comparison tab to see results."
            )
            st.session_state.optimization_running = False
            logger.info("Both algorithms compared successfully")
        except Exception:
            st.session_state.optimization_running = False
            logger.exception("Comparison run failed")
            st.error(
                "An error occurred during the comparison run. "
                "Please check the structure setup and try again."
            )
        st.rerun()

# Visualisation tabs
st.divider()
tab_init, tab_current, tab_deformed, tab_heatmap, tab_loadpath, tab_bw, tab_density, tab_compare = st.tabs(
    [
        "📐 Initial Structure",
        "🏗️ Current Structure",
        "📏 Deformation",
        "🌡️ Strain Energy",
        "⚡ Internal Forces",
        "⬛ B/W Density",
        "🎨 Density Field",
        "🔬 Algorithm Comparison",
    ]
)

with tab_init:
    if st.session_state.initial_structure is not None:
        try:
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
        except Exception:
            logger.exception("Failed to render initial structure")
            st.error("Could not render the initial structure plot.")

with tab_current:
    try:
        if _is_simp_result:
            fig_cur = Visualizer.plot_structure(
                struct, title="Current Structure",
                densities=st.session_state.result.densities,
            )
        else:
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
    except Exception:
        logger.exception("Failed to render current structure")
        st.error("Could not render the current structure plot.")

with tab_deformed:
    _is_simp = st.session_state.algorithm == "SIMP" and (
        st.session_state.result is not None
        and st.session_state.result.densities is not None
    )
    u = st.session_state.displacement
    if u is not None:
        try:
            auto_scale = st.checkbox("Auto Scaling", value=True, key="auto_scale")
            if _is_simp:
                # For SIMP: show density-weighted deformed structure
                _dens_def = st.session_state.result.densities
                _pen_def = st.session_state.result.penalization
                if auto_scale:
                    fig_def = Visualizer.plot_structure(
                        struct, u=u, scale=0,
                        title="Deformed Structure (auto-scaled)",
                        densities=_dens_def,
                    )
                else:
                    scale = st.slider("Magnification Factor", 1.0, 500.0, 50.0, 1.0, key="def_scale")
                    fig_def = Visualizer.plot_structure(
                        struct, u=u, scale=scale,
                        title="Deformed Structure",
                        densities=_dens_def,
                    )
            else:
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
        except Exception:
            logger.exception("Failed to render deformed structure")
            st.error("Could not render the deformation plot.")
    else:
        st.info("Run an optimization or a single step first.")

with tab_heatmap:
    ne = st.session_state.node_energies
    if ne is not None:
        try:
            _hm_dens = None
            if _is_simp_result and st.session_state.result is not None:
                _hm_dens = st.session_state.result.densities
            fig_hm = Visualizer.plot_energy_heatmap(
                struct, ne, title="Strain Energy", densities=_hm_dens,
            )
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
        except Exception:
            logger.exception("Failed to render energy heatmap")
            st.error("Could not render the strain energy heatmap.")
    else:
        st.info("Run an optimization or a single step first.")

with tab_loadpath:
    u_lp = st.session_state.displacement
    if u_lp is not None:
        try:
            struct.renumber_dofs()
            # Pass SIMP densities for density-scaled internal forces
            _simp_dens = None
            _simp_pen = 3.0
            if _is_simp and st.session_state.result is not None:
                _simp_dens = st.session_state.result.densities
                _simp_pen = st.session_state.result.penalization
            spring_forces = FEMSolver.compute_internal_forces(
                struct, u_lp, densities=_simp_dens, penalization=_simp_pen,
            )
            fig_lp = Visualizer.plot_internal_forces(
                struct,
                spring_forces,
                title="Tension & Compression",
                densities=_simp_dens,
            )
            st.pyplot(fig_lp)

            st.write("")
            png_lp = Visualizer.fig_to_png_bytes(fig_lp)
            st.download_button(
                "⬇️ Download Image (PNG)",
                data=png_lp,
                file_name="load_path.png",
                mime="image/png",
                key="dl_lp",
            )
            plt.close(fig_lp)
        except Exception:
            logger.exception("Failed to render load path plot")
            st.error("Could not render the load path plot.")
    else:
        st.info("Run an optimization or a single step first.")

with tab_bw:
    if _is_simp and st.session_state.result is not None and st.session_state.result.densities:
        # For SIMP: render B/W density from the spring density values
        try:
            fig_bw = Visualizer.plot_bw_density_from_springs(
                struct,
                st.session_state.result.densities,
                initial_structure=st.session_state.initial_structure,
                title="Topology (Black & White) - SIMP",
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
        except Exception:
            logger.exception("Failed to render B/W density plot for SIMP")
            st.error("Could not render the density plot.")
    else:
        try:
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
        except Exception:
            logger.exception("Failed to render B/W density plot")
            st.error("Could not render the density plot.")

with tab_density:
    res_density = st.session_state.result
    if res_density is not None and res_density.densities is not None:
        try:
            fig_dens = Visualizer.plot_density_field(
                struct, res_density.densities,
                title="SIMP Density Field",
            )
            st.pyplot(fig_dens)

            st.write("")
            png_dens = Visualizer.fig_to_png_bytes(fig_dens)
            st.download_button(
                "⬇️ Download Image (PNG)",
                data=png_dens,
                file_name="density_field.png",
                mime="image/png",
                key="dl_dens",
            )
            plt.close(fig_dens)
        except Exception:
            logger.exception("Failed to render density field plot")
            st.error("Could not render the density field plot.")
    else:
        st.info(
            "Run a **SIMP** optimization to see the continuous density field here."
        )

with tab_compare:
    comparison_results = st.session_state.comparison_results
    _n_comp = len(comparison_results)

    if _n_comp == 0:
        st.info(
            "No comparison data yet.  \n\n"
            "**How to compare algorithms:**\n"
            "- Click the **🔬 Compare Both** button above to "
            "automatically run both INR and SIMP on the same structure.\n"
            "- Alternatively, run optimizations individually with each "
            "algorithm — results accumulate here automatically."
        )
    elif _n_comp == 1:
        _done_algo = list(comparison_results.keys())[0]
        _missing = "SIMP" if "INR" in _done_algo else "Node Removal (INR)"
        st.warning(
            f"**{_done_algo}** result collected. "
            f"Run **{_missing}** to see the comparison, "
            f"or click **🔬 Compare Both** to run both automatically."
        )
        # Show single result preview
        _single_res = comparison_results[_done_algo]
        _c_single = _single_res.compliance_history[-1] if _single_res.compliance_history else 0.0
        st.metric(f"{_done_algo} — Final Compliance", f"{_c_single:.4g}")
    else:
        # Full comparison view
        st.subheader("Performance Summary")

        # Compute detailed metrics for each algorithm
        _compliances = {}
        _iterations = {}
        _masses = {}
        _mass_fractions = {}
        _nodes = {}
        _springs = {}
        _compliance_changes = {}
        _initial_mass = (
            st.session_state.initial_structure.total_mass()
            if st.session_state.initial_structure else 1.0
        )
        _initial_nodes = (
            st.session_state.initial_structure.num_nodes
            if st.session_state.initial_structure else 0
        )
        _initial_springs = (
            st.session_state.initial_structure.graph.number_of_edges()
            if st.session_state.initial_structure else 0
        )

        for algo_name, algo_res in comparison_results.items():
            c_final = algo_res.compliance_history[-1] if algo_res.compliance_history else 0.0
            c_init = algo_res.compliance_history[0] if algo_res.compliance_history else 0.0
            _compliances[algo_name] = c_final
            _iterations[algo_name] = algo_res.iterations
            _compliance_changes[algo_name] = (
                ((c_final - c_init) / c_init * 100) if c_init != 0 else 0.0
            )

            final_struct = algo_res.history[-1] if algo_res.history else None
            if algo_res.densities is not None and final_struct is not None:
                # SIMP: density-weighted metrics
                _dens = algo_res.densities
                _threshold = 0.1
                _sp_list = final_struct.get_springs()
                eff_mass = sum(
                    _dens.get(sp.node_ids, _dens.get((sp.node_ids[1], sp.node_ids[0]), 1.0)) * sp.length
                    for sp in _sp_list
                )
                tot_mass = sum(sp.length for sp in _sp_list)
                active_spr = sum(1 for d in _dens.values() if d >= _threshold)
                active_nd = set()
                for sp in _sp_list:
                    key = sp.node_ids
                    rkey = (key[1], key[0])
                    xe = _dens.get(key, _dens.get(rkey, 1.0))
                    if xe >= _threshold:
                        active_nd.add(key[0])
                        active_nd.add(key[1])
                _masses[algo_name] = eff_mass
                _mass_fractions[algo_name] = eff_mass / _initial_mass if _initial_mass > 0 else 0
                _nodes[algo_name] = len(active_nd)
                _springs[algo_name] = active_spr
            elif final_struct is not None:
                # Node Removal (INR)
                _masses[algo_name] = final_struct.total_mass()
                _mass_fractions[algo_name] = (
                    final_struct.total_mass() / _initial_mass if _initial_mass > 0 else 0
                )
                _nodes[algo_name] = final_struct.num_nodes
                _springs[algo_name] = final_struct.graph.number_of_edges()
            else:
                _masses[algo_name] = 0
                _mass_fractions[algo_name] = 0
                _nodes[algo_name] = 0
                _springs[algo_name] = 0

        # Metrics cards
        comp_cols = st.columns(len(comparison_results))
        for col, algo_name in zip(comp_cols, comparison_results):
            with col:
                st.markdown(f"**{algo_name}**")
                m1, m2 = st.columns(2)
                m1.metric("Iterations", _iterations[algo_name])
                m2.metric("Final Compliance", f"{_compliances[algo_name]:.4g}")
                m3, m4 = st.columns(2)
                m3.metric("Nodes", _nodes[algo_name],
                          delta=f"{_nodes[algo_name] - _initial_nodes}" if _initial_nodes else None,
                          delta_color="off")
                m4.metric("Springs", _springs[algo_name],
                          delta=f"{_springs[algo_name] - _initial_springs}" if _initial_springs else None,
                          delta_color="off")
                m5, m6 = st.columns(2)
                m5.metric("Mass", f"{_masses[algo_name]:.0f}")
                m6.metric("Mass Fraction", f"{_mass_fractions[algo_name]:.1%}")
                st.metric("Compliance Change", f"{_compliance_changes[algo_name]:+.1f}%")

        # Determine winner
        if len(_compliances) == 2:
            names = list(_compliances.keys())
            c0, c1 = _compliances[names[0]], _compliances[names[1]]
            if c0 > 0 and c1 > 0:
                if c0 < c1:
                    diff_pct = (c1 - c0) / c1 * 100
                    st.success(
                        f"**{names[0]}** achieved {diff_pct:.1f}% lower compliance "
                        f"(stiffer result) than **{names[1]}**."
                    )
                elif c1 < c0:
                    diff_pct = (c0 - c1) / c0 * 100
                    st.success(
                        f"**{names[1]}** achieved {diff_pct:.1f}% lower compliance "
                        f"(stiffer result) than **{names[0]}**."
                    )
                else:
                    st.info("Both algorithms achieved the same compliance.")

        # Side-by-side structure plots
        st.subheader("Structure Comparison")
        try:
            fig_struct_comp = Visualizer.plot_comparison_structures(
                comparison_results,
            )
            st.pyplot(fig_struct_comp)

            png_struct_comp = Visualizer.fig_to_png_bytes(fig_struct_comp)
            st.download_button(
                "⬇️ Download Structure Comparison (PNG)",
                data=png_struct_comp,
                file_name="structure_comparison.png",
                mime="image/png",
                key="dl_struct_comp",
            )
            plt.close(fig_struct_comp)
        except Exception:
            logger.exception("Failed to render structure comparison plot")
            st.error("Could not render the structure comparison plot.")

        # Compliance convergence overlay
        st.subheader("Compliance Convergence")
        try:
            fig_cc = Visualizer.plot_compliance_comparison(comparison_results)
            st.pyplot(fig_cc)

            png_cc = Visualizer.fig_to_png_bytes(fig_cc)
            st.download_button(
                "⬇️ Download Compliance Comparison (PNG)",
                data=png_cc,
                file_name="compliance_comparison.png",
                mime="image/png",
                key="dl_cc",
            )
            plt.close(fig_cc)
        except Exception:
            logger.exception("Failed to render compliance comparison chart")
            st.error("Could not render the compliance comparison chart.")

# Animation export (if full optimisation was run)
result: OptimizationResult | None = st.session_state.result
if result is not None and len(result.history) > 1:
    st.divider()
    st.subheader("🎬 Optimization Animation")

    anim_cols = st.columns([1, 1, 2])
    with anim_cols[0]:
        anim_mode = st.selectbox(
            "Style",
            ["B/W Density", "Structure"],
            key="anim_mode",
        )
    with anim_cols[1]:
        anim_speed = st.slider(
            "Frame Duration (ms)", 100, 1000, 300, 50,
            key="anim_speed",
        )

    mode_key = "bw" if anim_mode == "B/W Density" else "structure"

    if st.button("🎞️ Generate Animation (GIF)", width='content', key="gen_anim"):
        try:
            with st.spinner("Rendering animation frames..."):
                gif_bytes = Visualizer.create_animation_gif(
                    result.history,
                    initial_structure=st.session_state.initial_structure,
                    mode=mode_key,
                    duration_ms=anim_speed,
                )
            st.session_state["animation_gif"] = gif_bytes
            st.session_state["animation_mode"] = mode_key
        except Exception:
            logger.exception("Failed to create animation GIF")
            st.error("Could not create the animation. Check logs for details.")

    if "animation_gif" in st.session_state and st.session_state["animation_gif"] is not None:
        st.image(st.session_state["animation_gif"], caption="Optimization Animation", width='stretch')
        st.download_button(
            "⬇️ Download Animation (GIF)",
            data=st.session_state["animation_gif"],
            file_name="optimization_animation.gif",
            mime="image/gif",
            key="dl_anim",
        )

# Compliance history chart (if full optimisation was run)
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

    try:
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
    except Exception:
        logger.exception("Failed to render compliance history chart")
        st.error("Could not render the compliance history chart.")