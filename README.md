# Topology Optimization - 2-D Mass-Spring Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests: 187](https://img.shields.io/badge/Tests-187-brightgreen.svg)](tests/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Deploy](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

> **Topology Optimization** of 2-D mechanical structures using a mass-spring
> finite-element method with **Iterative Node Removal (INR)** and
> **SIMP** (Solid Isotropic Material with Penalization) algorithms,
> implemented in Python with a Streamlit web UI.  
> **Version 2.0.0**

---

## Overview

This application performs **topology optimization** on a 2-D rectangular grid of
mass points connected by linear springs. Two optimization algorithms are
available:

* **Iterative Node Removal (INR)** – removes the nodes with the lowest
  strain energy until the target mass fraction is reached.
* **SIMP** – assigns continuous design densities to every spring and
  drives them towards 0/1 using the Optimality Criteria (OC) update.

Both algorithms share a common FEM solver, spatial sensitivity filter,
and visualization pipeline. They can be run independently or compared
side-by-side on the same structure.

### Features

* Parametric design space (width × height)
* Boundary conditions (pin / roller supports) and external forces via UI
* Quick-apply presets for standard MBB beam BCs
* FEM solver (Hooke's law, stiffness matrix, penalty BCs)
* Two optimization algorithms: INR (discrete) and SIMP (density-based)
* Side-by-side algorithm comparison with detailed metrics
* Spatial sensitivity filter for regularisation
* Live progress display with iteration feedback
* Visualization tabs: initial structure, current structure, deformation,
  strain-energy heatmap, internal forces (tension/compression),
  B/W density, SIMP density field
* Import structure from **black & white image** (PNG, JPG, BMP, …)
* **Draw** a structure directly on an in-browser canvas
* Animated GIF export of the optimization history
* Self-contained **HTML report** generation with embedded plots
* Save / load full state (JSON), including optimization results
* Image export (PNG) for every visualization tab
* MBB beam benchmark preset (full & half-symmetry)
* Dangling-node cleanup (dead-end branch removal)
* Structured logging (`logging` module) with rotating file handler
* Explicit error handling with user-friendly toast messages in the UI
* Input validation on all public APIs

---

## Project Structure

```
topology-optimization/
├── src/
│   ├── __init__.py              # Package metadata & version
│   ├── models/
│   │   ├── node.py              # Node (mass point)
│   │   ├── spring.py            # Linear spring
│   │   └── structure.py         # Graph-based structure (networkx)
│   ├── solver/
│   │   ├── fem_solver.py        # Stiffness matrix assembly & solver
│   │   ├── optimizer_base.py    # Abstract base class & OptimizationResult
│   │   ├── optimizer.py         # Iterative Node Removal (INR) optimizer
│   │   └── simp_optimizer.py    # SIMP density-based optimizer
│   ├── utils/
│   │   ├── visualization.py     # Matplotlib plots & animation
│   │   ├── io_handler.py        # JSON save / load
│   │   ├── image_import.py      # B/W image -> structure import
│   │   └── report_generator.py  # Self-contained HTML report builder
│   └── presets/
│       └── mbb_beam.py          # MBB beam benchmark
├── tests/
│   ├── conftest.py              # Shared pytest fixtures
│   ├── test_node.py             # Node model tests
│   ├── test_spring.py           # Spring model tests
│   ├── test_structure.py        # Structure model tests
│   ├── test_fem_solver.py       # FEM solver tests
│   ├── test_optimizer.py        # INR optimizer tests
│   ├── test_simp_optimizer.py   # SIMP optimizer tests
│   ├── test_mbb_beam.py         # MBB beam preset tests
│   ├── test_io_handler.py       # JSON I/O tests
│   ├── test_image_import.py     # Image import tests
│   ├── test_report_generator.py # Report generator tests
│   └── test_visualization.py    # Visualization tests
├── assets/
│   ├── gifs/                    # Example optimization animation GIFs
│   ├── img_examples/            # Example B/W images for structure import
│   ├── reports/                 # Example generated HTML reports
│   └── state_examples/          # Example saved-state JSON files
├── app.py                       # Streamlit web UI
├── requirements.txt
├── README.md
├── LICENSE
├── logs/                        # Auto-created rotating log files
└── docs/
    └── Aufgabenstellung.pdf
```

---

## Live Demo

The app is deployed on **Streamlit Community Cloud** — try it without
installing anything:

> **<https://2d-topology-optimization.streamlit.app>**

---

## Installation (local)

### Prerequisites

* **Python 3.10+**
* (recommended) a virtual environment

```bash
# Clone the repository
git clone <repo-url>
cd topology-optimization

# Create & activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
streamlit run app.py
```

The app will open in the browser at `http://localhost:8501`.

### Quick Start Guide

1. **Sidebar** — Select a preset (e.g. *MBB Beam*), set width and height,
   then click **🔨 Create Structure**.
2. **Boundary Conditions & Forces** (expandable) — Select a node, set
   supports, define forces (or use the **⚡ Apply Default Boundary
   Conditions** quick-setup).
3. **Algorithm** — Choose *Node Removal (INR)* or *SIMP* and tune
   algorithm-specific parameters.
4. **Run** — Click **▶️ Start** for a full run, **⏩ Single Step** for
   one iteration, or **🔬 Compare Both** for a head-to-head comparison.
5. **Visualization** — Switch between the tabs: *Initial Structure*,
   *Current Structure*, *Deformation*, *Strain Energy*, *Internal
   Forces*, *B/W Density*, *Density Field*, and *Algorithm Comparison*.
6. **Export** — Download any plot as PNG, the full state as JSON, an
   animated GIF of the optimization history, or a self-contained HTML
   report.

### MBB Beam Demo

1. Preset → **MBB Beam**, Width = 30, Height = 10.
2. Create structure.
3. Target mass fraction = 0.5, Removal rate = 3.
4. Start optimization → the characteristic arch/truss topology
   should emerge.

### Image Import

Upload a **black & white** image (PNG, JPG, BMP, …). Black pixels become
material nodes; white pixels become void. The image is resized to the
grid dimensions set in the sidebar.

### Canvas Drawing

Draw a custom shape directly in the browser using freehand, rectangle,
circle, or line tools. Click **🎨 Apply Drawing** to generate the
structure.

---

## Architecture

| Class                  | Description                                                             |
|------------------------|-------------------------------------------------------------------------|
| `Node`                 | Mass point with coordinates, boundary conditions, forces                |
| `Spring`               | Spring with stiffness matrix (globally transformed) and energy          |
| `Structure`            | NetworkX graph of nodes & springs, connectivity / mechanism checks      |
| `FEMSolver`            | Assembles $K_g$, applies BCs, solves $K \cdot u = F$                   |
| `OptimizerBase`        | Abstract base class for all optimizers                                  |
| `NodeRemovalOptimizer` | INR: solve → energy → remove → validate                                |
| `SIMPOptimizer`        | SIMP: solve → sensitivity → OC update → convergence check              |
| `OptimizationResult`   | Serialisable container for optimizer output (history, densities, …)     |
| `Visualizer`           | Matplotlib plots (structure, deformation, heatmap, forces, animation)   |
| `generate_report`      | Self-contained HTML report builder with embedded base64 images          |

### Logging

All modules use Python's built-in `logging` module.  The Streamlit app
configures the root logger at startup with a rotating file handler
(`logs/app.log`, 5 MB, 3 backups); backend modules log technical
details at `DEBUG` / `INFO` level while user-facing errors are surfaced
through `st.error()` / `st.warning()`.

### Error Handling

Public APIs validate inputs and raise descriptive exceptions (`ValueError`,
`KeyError`, `OSError`, …).  The Streamlit app wraps all I/O, solver, and
rendering calls in `try / except` blocks so that failures are reported as
friendly toast messages rather than raw tracebacks.

### Physical Model

* Coordinate system: origin top-left, x → right, z → down.
* Spring constants: horizontal/vertical $k = 1$ N/m, diagonal $k = 1/\sqrt{2}$ N/m.
* Node mass: $m = 1$ kg.
* Hooke's law: $K \cdot u = F$.
* Strain energy: $c_e = \frac{1}{2}\, u_e^T\, K_e\, u_e$.

---

## Testing

The project includes a comprehensive test suite with **187 unit tests**
covering all modules. Tests are written with
[pytest](https://docs.pytest.org/).

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run tests for a specific module
python -m pytest tests/test_node.py -v

# Run with short summary
python -m pytest tests/ --tb=short
```

| Test file                 | Module covered         | Tests |
|---------------------------|------------------------|------:|
| `test_node.py`            | `Node`                 |    19 |
| `test_spring.py`          | `Spring`               |    15 |
| `test_structure.py`       | `Structure`            |    38 |
| `test_fem_solver.py`      | `FEMSolver`            |    15 |
| `test_optimizer.py`       | `NodeRemovalOptimizer` |    22 |
| `test_simp_optimizer.py`  | `SIMPOptimizer`        |    12 |
| `test_mbb_beam.py`        | `create_mbb_beam`      |    14 |
| `test_io_handler.py`      | `io_handler`           |    13 |
| `test_image_import.py`    | `image_import`         |    10 |
| `test_report_generator.py`| `report_generator`     |     8 |
| `test_visualization.py`   | `Visualizer`           |    21 |

Shared fixtures (e.g. pre-built structures, nodes) live in `tests/conftest.py`.

---

## Changelog

### v2.0.0

* **SIMP optimizer** – density-based topology optimization with
  Optimality Criteria update, convergence detection, and per-spring
  density field visualization.
* **Algorithm comparison** – run INR and SIMP side-by-side on the same
  structure with detailed metric cards and overlaid compliance curves.
* **Image import** – create structures from black & white images.
* **Canvas drawing** – freehand / shape drawing directly in the browser.
* **HTML report generator** – self-contained downloadable report with all
  plots and parameters.
* **Internal force visualization** – tension/compression plot using axial
  spring forces.
* **Animated GIF export** – frame-by-frame GIF of the optimization
  history (INR snapshots or SIMP density evolution).
* **Optimizer base class** – shared abstract interface (`OptimizerBase`)
  and serialisable result container (`OptimizationResult`).
* **Dangling-node cleanup** – post-optimization removal of dead-end
  branches via articulation-point analysis.
* Expanded test suite: **187 tests**.

### v1.0.0

* Initial release with INR optimizer, FEM solver, MBB beam preset,
  Streamlit UI, JSON save/load, and PNG export.

---

## License

MIT License — see [LICENSE](LICENSE).