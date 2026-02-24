# Topology Optimization - 2-D Mass-Spring Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests: 132+](https://img.shields.io/badge/Tests-132%2B-brightgreen.svg)](tests/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B.svg)](https://streamlit.io/)

> **Topology Optimization** of 2-D mechanical structures using a mass-spring
> finite-element method with iterative node removal, implemented in Python
> with a Streamlit web UI.  
> **Version 1.0.0**

---

## Overview

This application performs **topology optimization** on a 2-D rectangular grid of
mass points connected by linear springs. An iterative algorithm removes
the nodes with the lowest strain energy until the target mass fraction is
reached.

### Features

* Parametric design space (width x height)
* Boundary conditions (pin / roller supports) and external forces via UI
* FEM solver (Hooke's law, stiffness matrix, penalty BCs)
* Iterative node removal with graph connectivity checking
* Spatial sensitivity filter for regularisation
* Live progress display, deformation visualization, energy heatmap
* Save / load state (JSON)
* Image export (PNG)
* MBB beam benchmark preset (full & half-symmetry)
* Structured logging (`logging` module) across all modules
* Explicit error handling with user-friendly messages in the UI
* Input validation on all public APIs

---

## Project Structure

```
topology-optimization/
├── src/
│   ├── __init__.py          # Package metadata & version
│   ├── models/
│   │   ├── node.py          # Node (mass point)
│   │   ├── spring.py        # Linear spring
│   │   └── structure.py     # Graph-based structure (networkx)
│   ├── solver/
│   │   ├── fem_solver.py    # Stiffness matrix assembly & solver
│   │   └── optimizer.py     # Iterative topology optimizer
│   ├── utils/
│   │   ├── visualization.py # Matplotlib plots
│   │   └── io_handler.py    # JSON save / load
│   └── presets/
│       └── mbb_beam.py      # MBB beam benchmark
├── tests/
│   ├── conftest.py          # Shared pytest fixtures
│   ├── test_node.py         # Node model tests
│   ├── test_spring.py       # Spring model tests
│   ├── test_structure.py    # Structure model tests
│   ├── test_fem_solver.py   # FEM solver tests
│   ├── test_optimizer.py    # Topology optimizer tests
│   ├── test_mbb_beam.py     # MBB beam preset tests
│   ├── test_io_handler.py   # JSON I/O tests
│   └── test_visualization.py# Visualization tests
├── app.py                   # Streamlit web UI
├── requirements.txt
├── README.md
├── LICENSE
└── docs/
    └── Aufgabenstellung.pdf
```

---

## Installation

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

1. **Sidebar**: Select a preset (e.g. *MBB Beam*), set width and height,
   then click **🔨 Create Structure**.
2. **Boundary Conditions & Forces** (expandable): Select a node, set supports,
   define forces.
3. **Optimization**: Set target mass fraction and removal rate, then
   click **▶️ Start Optimization** or **⏩ Single Step**.
4. **Visualization**: Switch between the tabs *Initial Structure*, *Current Structure*,
   *Deformation* and *Heatmap*.
5. **Export**: Download the result as PNG or the full state as JSON.

### MBB Beam Demo

1. Preset -> **MBB Beam**, Width = 30, Height = 10.
2. Create structure.
3. Target mass fraction = 0.5, Removal rate = 3.
4. Start optimization -> the characteristic arch/truss topology
   should emerge.

---

## Architecture

| Class              | Description                                                           |
|--------------------|-----------------------------------------------------------------------|
| `Node`             | Mass point with coordinates, boundary conditions, forces              |
| `Spring`           | Spring with stiffness matrix (globally transformed) and energy calculation |
| `Structure`        | NetworkX graph of nodes & springs, connectivity checking              |
| `FEMSolver`        | Assembles $K_g$, applies BCs, solves $K \cdot u = F$                 |
| `TopologyOptimizer`| Iterative loop: solve -> energy -> remove -> validate                   |
| `Visualizer`       | Matplotlib plots (structure, deformation, heatmap)                    |

### Logging

All modules use Python's built-in `logging` module.  The Streamlit app
configures the root logger at startup; backend modules log technical
details at `DEBUG` / `INFO` level while user-facing errors are surfaced
through `st.error()` / `st.warning()`.

### Error Handling

Public APIs validate inputs and raise descriptive exceptions (`ValueError`,
`KeyError`, `OSError`, ...).  The Streamlit app wraps all I/O, solver, and
rendering calls in `try / except` blocks so that failures are reported as
friendly toast messages rather than raw tracebacks.

### Physical Model

* Coordinate system: origin top-left, x -> right, z -> down.
* Spring constants: horizontal/vertical $k = 1$ N/m, diagonal $k = 1/\sqrt{2}$ N/m.
* Node mass: $m = 1$ kg.
* Hooke's law: $K \cdot u = F$.
* Strain energy: $c_e = \frac{1}{2} u_e^T K_e u_e$.

---

## Testing

The project includes a comprehensive test suite with **132+ unit tests** covering
all modules. Tests are written with [pytest](https://docs.pytest.org/).

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run tests for a specific module
python -m pytest tests/test_node.py -v

# Run with short summary
python -m pytest tests/ --tb=short
```

| Test file               | Module covered      | Tests |
|-------------------------|---------------------|-------|
| `test_node.py`          | `Node`              | 13    |
| `test_spring.py`        | `Spring`            | 12    |
| `test_structure.py`     | `Structure`         | 22    |
| `test_fem_solver.py`    | `FEMSolver`         | 10    |
| `test_optimizer.py`     | `TopologyOptimizer` | 17    |
| `test_mbb_beam.py`      | `create_mbb_beam`   | 14    |
| `test_io_handler.py`    | `io_handler`        | 9     |
| `test_visualization.py` | `Visualizer`        | 11    |

Shared fixtures (e.g. pre-built structures, nodes) live in `tests/conftest.py`.

---

## License

MIT License - see [LICENSE](LICENSE).