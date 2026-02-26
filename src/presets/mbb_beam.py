"""Messerschmitt-Bölkow-Blohm (MBB) beam preset.

The MBB beam is a standard benchmark for 2-D topology optimisation.

Full-beam setup
---------------
* Rectangular domain of *nx* x *nz* cells.
* Bottom-left node: pin support  (u_x = u_z = 0).
* Bottom-right node: roller      (u_z = 0).
* Load: vertical force F_z at the **top-centre** node.

Half-beam (symmetry)
--------------------
Optionally the left-half symmetry model can be created:
* Left edge: all nodes have u_x = 0 (roller support).
* Bottom-right node: u_z = 0.
* Load at top-left corner (F_z downward).
"""

from __future__ import annotations

import logging

from ..models.structure import Structure

logger = logging.getLogger(__name__)


def create_mbb_beam(
    nx: int = 30,
    nz: int = 10,
    load: float = -1.0,
    half: bool = False,
) -> Structure:
    """Return a fully configured MBB beam :class:`Structure`.

    Parameters
    ----------
    nx : int
        Number of cells in x-direction.
    nz : int
        Number of cells in z-direction.
    load : float
        Vertical force magnitude (negative = downward in z-up convention,
        but our z-axis points **down**, so a *positive* value pushes
        downward).  Default is -1.0  (^ upward in screen coords -> we use
        positive 1.0 internally since z is downward).
    half : bool
        If ``True`` generate the half-symmetry model instead of the full
        beam.

    Returns
    -------
    Structure

    Raises
    ------
    ValueError
        If *nx* or *nz* is less than 1, or *load* is zero.
    """
    if nx < 1 or nz < 1:
        raise ValueError(f"nx and nz must be >= 1, got nx={nx}, nz={nz}")
    if load == 0.0:
        raise ValueError("Load must be non-zero for a meaningful MBB beam")

    struct = Structure.create_rectangular(nx, nz)

    cols = nx + 1
    rows = nz + 1

    # Helper: node-id from grid position (row-major, top-to-bottom).
    def _nid(ix: int, iz: int) -> int:
        return iz * cols + ix

    if half:
        # Half-beam (left half with symmetry BCs)
        # Left edge: u_x = 0 (roller on every left-edge node)
        for iz in range(rows):
            node = struct.get_node(_nid(0, iz))
            node.fixed_x = True
        # Bottom-right node: u_z = 0
        br = struct.get_node(_nid(nx, nz))
        br.fixed_z = True
        # Load at top-left corner
        tl = struct.get_node(_nid(0, 0))
        tl.fz = abs(load)  # positive -> downward
    else:
        # Full beam
        # Bottom-left: pin support
        bl = struct.get_node(_nid(0, nz))
        bl.fixed_x = True
        bl.fixed_z = True
        # Bottom-right: roller (only u_z fixed)
        br = struct.get_node(_nid(nx, nz))
        br.fixed_z = True
        # Load at top-centre
        mid_x = nx // 2
        tc = struct.get_node(_nid(mid_x, 0))
        tc.fz = abs(load)  # positive -> downward

    mode = "half-symmetry" if half else "full"
    logger.info(
        "Created MBB beam (%s): nx=%d, nz=%d, load=%.2f, nodes=%d",
        mode, nx, nz, load, struct.num_nodes,
    )
    return struct