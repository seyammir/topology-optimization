"""Linear-elastic spring connecting two nodes.

The spring stiffness matrix is computed in the global coordinate system
using the standard rotation/transformation approach for a 2-D bar element.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .node import Node


class Spring:
    """A linear-elastic spring between two :class:`Node` objects.

    Parameters
    ----------
    node_i, node_j : Node
        The two end-nodes.
    k : float | None
        Spring constant [N/m].  If *None* the value is inferred
        automatically:
        * horizontal / vertical → ``1.0``
        * diagonal (45°)       → ``1 / sqrt(2)``
    """

    def __init__(self, node_i: Node, node_j: Node, k: float | None = None) -> None:
        self.node_i = node_i
        self.node_j = node_j

        dx = node_j.x - node_i.x
        dz = node_j.z - node_i.z
        self._length = math.hypot(dx, dz)
        self._angle = math.atan2(dz, dx)  # radians

        if k is not None:
            self.k = k
        else:
            # Auto-detect: if both dx and dz are non-zero → diagonal
            if dx != 0 and dz != 0:
                self.k = 1.0 / math.sqrt(2)
            else:
                self.k = 1.0

    # Properties
    @property
    def angle(self) -> float:
        """Angle of the spring w.r.t. the positive x-axis [rad]."""
        return self._angle

    @property
    def length(self) -> float:
        """Undeformed length of the spring."""
        return self._length

    @property
    def dof_indices(self) -> list[int]:
        """Global DOF indices ``[ix, iz, jx, jz]``."""
        ix, iz = self.node_i.dof_indices
        jx, jz = self.node_j.dof_indices
        return [ix, iz, jx, jz]

    @property
    def node_ids(self) -> tuple[int, int]:
        """Return ``(node_i.id, node_j.id)``."""
        return (self.node_i.id, self.node_j.id)

    # Stiffness matrix
    def local_stiffness_matrix(self) -> np.ndarray:
        r"""Return the 4x4 element stiffness matrix in global coordinates.

        The transformation follows the standard bar-element formulation:

        $$K_e = k \begin{bmatrix} c^2 & cs & -c^2 & -cs \\ cs & s^2 & -cs & -s^2 \\ -c^2 & -cs & c^2 & cs \\ -cs & -s^2 & cs & s^2 \end{bmatrix}$$

        where *c* = cos(θ) and *s* = sin(θ).
        """
        c = math.cos(self._angle)
        s = math.sin(self._angle)
        cc, ss, cs_ = c * c, s * s, c * s
        ke = self.k * np.array(
            [
                [ cc,  cs_, -cc, -cs_],
                [ cs_,  ss, -cs_, -ss],
                [-cc, -cs_,  cc,  cs_],
                [-cs_, -ss,  cs_,  ss],
            ],
            dtype=float,
        )
        return ke

    # Strain energy
    def strain_energy(self, u_global: np.ndarray) -> float:
        r"""Compute the elastic strain energy stored in this spring.

        $$c_e = \tfrac{1}{2} \, u_e^T \, K_e \, u_e$$

        Parameters
        ----------
        u_global : ndarray, shape (2*N,)
            Full global displacement vector.

        Returns
        -------
        float
            Strain energy of this element.
        """
        dofs = self.dof_indices
        u_e = u_global[dofs]
        ke = self.local_stiffness_matrix()
        return 0.5 * float(u_e @ ke @ u_e)

    # Serialisation
    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict (references nodes by id)."""
        return {
            "node_i": self.node_i.id,
            "node_j": self.node_j.id,
            "k": self.k,
        }

    def __repr__(self) -> str:
        return (
            f"Spring({self.node_i.id}↔{self.node_j.id}, "
            f"k={self.k:.4f}, θ={math.degrees(self._angle):.1f}°)"
        )
