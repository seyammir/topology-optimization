"""Finite-Element (mass-spring) solver.

Assembles the global stiffness matrix from all springs, applies
boundary conditions via the penalty method, and solves
:math:`K_g \\cdot u = F` for the displacement vector *u*.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..models.structure import Structure

logger = logging.getLogger(__name__)


class FEMSolver:
    """Static linear-elastic solver for mass-spring structures."""

    # Large penalty value for enforcing BCs.
    PENALTY = 1.0e10

    # Small regularisation to prevent singularity from mechanism modes.
    # This acts like a very soft "ground spring" on every DOF.
    REGULARISATION = 1.0e-6

    # Public API
    def solve(self, structure: Structure) -> np.ndarray:
        """Assemble and solve the system, returning the displacement vector.

        Parameters
        ----------
        structure : Structure
            The current (possibly reduced) structure.  DOFs must have been
            numbered (``renumber_dofs()``).

        Returns
        -------
        u : ndarray, shape (num_dofs,)
            Global displacement vector.
        """
        structure.renumber_dofs()
        n_dof = structure.num_dofs
        K = self._assemble_global_stiffness(structure)
        F = self._build_force_vector(structure)
        K, F = self._apply_boundary_conditions(K, F, structure)

        # Small diagonal regularisation prevents singularity when some
        # nodes only have collinear springs (mechanism modes).
        K = K.tocsr() + self.REGULARISATION * sparse.eye(n_dof, format="csr")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", sparse.SparseEfficiencyWarning)
            warnings.filterwarnings("ignore", message=".*singular.*")
            u = spsolve(K.tocsc(), F)

        # Replace any NaN / Inf with 0 (safety net for near-singular K).
        if np.any(np.isnan(u)) or np.any(np.isinf(u)):
            logger.warning(
                "Solver produced NaN/Inf displacements — replacing with 0. "
                "The stiffness matrix may be (near-)singular."
            )
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        return u

    # Load-path analysis
    @staticmethod
    def compute_internal_forces(
        structure: Structure, u: np.ndarray,
    ) -> dict[tuple[int, int], dict]:
        """Compute the internal (axial) force in every spring.

        Parameters
        ----------
        structure : Structure
            The solved structure (DOFs must be numbered).
        u : ndarray, shape (num_dofs,)
            Global displacement vector from :meth:`solve`.

        Returns
        -------
        dict[(node_i_id, node_j_id), info]
            For each spring a dict with keys:

            * ``"axial_force"`` - signed scalar (positive = tension,
              negative = compression).
            * ``"abs_force"`` - absolute value.
            * ``"force_vec"`` - (fx, fz) global force vector acting on
              node_j (reaction to the spring elongation).
            * ``"node_i"`` / ``"node_j"`` - endpoint coordinates.
            * ``"angle"`` - spring angle [rad].
        """
        import math
        results: dict[tuple[int, int], dict] = {}
        for spring in structure.get_springs():
            dofs = spring.dof_indices  # [ix, iz, jx, jz]
            u_e = u[dofs]
            c = math.cos(spring.angle)
            s = math.sin(spring.angle)
            # Axial elongation: δ = c*(ux_j-ux_i) + s*(uz_j-uz_i)
            delta = c * (u_e[2] - u_e[0]) + s * (u_e[3] - u_e[1])
            axial = spring.k * delta  # positive = tension
            ni, nj = spring.node_ids
            results[(ni, nj)] = {
                "axial_force": axial,
                "abs_force": abs(axial),
                "force_vec": (axial * c, axial * s),
                "node_i": (spring.node_i.x, spring.node_i.z),
                "node_j": (spring.node_j.x, spring.node_j.z),
                "angle": spring.angle,
            }
        return results

    # Assembly
    @staticmethod
    def _assemble_global_stiffness(structure: Structure) -> sparse.lil_matrix:
        """Build the global stiffness matrix (sparse, LIL format)."""
        n_dof = structure.num_dofs
        K = sparse.lil_matrix((n_dof, n_dof), dtype=float)

        for spring in structure.get_springs():
            ke = spring.local_stiffness_matrix()  # 4x4
            dofs = spring.dof_indices               # [ix, iz, jx, jz]
            for r in range(4):
                for c in range(4):
                    K[dofs[r], dofs[c]] += ke[r, c]
        return K

    @staticmethod
    def _build_force_vector(structure: Structure) -> np.ndarray:
        """Assemble the global force vector from node external forces."""
        F = np.zeros(structure.num_dofs, dtype=float)
        for node in structure.get_nodes():
            dx, dz = node.dof_indices
            F[dx] += node.fx
            F[dz] += node.fz
        return F

    @classmethod
    def _apply_boundary_conditions(
        cls,
        K: sparse.lil_matrix,
        F: np.ndarray,
        structure: Structure,
    ) -> tuple[sparse.lil_matrix, np.ndarray]:
        """Enforce displacement BCs using the penalty method.

        For each fixed DOF the diagonal entry is set to ``PENALTY`` and
        the corresponding force entry is set to 0.
        """
        for node in structure.get_nodes():
            dx, dz = node.dof_indices
            if node.fixed_x:
                K[dx, dx] = cls.PENALTY
                F[dx] = 0.0
            if node.fixed_z:
                K[dz, dz] = cls.PENALTY
                F[dz] = 0.0
        return K, F
