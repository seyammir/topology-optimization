"""SIMP (Solid Isotropic Material with Penalization) topology optimiser.

Operates on *spring densities* rather than continuum elements.  Each
spring is assigned a continuous design variable ``x_e in [x_min, 1]``
that scales its stiffness as ``x_e^p * K_e^0``.  The Optimality
Criteria (OC) update scheme is used to drive the densities towards the
target volume (mass) fraction.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ..models.structure import Structure
from .fem_solver import FEMSolver
from .optimizer_base import OptimizerBase, OptimizationResult

logger = logging.getLogger(__name__)


class SIMPOptimizer(OptimizerBase):
    """Density-based SIMP optimiser with Optimality Criteria update.

    Parameters
    ----------
    target_mass_fraction : float
        Target volume/mass fraction to retain.
    filter_radius : float
        Spatial sensitivity-filter radius (same cone filter as INR).
    penalization : float
        SIMP penalization exponent *p* (default 3).
    move_limit : float
        OC move limit *m* - maximum change per variable per iteration.
    x_min : float
        Minimum density (avoid singularity).
    max_iterations : int
        Hard cap on number of iterations.
    convergence_tol : float
        Stop when ``max |x_new - x_old| < tol``.
    solver : FEMSolver | None
    """

    def __init__(
        self,
        target_mass_fraction: float = 0.5,
        filter_radius: float = 1.5,
        penalization: float = 3.0,
        move_limit: float = 0.2,
        x_min: float = 1e-3,
        max_iterations: int = 200,
        convergence_tol: float = 0.01,
        solver: FEMSolver | None = None,
    ) -> None:
        super().__init__(
            target_mass_fraction=target_mass_fraction,
            filter_radius=filter_radius,
            solver=solver,
        )
        self.penalization = penalization
        self.move_limit = move_limit
        self.x_min = x_min
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol

    # Public API

    def optimize(
        self,
        structure: Structure,
        callback: Callable[[Structure, int, dict[int, float]], None] | None = None,
    ) -> OptimizationResult:
        """Run the full SIMP optimisation loop."""
        result = OptimizationResult(algorithm="SIMP")
        result.penalization = self.penalization
        result.history.append(structure.snapshot())

        # Initialise densities: every spring starts at 1.0
        springs = structure.get_springs()
        densities: dict[tuple[int, int], float] = {}
        for sp in springs:
            densities[sp.node_ids] = 1.0

        # Pre-compute total spring "volume" (length) for mass constraint
        spring_volumes = {
            sp.node_ids: sp.length for sp in springs
        }
        total_volume = sum(spring_volumes.values())
        target_volume = self.target_mass_fraction * total_volume

        for iteration in range(1, self.max_iterations + 1):
            logger.info("SIMP iteration %d", iteration)

            # 1) Solve with penalised stiffnesses
            structure.renumber_dofs()
            u = self.solver.solve_with_densities(
                structure, densities, self.penalization,
            )

            # 2) Compute sensitivities  dc/dx_e = -p * x_e^(p-1) * u_e^T K_e^0 u_e
            sensitivities: dict[tuple[int, int], float] = {}
            total_compliance = 0.0
            for sp in springs:
                key = sp.node_ids
                xe = densities[key]
                dofs = sp.dof_indices
                u_e = u[dofs]
                ke0 = sp.local_stiffness_matrix()  # un-penalised
                se = float(u_e @ ke0 @ u_e)  # u_e^T K_e^0 u_e
                total_compliance += xe ** self.penalization * se
                sensitivities[key] = -self.penalization * xe ** (self.penalization - 1) * se

            result.compliance_history.append(total_compliance)

            # 3) Filter sensitivities (spring-centre spatial filter)
            if self.filter_radius > 0:
                sensitivities = self._filter_spring_sensitivities(
                    structure, sensitivities, self.filter_radius,
                )

            # 4) OC update with bisection
            old_densities = dict(densities)
            densities = self._oc_update(
                densities, sensitivities, spring_volumes,
                target_volume, self.move_limit, self.x_min,
            )

            # Record per-node "energy" for visualisation (average SE per node)
            node_energies = self._compute_node_energies_from_densities(
                structure, u, densities,
            )
            result.energies_history.append(node_energies)

            # Convergence check
            max_change = max(
                abs(densities[k] - old_densities[k]) for k in densities
            )
            logger.info(
                "  compliance=%.6g, max_change=%.4f, vol_frac=%.3f",
                total_compliance, max_change,
                sum(densities[k] * spring_volumes[k] for k in densities) / total_volume,
            )

            if callback is not None:
                # Expose current densities on the structure so the
                # callback can compute live metrics (vol fraction etc.)
                structure._simp_densities = dict(densities)        # type: ignore[attr-defined]
                structure._simp_spring_volumes = dict(spring_volumes)  # type: ignore[attr-defined]
                callback(structure, iteration, node_energies)

            if max_change < self.convergence_tol:
                logger.info("SIMP converged at iteration %d.", iteration)
                break

        result.iterations = iteration  # type: ignore[possibly-undefined]
        result.densities = dict(densities)
        return result

    def step(
        self,
        structure: Structure,
    ) -> tuple[np.ndarray, dict[int, float], int]:
        """Execute a single SIMP iteration.

        Uses ``_simp_densities`` stored on the structure (or initialises
        them to 1.0 on first call).

        Returns (u, node_energies, 0).  The third element is always 0
        because SIMP does not remove nodes.
        """
        # Persistent density storage on the structure object
        if not hasattr(structure, "_simp_densities"):
            structure._simp_densities = {  # type: ignore[attr-defined]
                sp.node_ids: 1.0 for sp in structure.get_springs()
            }
            structure._simp_spring_volumes = {  # type: ignore[attr-defined]
                sp.node_ids: sp.length for sp in structure.get_springs()
            }

        densities = structure._simp_densities  # type: ignore[attr-defined]
        spring_volumes = structure._simp_spring_volumes  # type: ignore[attr-defined]
        total_volume = sum(spring_volumes.values())
        target_volume = self.target_mass_fraction * total_volume

        structure.renumber_dofs()
        u = self.solver.solve_with_densities(
            structure, densities, self.penalization,
        )

        sensitivities: dict[tuple[int, int], float] = {}
        for sp in structure.get_springs():
            key = sp.node_ids
            xe = densities[key]
            dofs = sp.dof_indices
            u_e = u[dofs]
            ke0 = sp.local_stiffness_matrix()
            se = float(u_e @ ke0 @ u_e)
            sensitivities[key] = -self.penalization * xe ** (self.penalization - 1) * se

        if self.filter_radius > 0:
            sensitivities = self._filter_spring_sensitivities(
                structure, sensitivities, self.filter_radius,
            )

        densities = self._oc_update(
            densities, sensitivities, spring_volumes,
            target_volume, self.move_limit, self.x_min,
        )
        structure._simp_densities = densities  # type: ignore[attr-defined]

        node_energies = self._compute_node_energies_from_densities(
            structure, u, densities,
        )
        return u, node_energies, 0

    # Internals

    @staticmethod
    def _oc_update(
        densities: dict[tuple[int, int], float],
        sensitivities: dict[tuple[int, int], float],
        volumes: dict[tuple[int, int], float],
        target_volume: float,
        move: float,
        x_min: float,
    ) -> dict[tuple[int, int], float]:
        """Optimality Criteria update with bisection on the Lagrange multiplier."""
        keys = list(densities.keys())
        x_old = np.array([densities[k] for k in keys])
        dc = np.array([sensitivities.get(k, 0.0) for k in keys])
        vol = np.array([volumes[k] for k in keys])

        lam_lo, lam_hi = 1e-10, 1e10

        for _ in range(100):  # bisection iterations
            lam_mid = 0.5 * (lam_lo + lam_hi)
            # OC formula:  x_new = x_old * sqrt(-dc / (lam * vol))
            Be = np.sqrt(np.maximum(-dc / (lam_mid * vol), 1e-30))
            x_new = x_old * Be
            # Clamp by move limit
            x_new = np.maximum(x_old - move, np.minimum(x_old + move, x_new))
            # Clamp to [x_min, 1]
            x_new = np.clip(x_new, x_min, 1.0)

            current_vol = float(x_new @ vol)
            if current_vol > target_volume:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid

            if lam_hi - lam_lo < 1e-12:
                break

        return {k: float(x_new[i]) for i, k in enumerate(keys)}

    @staticmethod
    def _filter_spring_sensitivities(
        structure: Structure,
        sensitivities: dict[tuple[int, int], float],
        filter_radius: float,
    ) -> dict[tuple[int, int], float]:
        """Spatial cone filter for spring-based sensitivities.

        Each spring's sensitivity is smoothed using the midpoint of
        the spring as its spatial coordinate, with a linear cone weight.
        """
        springs = structure.get_springs()
        if not springs:
            return {}

        keys = []
        coords = []
        sens_vals = []
        for sp in springs:
            k = sp.node_ids
            keys.append(k)
            mx = (sp.node_i.x + sp.node_j.x) / 2.0
            mz = (sp.node_i.z + sp.node_j.z) / 2.0
            coords.append([mx, mz])
            sens_vals.append(sensitivities.get(k, 0.0))

        coords_arr = np.array(coords)
        sens_arr = np.array(sens_vals)

        diff = coords_arr[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        weights = np.maximum(0.0, filter_radius - dist)

        denom = weights.sum(axis=1)
        denom[denom == 0] = 1.0
        filtered = (weights @ sens_arr) / denom

        return {keys[i]: float(filtered[i]) for i in range(len(keys))}

    @staticmethod
    def _compute_node_energies_from_densities(
        structure: Structure,
        u: np.ndarray,
        densities: dict[tuple[int, int], float],
    ) -> dict[int, float]:
        """Compute per-node average strain energy weighted by density."""
        totals: dict[int, float] = {nid: 0.0 for nid in structure.get_node_ids()}
        counts: dict[int, int] = {nid: 0 for nid in structure.get_node_ids()}
        for sp in structure.get_springs():
            key = sp.node_ids
            xe = densities.get(key, 1.0)
            se = xe * sp.strain_energy(u)
            ni, nj = key
            totals[ni] += se
            totals[nj] += se
            counts[ni] += 1
            counts[nj] += 1
        for nid in totals:
            if counts[nid] > 0:
                totals[nid] /= counts[nid]
        return totals
