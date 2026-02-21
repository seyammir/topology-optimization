"""Iterative topology optimiser.

Repeatedly: solve -> compute strain energies -> remove lowest-energy
unprotected nodes -> validate connectivity.  Stops when the target mass
fraction is reached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..models.node import Node
from ..models.spring import Spring
from ..models.structure import Structure
from .fem_solver import FEMSolver

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container returned by :meth:`TopologyOptimizer.optimize`."""
    history: list[Structure] = field(default_factory=list)
    energies_history: list[dict[int, float]] = field(default_factory=list)
    compliance_history: list[float] = field(default_factory=list)
    iterations: int = 0


class TopologyOptimizer:
    """Remove low-energy nodes iteratively until the target mass is met.

    Parameters
    ----------
    target_mass_fraction : float
        Fraction of original mass to keep (e.g. 0.5 = 50 %).
    removal_per_iteration : int
        Maximum number of nodes to try removing per iteration.
    filter_radius : float
        Spatial sensitivity-filter radius.  Energies are smoothed over
        all neighbours within this Euclidean distance.  A value of 0
        disables the filter.  Typical good value: 1.5 - 3.0.
    """

    def __init__(
        self,
        target_mass_fraction: float = 0.5,
        removal_per_iteration: int = 1,
        filter_radius: float = 1.5,
        solver: FEMSolver | None = None,
    ) -> None:
        if not 0.0 < target_mass_fraction < 1.0:
            raise ValueError("target_mass_fraction must be in (0, 1)")
        self.target_mass_fraction = target_mass_fraction
        self.removal_per_iteration = max(1, removal_per_iteration)
        self.filter_radius = max(0.0, filter_radius)
        self.solver = solver if solver is not None else FEMSolver()

    # Public API
    def optimize(
        self,
        structure: Structure,
        callback: Callable[[Structure, int, dict[int, float]], None] | None = None,
    ) -> OptimizationResult:
        """Run the full optimisation loop.

        Parameters
        ----------
        structure : Structure
            The *mutable* structure that will be modified in-place.
        callback : callable, optional
            ``callback(structure, iteration, node_energies)`` is called
            after each iteration for live visualisation.

        Returns
        -------
        OptimizationResult
        """
        initial_mass = structure.total_mass()
        target_mass = self.target_mass_fraction * initial_mass
        result = OptimizationResult()

        # Store initial snapshot.
        result.history.append(structure.snapshot())

        iteration = 0
        while structure.total_mass() > target_mass:
            iteration += 1
            logger.info(
                "Iteration %d - nodes: %d, mass: %.1f / %.1f",
                iteration, structure.num_nodes,
                structure.total_mass(), target_mass,
            )

            # 1) Solve
            structure.renumber_dofs()
            u = self.solver.solve(structure)

            # 2) Compute per-node relevance energy + spatial filter
            raw_energies = self._compute_node_energies(structure, u)
            node_energies = (
                self._filter_energies(structure, raw_energies, self.filter_radius)
                if self.filter_radius > 0
                else raw_energies
            )
            result.energies_history.append(dict(node_energies))

            total_compliance = sum(raw_energies.values())
            result.compliance_history.append(total_compliance)

            # 3) Sort removable candidates by ascending filtered energy
            protected = structure.get_protected_node_ids()
            candidates = [
                (nid, e)
                for nid, e in sorted(node_energies.items(), key=lambda x: x[1])
                if nid not in protected
            ]

            # 4) Try removing up to `removal_per_iteration` nodes
            removed = 0
            for nid, energy in candidates:
                if removed >= self.removal_per_iteration:
                    break
                if structure.total_mass() <= target_mass:
                    break

                # Tentative removal - record neighbours first
                neighbor_ids = list(structure.graph.neighbors(nid))
                node_backup = structure.get_node(nid)
                springs_backup = structure.get_springs_for_node(nid)
                structure.remove_node(nid)

                # Validate: connectivity + no mechanism in affected neighbours
                neighbours_ok = not any(
                    structure.node_is_mechanism(nb)
                    for nb in neighbor_ids
                    if nb in structure.graph
                )
                if (
                    neighbours_ok
                    and structure.num_nodes > 0
                    and structure.is_connected()
                    and structure.supports_reachable_from_loads()
                ):
                    removed += 1
                    logger.debug("  Removed node %d (energy=%.6f)", nid, energy)
                else:
                    # Rollback
                    self._restore_node(structure, node_backup, springs_backup)
                    logger.debug("  Skipped node %d (would disconnect/mechanism)", nid)

            if removed == 0:
                logger.warning("No node could be removed - stopping early.")
                break

            # Snapshot after this iteration
            result.history.append(structure.snapshot())

            if callback is not None:
                callback(structure, iteration, node_energies)

        # Post-processing: remove dangling (dead-end) members
        dangling = structure.remove_dangling_nodes()
        if dangling > 0:
            logger.info("Post-processing: removed %d dangling nodes.", dangling)
            result.history.append(structure.snapshot())

        result.iterations = iteration
        return result

    # Single step (for UI "Step" button)
    def step(
        self,
        structure: Structure,
    ) -> tuple[np.ndarray, dict[int, float], int]:
        """Execute a single optimisation step.

        Returns
        -------
        u : ndarray
            Displacement vector after solving.
        node_energies : dict[int, float]
            Per-node relevance energy.
        removed_count : int
            How many nodes were actually removed.
        """
        structure.renumber_dofs()
        u = self.solver.solve(structure)
        raw_energies = self._compute_node_energies(structure, u)
        node_energies = (
            self._filter_energies(structure, raw_energies, self.filter_radius)
            if self.filter_radius > 0
            else raw_energies
        )

        protected = structure.get_protected_node_ids()
        candidates = [
            (nid, e)
            for nid, e in sorted(node_energies.items(), key=lambda x: x[1])
            if nid not in protected
        ]

        removed = 0
        for nid, energy in candidates:
            if removed >= self.removal_per_iteration:
                break

            neighbor_ids = list(structure.graph.neighbors(nid))
            node_backup = structure.get_node(nid)
            springs_backup = structure.get_springs_for_node(nid)
            structure.remove_node(nid)

            neighbours_ok = not any(
                structure.node_is_mechanism(nb)
                for nb in neighbor_ids
                if nb in structure.graph
            )
            if (
                neighbours_ok
                and structure.num_nodes > 0
                and structure.is_connected()
                and structure.supports_reachable_from_loads()
            ):
                removed += 1
            else:
                self._restore_node(structure, node_backup, springs_backup)

        return u, node_energies, removed

    # Internals
    @staticmethod
    def _compute_node_energies(
        structure: Structure, u: np.ndarray
    ) -> dict[int, float]:
        """Average strain energy per spring for each node.

        Using the *average* (instead of the sum) prevents boundary nodes
        from being penalised simply for having fewer springs.  This
        produces much better topologies because interior and boundary
        nodes are compared on an equal footing.
        """
        totals: dict[int, float] = {nid: 0.0 for nid in structure.get_node_ids()}
        counts: dict[int, int] = {nid: 0 for nid in structure.get_node_ids()}
        for spring in structure.get_springs():
            se = spring.strain_energy(u)
            ni, nj = spring.node_ids
            totals[ni] += se
            totals[nj] += se
            counts[ni] += 1
            counts[nj] += 1
        # Normalise to average energy per spring
        for nid in totals:
            if counts[nid] > 0:
                totals[nid] /= counts[nid]
        return totals

    @staticmethod
    def _filter_energies(
        structure: Structure,
        raw_energies: dict[int, float],
        filter_radius: float = 3.0,
    ) -> dict[int, float]:
        r"""Spatial sensitivity filter (density-filter approach).

        For each node *i* the filtered energy is:

        .. math::

            \tilde{e}_i = \frac{\sum_j w_{ij}\, e_j}{\sum_j w_{ij}}
            \quad\text{with } w_{ij} = \max(0,\; r - d_{ij})

        where *r* is ``filter_radius`` and *d* the Euclidean distance.

        This prevents isolated low-energy nodes from being removed
        when they are surrounded by high-energy members, which is the
        standard regularisation technique in topology optimisation.
        """
        nodes = structure.get_nodes()
        if not nodes:
            return {}

        nids = [n.id for n in nodes]
        coords = np.array([[n.x, n.z] for n in nodes])  # (N, 2)
        energies = np.array([raw_energies.get(n.id, 0.0) for n in nodes])  # (N,)

        # Pairwise Euclidean distances  (N, N)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)

        # Linear cone weights - nodes within radius get positive weight
        weights = np.maximum(0.0, filter_radius - dist)

        denom = weights.sum(axis=1)
        denom[denom == 0] = 1.0  # avoid division by zero
        filtered = (weights @ energies) / denom

        return dict(zip(nids, filtered.tolist()))

    @staticmethod
    def _restore_node(
        structure: Structure,
        node: Node,
        springs: list[Spring],
    ) -> None:
        """Re-insert a previously removed node and its springs."""
        structure.graph.add_node(node.id, obj=node)
        for sp in springs:
            ni, nj = sp.node_i, sp.node_j
            # Only re-add if both end-nodes are (still) present.
            if ni.id in structure.graph and nj.id in structure.graph:
                structure.graph.add_edge(ni.id, nj.id, obj=sp)
