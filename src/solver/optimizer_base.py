"""Abstract base class for topology optimisers.

All concrete optimisers (Node Removal, SIMP, ...) inherit from
:class:`OptimizerBase` and implement ``optimize`` / ``step``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..models.structure import Structure
from .fem_solver import FEMSolver

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container returned by any optimiser's ``optimize`` method.

    Attributes
    ----------
    history : list[Structure]
        Snapshots of the structure after each iteration (discrete methods)
        or the final state (density-based methods).
    energies_history : list[dict[int, float]]
        Per-node or per-spring energy values for each iteration.
    compliance_history : list[float]
        Scalar compliance (total strain energy) for each iteration.
    iterations : int
        Number of iterations executed.
    densities : dict[tuple[int, int], float] | None
        Final per-spring density field (SIMP only; *None* for discrete
        methods).
    algorithm : str
        Name of the algorithm that produced this result.
    """

    history: list[Structure] = field(default_factory=list)
    energies_history: list[dict[int, float]] = field(default_factory=list)
    compliance_history: list[float] = field(default_factory=list)
    density_history: list[dict[tuple[int, int], float]] = field(default_factory=list)
    iterations: int = 0
    densities: dict[tuple[int, int], float] | None = None
    penalization: float = 3.0
    algorithm: str = ""


class OptimizerBase(ABC):
    """Abstract base interface for topology optimisers.

    Parameters
    ----------
    target_mass_fraction : float
        Fraction of original mass/volume to retain (0, 1).
    filter_radius : float
        Spatial sensitivity-filter radius.
    solver : FEMSolver | None
        FEM solver instance; a default is created if *None*.
    """

    def __init__(
        self,
        target_mass_fraction: float = 0.5,
        filter_radius: float = 1.5,
        solver: FEMSolver | None = None,
    ) -> None:
        if not 0.0 < target_mass_fraction < 1.0:
            raise ValueError("target_mass_fraction must be in (0, 1)")
        self.target_mass_fraction = target_mass_fraction
        self.filter_radius = max(0.0, filter_radius)
        self.solver = solver if solver is not None else FEMSolver()

    # Abstract interface

    @abstractmethod
    def optimize(
        self,
        structure: Structure,
        callback: Callable[[Structure, int, dict[int, float]], None] | None = None,
    ) -> OptimizationResult:
        """Run the full optimisation loop.

        Parameters
        ----------
        structure : Structure
            Mutable structure modified in place (discrete methods) or
            used as the design domain (density methods).
        callback : callable, optional
            ``callback(structure, iteration, energies)`` for live feedback.

        Returns
        -------
        OptimizationResult
        """
        ...

    @abstractmethod
    def step(
        self,
        structure: Structure,
    ) -> tuple[np.ndarray, dict[int, float], int]:
        """Execute a single optimisation step.

        Returns
        -------
        u : ndarray
            Displacement vector.
        energies : dict[int, float]
            Per-node or per-spring relevance energies.
        count : int
            Number of elements removed / updated.
        """
        ...

    # Shared utilities

    @staticmethod
    def _filter_energies(
        structure: Structure,
        raw_energies: dict[int, float],
        filter_radius: float = 3.0,
    ) -> dict[int, float]:
        r"""Spatial sensitivity filter (linear-cone weighting).

        For each node *i* the filtered energy is:

        .. math::

            \tilde{e}_i = \frac{\sum_j w_{ij}\, e_j}{\sum_j w_{ij}}
            \quad\text{with } w_{ij} = \max(0,\; r - d_{ij})

        This is the standard regularisation technique in topology
        optimisation to prevent checkerboard / mesh-dependent artefacts.
        """
        nodes = structure.get_nodes()
        if not nodes:
            return {}

        nids = [n.id for n in nodes]
        coords = np.array([[n.x, n.z] for n in nodes])
        energies = np.array([raw_energies.get(n.id, 0.0) for n in nodes])

        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)

        weights = np.maximum(0.0, filter_radius - dist)
        denom = weights.sum(axis=1)
        denom[denom == 0] = 1.0
        filtered = (weights @ energies) / denom

        return dict(zip(nids, filtered.tolist()))