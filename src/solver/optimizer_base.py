"""Abstract base class for topology optimisers.

All concrete optimisers (Node Removal, SIMP, ...) inherit from
:class:`OptimizerBase` and implement ``optimize`` / ``step``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

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

    # Serialisation helpers
    @staticmethod
    def _key_to_str(key: tuple[int, int]) -> str:
        return f"{key[0]},{key[1]}"

    @staticmethod
    def _str_to_key(s: str) -> tuple[int, int]:
        a, b = s.split(",")
        return (int(a), int(b))

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a JSON-compatible dict."""
        # Convert density dicts: tuple keys -> string keys
        def _dens_list(lst: list[dict[tuple[int, int], float]]) -> list[dict[str, float]]:
            return [
                {self._key_to_str(k): v for k, v in snap.items()}
                for snap in lst
            ]

        data: dict[str, Any] = {
            "algorithm": self.algorithm,
            "iterations": self.iterations,
            "penalization": self.penalization,
            "compliance_history": list(self.compliance_history),
        }
        if self.densities is not None:
            data["densities"] = {
                self._key_to_str(k): v for k, v in self.densities.items()
            }
        if self.density_history:
            data["density_history"] = _dens_list(self.density_history)
        if self.energies_history:
            data["energies_history"] = [
                {str(k): v for k, v in snap.items()}
                for snap in self.energies_history
            ]
        if self.history:
            data["history"] = [s.to_dict() for s in self.history]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationResult":
        """Reconstruct an :class:`OptimizationResult` from a dict."""
        result = cls(algorithm=data.get("algorithm", ""))
        result.iterations = data.get("iterations", 0)
        result.penalization = data.get("penalization", 3.0)
        result.compliance_history = data.get("compliance_history", [])

        raw_dens = data.get("densities")
        if raw_dens is not None:
            result.densities = {
                cls._str_to_key(k): v for k, v in raw_dens.items()
            }
        raw_hist = data.get("density_history")
        if raw_hist:
            result.density_history = [
                {cls._str_to_key(k): v for k, v in snap.items()}
                for snap in raw_hist
            ]
        raw_energies = data.get("energies_history")
        if raw_energies:
            result.energies_history = [
                {int(k): v for k, v in snap.items()}
                for snap in raw_energies
            ]
        raw_history = data.get("history")
        if raw_history:
            result.history = [Structure.from_dict(s) for s in raw_history]
        return result


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