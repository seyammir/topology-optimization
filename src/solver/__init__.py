"""Solver package - FEM solver and topology optimizers."""

from .fem_solver import FEMSolver
from .optimizer_base import OptimizerBase, OptimizationResult
from .optimizer import NodeRemovalOptimizer, TopologyOptimizer
from .simp_optimizer import SIMPOptimizer

__all__ = [
    "FEMSolver",
    "OptimizerBase",
    "OptimizationResult",
    "NodeRemovalOptimizer",
    "TopologyOptimizer",  # backward-compat alias
    "SIMPOptimizer",
]
