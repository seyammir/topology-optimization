"""Node (mass point) in the mass-spring system.

Coordinate system: origin top-left, x → right, z → down.
Each node carries 2 DOFs (u_x, u_z), optional boundary conditions and
external forces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    """A single mass point in the 2-D structure.

    Parameters
    ----------
    id : int
        Unique identifier for this node.
    x : float
        Horizontal position (origin at left).
    z : float
        Vertical position (origin at top, positive downward).
    mass : float
        Lumped mass associated with this node [kg].
    fixed_x : bool
        If ``True`` the horizontal DOF is constrained (= 0).
    fixed_z : bool
        If ``True`` the vertical DOF is constrained (= 0).
    fx : float
        External force in x-direction [N].
    fz : float
        External force in z-direction [N].
    """

    id: int
    x: float
    z: float
    mass: float = 1.0
    fixed_x: bool = False
    fixed_z: bool = False
    fx: float = 0.0
    fz: float = 0.0

    # Assigned later by the Structure during DOF numbering.
    _dof_x: int = field(default=-1, repr=False)
    _dof_z: int = field(default=-1, repr=False)

    # DOF helpers
    @property
    def dof_indices(self) -> tuple[int, int]:
        """Return the global DOF indices ``(dof_x, dof_z)``."""
        return (self._dof_x, self._dof_z)

    @dof_indices.setter
    def dof_indices(self, value: tuple[int, int]) -> None:
        self._dof_x, self._dof_z = value

    # Boundary-condition helpers
    @property
    def is_fixed(self) -> bool:
        """``True`` if *any* DOF is constrained (pin or roller)."""
        return self.fixed_x or self.fixed_z

    @property
    def is_pinned(self) -> bool:
        """``True`` if *both* DOFs are constrained (pin support)."""
        return self.fixed_x and self.fixed_z

    @property
    def has_load(self) -> bool:
        """``True`` if an external force is applied."""
        return self.fx != 0.0 or self.fz != 0.0

    @property
    def is_protected(self) -> bool:
        """A node is protected (cannot be removed) when it is a support
        or carries an external load."""
        return self.is_fixed or self.has_load

    # Serialisation
    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-friendly)."""
        return {
            "id": self.id,
            "x": self.x,
            "z": self.z,
            "mass": self.mass,
            "fixed_x": self.fixed_x,
            "fixed_z": self.fixed_z,
            "fx": self.fx,
            "fz": self.fz,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Node:
        """Reconstruct a :class:`Node` from a dict."""
        return cls(
            id=data["id"],
            x=data["x"],
            z=data["z"],
            mass=data.get("mass", 1.0),
            fixed_x=data.get("fixed_x", False),
            fixed_z=data.get("fixed_z", False),
            fx=data.get("fx", 0.0),
            fz=data.get("fz", 0.0),
        )

    # Dunder helpers
    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Node):
            return self.id == other.id
        return NotImplemented
