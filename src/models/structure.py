"""Structure - the mass-spring graph that represents the 2-D domain.

Internally the structure is stored as a ``networkx.Graph``.  Each graph
node holds a :class:`Node` object and each graph edge holds a :class:`Spring`
object.  This gives O(1) neighbour look-ups and built-in connectivity
checks.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any

import networkx as nx
import numpy as np

from .node import Node
from .spring import Spring

logger = logging.getLogger(__name__)


class Structure:
    """Graph-based mass-spring structure.

    The graph is stored in ``self.graph`` (a ``networkx.Graph``).

    * Graph-node keys are **node ids** (``int``).
    * Graph-node attribute ``"obj"`` -> the :class:`Node` instance.
    * Graph-edge attribute ``"obj"`` -> the :class:`Spring` instance.
    """

    def __init__(self) -> None:
        self.graph: nx.Graph = nx.Graph()
        self._next_node_id: int = 0

    # Factory - rectangular grid
    @classmethod
    def create_rectangular(cls, width: int, height: int) -> "Structure":
        """Create a rectangular grid of ``(width+1) x (height+1)`` nodes.

        Spacing between adjacent nodes is **1 unit** in both directions.
        Each rectangular cell is connected by:

        * 2 horizontal springs
        * 2 vertical springs
        * 2 diagonal springs (both diagonals)

        Parameters
        ----------
        width : int
            Number of cells in x-direction.  Must be >= 1.
        height : int
            Number of cells in z-direction.  Must be >= 1.

        Returns
        -------
        Structure
            Fully connected rectangular structure.

        Raises
        ------
        ValueError
            If *width* or *height* is less than 1.
        """
        if width < 1 or height < 1:
            raise ValueError(
                f"width and height must be >= 1, got width={width}, height={height}"
            )
        struct = cls()
        cols = width + 1
        rows = height + 1

        # 1) Create nodes row by row (top -> bottom).
        node_map: dict[tuple[int, int], Node] = {}
        for iz in range(rows):
            for ix in range(cols):
                node = struct.add_node(float(ix), float(iz))
                node_map[(ix, iz)] = node

        # 2) Create springs.
        for iz in range(rows):
            for ix in range(cols):
                n = node_map[(ix, iz)]
                # Right neighbour (horizontal)
                if ix + 1 < cols:
                    struct.add_spring(n, node_map[(ix + 1, iz)])
                # Bottom neighbour (vertical)
                if iz + 1 < rows:
                    struct.add_spring(n, node_map[(ix, iz + 1)])
                # Bottom-right diagonal
                if ix + 1 < cols and iz + 1 < rows:
                    struct.add_spring(n, node_map[(ix + 1, iz + 1)])
                # Bottom-left diagonal
                if ix - 1 >= 0 and iz + 1 < rows:
                    struct.add_spring(n, node_map[(ix - 1, iz + 1)])

        struct.renumber_dofs()
        logger.info(
            "Created rectangular structure: %dx%d (%d nodes, %d springs)",
            width, height, struct.num_nodes, struct.graph.number_of_edges(),
        )
        return struct

    # Node operations
    def add_node(self, x: float, z: float, **kwargs: Any) -> Node:
        """Add a new node and return it."""
        node = Node(id=self._next_node_id, x=x, z=z, **kwargs)
        self.graph.add_node(node.id, obj=node)
        self._next_node_id += 1
        return node

    def remove_node(self, node_id: int) -> None:
        """Remove a node *and* all incident springs from the graph."""
        if node_id not in self.graph:
            raise KeyError(f"Node {node_id} not in structure")
        logger.debug("Removing node %d", node_id)
        self.graph.remove_node(node_id)  # also removes edges

    def get_node(self, node_id: int) -> Node:
        """Return the :class:`Node` object for *node_id*.

        Raises
        ------
        KeyError
            If *node_id* does not exist in the structure.
        """
        if node_id not in self.graph:
            raise KeyError(f"Node {node_id} not in structure")
        return self.graph.nodes[node_id]["obj"]

    def get_nodes(self) -> list[Node]:
        """Return all nodes (unordered list)."""
        return [data["obj"] for _, data in self.graph.nodes(data=True)]

    def get_node_ids(self) -> list[int]:
        """Return all node ids."""
        return list(self.graph.nodes)

    # Spring operations
    def add_spring(self, node_i: Node, node_j: Node, k: float | None = None) -> Spring:
        """Create a spring between two existing nodes and return it."""
        spring = Spring(node_i, node_j, k=k)
        self.graph.add_edge(node_i.id, node_j.id, obj=spring)
        return spring

    def get_spring(self, id_i: int, id_j: int) -> Spring:
        """Return the spring between two nodes (order-independent).

        Raises
        ------
        KeyError
            If no spring exists between *id_i* and *id_j*.
        """
        if not self.graph.has_edge(id_i, id_j):
            raise KeyError(f"No spring between nodes {id_i} and {id_j}")
        return self.graph.edges[id_i, id_j]["obj"]

    def get_springs(self) -> list[Spring]:
        """Return all springs."""
        return [data["obj"] for _, _, data in self.graph.edges(data=True)]

    def get_springs_for_node(self, node_id: int) -> list[Spring]:
        """Return all springs incident to *node_id*."""
        springs: list[Spring] = []
        for neighbour in self.graph.neighbors(node_id):
            springs.append(self.graph.edges[node_id, neighbour]["obj"])
        return springs

    # DOF management
    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_dofs(self) -> int:
        return 2 * self.num_nodes

    def renumber_dofs(self) -> None:
        """Assign consecutive DOF indices 0, 1, 2, ... to all nodes.

        Call this after any structural modification (node removal) before
        solving.
        """
        for idx, nid in enumerate(sorted(self.graph.nodes)):
            node: Node = self.graph.nodes[nid]["obj"]
            node.dof_indices = (2 * idx, 2 * idx + 1)

    # Topology queries
    def is_connected(self) -> bool:
        """Check whether the graph is connected."""
        if self.num_nodes == 0:
            return False
        return nx.is_connected(self.graph)

    def total_mass(self) -> float:
        """Sum of all node masses."""
        return sum(n.mass for n in self.get_nodes())

    # Angle-comparison tolerance for mechanism detection.
    _ANGLE_TOLERANCE: float = 1e-6
    # Number of decimal places for rounding angles in collinearity checks.
    _ANGLE_ROUND_DIGITS: int = 6

    def has_mechanism(self) -> bool:
        """Return ``True`` if any non-fixed node has all springs collinear.

        Such a node can move perpendicular to its springs without
        resistance, making the global stiffness matrix singular.
        """
        for nid in self.graph.nodes:
            if self.node_is_mechanism(nid):
                return True
        return False

    def would_create_mechanism(self, removed_id: int) -> bool:
        """Fast check: would removing *removed_id* leave any of its
        former neighbours in a mechanism state?

        Only checks neighbours of the removed node (already removed from
        graph) - O(degree) instead of O(N).
        """
        for nid in self.graph.nodes:
            if self.node_is_mechanism(nid):
                return True
        return False

    def node_is_mechanism(self, nid: int) -> bool:
        """Return ``True`` if *nid* has a mechanism mode.

        A node is a mechanism when it can move in some direction without
        any spring providing resistance - i.e. it has fewer than two
        springs, or all its springs are collinear and the perpendicular
        direction is not constrained by a boundary condition.
        """
        node: Node = self.graph.nodes[nid]["obj"]
        if node.fixed_x and node.fixed_z:
            return False
        springs = self.get_springs_for_node(nid)
        if len(springs) < 2:
            return True
        angles: set[float] = set()
        for sp in springs:
            a = sp.angle % math.pi
            angles.add(round(a, self._ANGLE_ROUND_DIGITS))
        if len(angles) <= 1:
            ref = springs[0].angle
            perp_is_x = abs(math.sin(ref)) < self._ANGLE_TOLERANCE
            perp_is_z = abs(math.cos(ref)) < self._ANGLE_TOLERANCE
            if perp_is_x and node.fixed_z:
                return False
            if perp_is_z and node.fixed_x:
                return False
            return True
        return False

    def get_protected_node_ids(self) -> set[int]:
        """IDs of nodes that must not be removed (supports / loads)."""
        return {n.id for n in self.get_nodes() if n.is_protected}

    def remove_dangling_nodes(self) -> int:
        """Remove dangling branches - appendages attached to the main
        structure through a single articulation point that contain no
        protected nodes (supports / loads).

        This catches both simple dead-ends (degree <= 1 chains) **and**
        branches whose tip forms a small loop (e.g.
        ``1-2-3-4-5-6-5``), which ordinary leaf-node removal misses.

        The algorithm:

        1. Find all *articulation points* (cut vertices).
        2. For each articulation point, temporarily remove it and
           inspect the resulting connected components.
        3. Any component that contains **no protected node** is a
           dangling branch - remove all its nodes.
        4. Repeat until stable (cascading).
        5. Final sweep: remove remaining degree-<=-1 non-protected
           leaf nodes iteratively.

        Returns
        -------
        int
            Total number of dangling nodes removed.
        """
        protected = self.get_protected_node_ids()
        total_removed = 0
        changed = True

        while changed:
            changed = False

            # Phase 1: dangling branches via articulation points
            if self.num_nodes < 2:
                break

            art_points = list(nx.articulation_points(self.graph))
            for ap in art_points:
                if ap not in self.graph:
                    continue

                # Back up the articulation-point node and its springs
                ap_obj = self.graph.nodes[ap]["obj"]
                ap_edges = [
                    (ap, nb, self.graph.edges[ap, nb]["obj"])
                    for nb in list(self.graph.neighbors(ap))
                ]
                neighbour_ids = {nb for _, nb, _ in ap_edges}

                # Temporarily remove ap
                self.graph.remove_node(ap)

                # Find which components are now cut off
                dangling_ids: set[int] = set()
                for comp in nx.connected_components(self.graph):
                    # Only look at components that contained a direct
                    # neighbour of ap (those are the ones that got split)
                    if not comp & neighbour_ids:
                        continue
                    # If the component has no protected node -> dangling
                    if not comp & protected:
                        dangling_ids |= comp

                # Restore the articulation point (it's part of the main body)
                self.graph.add_node(ap, obj=ap_obj)
                for u, v, sp in ap_edges:
                    # Only restore edges to nodes that were NOT dangling
                    if v in self.graph and v not in dangling_ids:
                        self.graph.add_edge(u, v, obj=sp)

                # Now remove all dangling nodes
                for nid in dangling_ids:
                    if nid in self.graph:
                        self.graph.remove_node(nid)
                        total_removed += 1
                        changed = True

            # Phase 2: simple leaf-node sweep (degree <= 1)
            leaf_changed = True
            while leaf_changed:
                leaf_changed = False
                for nid in list(self.graph.nodes):
                    if nid in protected:
                        continue
                    if self.graph.degree(nid) <= 1:
                        self.graph.remove_node(nid)
                        total_removed += 1
                        leaf_changed = True
                        changed = True

        return total_removed

    def supports_reachable_from_loads(self) -> bool:
        """Return ``True`` if every loaded node can reach at least one
        support node through the graph."""
        support_ids = {n.id for n in self.get_nodes() if n.is_fixed}
        load_ids = {n.id for n in self.get_nodes() if n.has_load}
        if not support_ids or not load_ids:
            return False  # nothing to check
        for lid in load_ids:
            # BFS / DFS from load node - does it reach any support?
            reachable = nx.node_connected_component(self.graph, lid)
            if not reachable & support_ids:
                return False
        return True

    # Deep copy (for history snapshots)
    def snapshot(self) -> "Structure":
        """Return an independent deep copy of the current state."""
        return copy.deepcopy(self)

    # Serialisation
    def to_dict(self) -> dict[str, Any]:
        """Serialise the full structure to a JSON-compatible dict."""
        nodes = [self.get_node(nid).to_dict() for nid in sorted(self.graph.nodes)]
        springs = [s.to_dict() for s in self.get_springs()]
        return {"nodes": nodes, "springs": springs}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Structure":
        """Reconstruct a :class:`Structure` from a serialised dict.

        Raises
        ------
        KeyError
            If required fields are missing.
        ValueError
            If the data references non-existent nodes.
        """
        if "nodes" not in data or "springs" not in data:
            raise KeyError(
                "Structure data must contain 'nodes' and 'springs' keys"
            )
        struct = cls()
        node_lookup: dict[int, Node] = {}
        for nd in data["nodes"]:
            node = Node.from_dict(nd)
            struct.graph.add_node(node.id, obj=node)
            node_lookup[node.id] = node
            if node.id >= struct._next_node_id:
                struct._next_node_id = node.id + 1
        for sd in data["springs"]:
            ni_id = sd.get("node_i")
            nj_id = sd.get("node_j")
            if ni_id not in node_lookup or nj_id not in node_lookup:
                logger.warning(
                    "Skipping spring (%s, %s): node not found", ni_id, nj_id
                )
                continue
            ni = node_lookup[ni_id]
            nj = node_lookup[nj_id]
            spring = Spring(ni, nj, k=sd.get("k"))
            struct.graph.add_edge(ni.id, nj.id, obj=spring)
        struct.renumber_dofs()
        logger.info(
            "Loaded structure: %d nodes, %d springs",
            struct.num_nodes, struct.graph.number_of_edges(),
        )
        return struct

    # Dunder
    def __repr__(self) -> str:
        return (
            f"Structure(nodes={self.num_nodes}, "
            f"springs={self.graph.number_of_edges()})"
        )