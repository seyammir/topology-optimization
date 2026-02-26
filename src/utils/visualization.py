"""Matplotlib-based visualisation helpers for the mass-spring structure."""

from __future__ import annotations

import io
import logging

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from PIL import Image

from ..models.structure import Structure
from ..solver.optimizer_base import OptimizationResult

logger = logging.getLogger(__name__)


class Visualizer:
    """Static helper methods for plotting structures."""

    # Colour palette
    COLOR_NODE = "#2563eb"
    COLOR_SPRING = "#94a3b8"
    COLOR_SUPPORT = "#dc2626"
    COLOR_LOAD = "#16a34a"
    COLOR_DEFORMED = "#f97316"
    COLOR_HEATMAP_LOW = "#3b82f6"
    COLOR_HEATMAP_HIGH = "#ef4444"

    # Scale factor used for auto-scaling deformed plots.
    AUTO_SCALE_FRACTION: float = 0.05
    # Threshold below which displacements are treated as zero.
    DISPLACEMENT_EPSILON: float = 1e-12
    # Small epsilon to avoid division by zero in force normalisation.
    FORCE_EPSILON: float = 1e-12

    # Main structure plot
    @classmethod
    def plot_structure(
        cls,
        structure: Structure,
        u: np.ndarray | None = None,
        scale: float = 10.0,
        title: str = "Structure",
        show_node_ids: bool = False,
        ax: plt.Axes | None = None,
        densities: dict[tuple[int, int], float] | None = None,
    ) -> Figure:
        """Draw the structure (optionally deformed).

        Parameters
        ----------
        structure : Structure
        u : ndarray, optional
            Global displacement vector; if given the deformed shape is
            overlaid.
        scale : float
            Amplification factor for deformed displacements.
        title : str
        show_node_ids : bool
        ax : Axes, optional
            If *None* a new figure is created.
        densities : dict, optional
            Per-spring SIMP density values.  When provided, springs
            with density below a threshold are hidden and remaining
            springs are drawn with thickness proportional to density.

        Returns
        -------
        Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        nodes = structure.get_nodes()
        springs = structure.get_springs()

        # auto-scale deformation if requested
        if u is not None and scale <= 0:
            max_disp = float(np.max(np.abs(u))) if len(u) > 0 else 0.0
            all_x = [n.x for n in nodes]
            all_z = [n.z for n in nodes]
            span = max(
                (max(all_x) - min(all_x)) if all_x else 1.0,
                (max(all_z) - min(all_z)) if all_z else 1.0,
            )
            scale = (
                cls.AUTO_SCALE_FRACTION * span / max_disp
                if max_disp > cls.DISPLACEMENT_EPSILON
                else 1.0
            )

        # Helper: look up density for a spring
        _DENS_THRESHOLD = 0.01  # springs below this are invisible
        def _get_density(sp):
            if densities is None:
                return 1.0
            key = sp.node_ids
            rkey = (key[1], key[0])
            return densities.get(key, densities.get(rkey, 1.0))

        # undeformed springs (light grey, thin)
        for sp in springs:
            xe = _get_density(sp)
            if xe < _DENS_THRESHOLD:
                continue
            xi, zi = sp.node_i.x, sp.node_i.z
            xj, zj = sp.node_j.x, sp.node_j.z
            lw = 0.5 if densities is None else 0.3 + 2.0 * xe
            alpha = 1.0 if densities is None else max(0.15, xe)
            ax.plot([xi, xj], [zi, zj], color=cls.COLOR_SPRING, lw=lw,
                    alpha=alpha, zorder=1)

        # deformed springs (orange, solid, prominent)
        if u is not None:
            structure.renumber_dofs()
            for sp in springs:
                xe = _get_density(sp)
                if xe < _DENS_THRESHOLD:
                    continue
                dofs = sp.dof_indices
                xi = sp.node_i.x + scale * u[dofs[0]]
                zi = sp.node_i.z + scale * u[dofs[1]]
                xj = sp.node_j.x + scale * u[dofs[2]]
                zj = sp.node_j.z + scale * u[dofs[3]]
                lw = 1.2 if densities is None else 0.3 + 2.5 * xe
                alpha = 1.0 if densities is None else max(0.15, xe)
                ax.plot(
                    [xi, xj], [zi, zj],
                    color=cls.COLOR_DEFORMED, lw=lw, alpha=alpha, zorder=2,
                )

        # Determine which nodes belong to active springs
        if densities is not None:
            _active_node_ids = set()
            for sp in springs:
                if _get_density(sp) >= _DENS_THRESHOLD:
                    ni, nj = sp.node_ids
                    _active_node_ids.add(ni)
                    _active_node_ids.add(nj)
            active_nodes = [n for n in nodes if n.id in _active_node_ids]
        else:
            active_nodes = nodes

        # deformed nodes
        if u is not None:
            structure.renumber_dofs()
            def_xs = [n.x + scale * u[n.dof_indices[0]] for n in active_nodes]
            def_zs = [n.z + scale * u[n.dof_indices[1]] for n in active_nodes]
            ax.scatter(
                def_xs, def_zs, s=14,
                c=cls.COLOR_DEFORMED, edgecolors="none",
                alpha=0.7, zorder=4, label="Deformed",
            )

        # nodes (undeformed)
        xs = [n.x for n in active_nodes]
        zs = [n.z for n in active_nodes]
        ax.scatter(xs, zs, s=12, c=cls.COLOR_NODE, zorder=3, label="Original")

        # supports (triangles)
        support_plotted = False
        for n in nodes:
            if n.is_fixed:
                marker = "^" if n.is_pinned else (">" if n.fixed_z else "v")
                label = "Support" if not support_plotted else None
                ax.plot(
                    n.x, n.z, marker=marker, color=cls.COLOR_SUPPORT,
                    markersize=10, zorder=4, linestyle="None", label=label,
                )
                support_plotted = True

        # external forces (arrows)
        force_plotted = False
        for n in nodes:
            if n.has_load:
                # Normalise arrow length for visibility
                mag = max(abs(n.fx), abs(n.fz), cls.FORCE_EPSILON)
                dx = n.fx / mag * 0.5
                dz = n.fz / mag * 0.5
                ax.annotate(
                    "", xy=(n.x + dx, n.z + dz), xytext=(n.x, n.z),
                    arrowprops=dict(arrowstyle="->", color=cls.COLOR_LOAD, lw=2),
                    zorder=5,
                )
                if not force_plotted:
                    # Invisible marker just for the legend entry
                    ax.plot(
                        [], [], color=cls.COLOR_LOAD, marker=r"$\rightarrow$",
                        markersize=10, linestyle="None", label="Applied Force",
                    )
                    force_plotted = True

        # node ids
        if show_node_ids:
            for n in nodes:
                ax.text(n.x, n.z, str(n.id), fontsize=5, ha="center", va="bottom")

        ax.set_aspect("equal")
        ax.invert_yaxis()  # z positive downward
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("z")

        # Info text
        active_spring_count = sum(1 for sp in springs if _get_density(sp) >= _DENS_THRESHOLD)
        info = f"Nodes: {len(active_nodes)}  |  Springs: {active_spring_count}"
        ax.text(
            0.01, 0.01, info,
            transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom", color="#555555",
            zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="#cccccc"),
        )

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles, labels, loc="upper right", fontsize=7,
                framealpha=0.85, edgecolor="#cccccc", fancybox=True,
            )

        fig.tight_layout()
        return fig

    # Heatmap
    @classmethod
    def plot_energy_heatmap(
        cls,
        structure: Structure,
        node_energies: dict[int, float],
        title: str = "Strain Energy Heatmap",
        ax: plt.Axes | None = None,
        densities: dict[tuple[int, int], float] | None = None,
    ) -> Figure:
        """Colour springs / nodes by strain energy.

        Parameters
        ----------
        structure : Structure
        node_energies : dict[int, float]
            Per-node energy values.
        title : str
        ax : Axes, optional
        densities : dict, optional
            Per-spring SIMP densities.  When given, springs below
            threshold are hidden.

        Returns
        -------
        Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        springs = structure.get_springs()
        nodes = structure.get_nodes()

        # Density helper
        _DENS_THRESHOLD = 0.01
        def _spring_density(sp):
            if densities is None:
                return 1.0
            key = sp.node_ids
            rkey = (key[1], key[0])
            return densities.get(key, densities.get(rkey, 1.0))

        # Filter springs by density
        active_springs = [sp for sp in springs if _spring_density(sp) >= _DENS_THRESHOLD]

        # Active nodes
        if densities is not None:
            active_ids = set()
            for sp in active_springs:
                ni, nj = sp.node_ids
                active_ids.add(ni)
                active_ids.add(nj)
            active_nodes = [n for n in nodes if n.id in active_ids]
        else:
            active_nodes = nodes

        # Spring energy = average of the two end-node energies.
        if not node_energies:
            return fig

        # Compute min/max only over active nodes
        active_energies = {n.id: node_energies.get(n.id, 0.0) for n in active_nodes}
        if active_energies:
            e_min = min(active_energies.values())
            e_max = max(active_energies.values())
        else:
            e_min, e_max = 0.0, 1.0
        if e_max == e_min:
            e_max = e_min + 1.0

        cmap = plt.cm.RdYlBu_r

        def _normed(val: float) -> float:
            return (val - e_min) / (e_max - e_min)

        for sp in active_springs:
            ni, nj = sp.node_ids
            ei = node_energies.get(ni, 0.0)
            ej = node_energies.get(nj, 0.0)
            avg = (ei + ej) / 2.0
            colour = cmap(_normed(avg))
            lw = 1.5 if densities is None else 0.5 + 2.5 * _spring_density(sp)
            ax.plot(
                [sp.node_i.x, sp.node_j.x],
                [sp.node_i.z, sp.node_j.z],
                color=colour, lw=lw, zorder=1,
            )

        # Nodes
        xs = [n.x for n in active_nodes]
        zs = [n.z for n in active_nodes]
        cs = [_normed(node_energies.get(n.id, 0.0)) for n in active_nodes]
        sc = ax.scatter(xs, zs, c=cs, cmap=cmap, s=18, zorder=3, edgecolors="k", linewidths=0.3)
        cbar = fig.colorbar(sc, ax=ax, label="Strain Energy")
        cbar.ax.tick_params(labelsize=8)

        # Min / max energy annotation
        ax.text(
            0.01, 0.01,
            f"Min: {e_min:.2e}  |  Max: {e_max:.2e}  |  Nodes: {len(active_nodes)}",
            transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom", color="#555555",
            zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="#cccccc"),
        )

        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        fig.tight_layout()
        return fig

    # Black & white density plot
    @classmethod
    def plot_bw_density(
        cls,
        structure: Structure,
        initial_structure: Structure | None = None,
        title: str = "Topology (B/W)",
        ax: plt.Axes | None = None,
    ) -> Figure:
        """Render a black-and-white density image of the topology.

        Each cell of the original rectangular grid is coloured from
        white (void) to black (solid) based on how many of its four
        corner nodes still exist in *structure*.  The result looks like
        the classic SIMP density plot.

        Parameters
        ----------
        structure : Structure
            The (optimised) structure.
        initial_structure : Structure, optional
            The original full-grid structure used to determine the grid
            dimensions.  If *None*, dimensions are inferred from the
            node coordinates present in *structure*.
        title : str
        ax : Axes, optional

        Returns
        -------
        Figure
        """
        nodes = structure.get_nodes()
        all_xs = {n.x for n in nodes}
        all_zs = {n.z for n in nodes}

        # Determine full grid dimensions.
        if initial_structure is not None:
            ref_nodes = initial_structure.get_nodes()
            ref_xs = {n.x for n in ref_nodes}
            ref_zs = {n.z for n in ref_nodes}
        else:
            ref_xs = all_xs
            ref_zs = all_zs

        xs_sorted = sorted(ref_xs)
        zs_sorted = sorted(ref_zs)

        if len(xs_sorted) < 2 or len(zs_sorted) < 2:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure
            ax.text(0.5, 0.5, "Grid too small", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        # Build a set of existing node positions for fast look-up.
        present = {(n.x, n.z) for n in nodes}

        n_cells_x = len(xs_sorted) - 1
        n_cells_z = len(zs_sorted) - 1
        density = np.zeros((n_cells_z, n_cells_x))

        for iz in range(n_cells_z):
            for ix in range(n_cells_x):
                corners = [
                    (xs_sorted[ix],     zs_sorted[iz]),
                    (xs_sorted[ix + 1], zs_sorted[iz]),
                    (xs_sorted[ix],     zs_sorted[iz + 1]),
                    (xs_sorted[ix + 1], zs_sorted[iz + 1]),
                ]
                density[iz, ix] = sum(1 for c in corners if c in present) / 4.0

        if ax is None:
            # Aspect ratio that matches the grid
            aspect = n_cells_x / max(n_cells_z, 1)
            fig_w = min(12, max(6, aspect * 4))
            fig_h = fig_w / max(aspect, 0.3)
            fig, ax = plt.subplots(figsize=(fig_w, min(fig_h, 8)))
        else:
            fig = ax.figure

        ax.imshow(
            density,
            cmap="gray_r",
            vmin=0,
            vmax=1,
            aspect="equal",
            interpolation="none",
            origin="upper",
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")
        fig.tight_layout()
        return fig

    # B/W density from SIMP spring densities
    @classmethod
    def plot_bw_density_from_springs(
        cls,
        structure: Structure,
        densities: dict[tuple[int, int], float],
        initial_structure: Structure | None = None,
        title: str = "Topology (B/W) - SIMP",
        ax: plt.Axes | None = None,
    ) -> Figure:
        """Render a B/W density image derived from SIMP spring densities.

        For each rectangular cell in the grid, the density is computed
        as the average density of the springs that form its edges.
        This gives a classic SIMP-style black/white density plot.

        Parameters
        ----------
        structure : Structure
        densities : dict[(ni, nj), float]
            Per-spring density values in [0, 1].
        initial_structure : Structure, optional
        title : str
        ax : Axes, optional

        Returns
        -------
        Figure
        """
        nodes = structure.get_nodes()
        ref = initial_structure if initial_structure is not None else structure

        ref_nodes = ref.get_nodes()
        xs_sorted = sorted({n.x for n in ref_nodes})
        zs_sorted = sorted({n.z for n in ref_nodes})

        if len(xs_sorted) < 2 or len(zs_sorted) < 2:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure
            ax.text(0.5, 0.5, "Grid too small", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        # Build a lookup: node position -> node id
        pos_to_id: dict[tuple[float, float], int] = {}
        for n in nodes:
            pos_to_id[(n.x, n.z)] = n.id

        # Build a fast density lookup that handles both key orderings
        dens_lookup: dict[tuple[int, int], float] = {}
        for k, v in densities.items():
            dens_lookup[k] = v
            dens_lookup[(k[1], k[0])] = v

        n_cells_x = len(xs_sorted) - 1
        n_cells_z = len(zs_sorted) - 1

        # Pre-compute the full node-id grid for fast edge lookups.
        # id_grid[iz][ix] is the node id at (xs_sorted[ix], zs_sorted[iz]).
        xs_arr = np.array(xs_sorted)
        zs_arr = np.array(zs_sorted)
        id_grid = np.full((len(zs_arr), len(xs_arr)), -1, dtype=np.intp)
        x_idx = {x: i for i, x in enumerate(xs_sorted)}
        z_idx = {z: i for i, z in enumerate(zs_sorted)}
        for n in nodes:
            ix_n = x_idx.get(n.x)
            iz_n = z_idx.get(n.z)
            if ix_n is not None and iz_n is not None:
                id_grid[iz_n, ix_n] = n.id

        # Vectorised density image: accumulate edge densities per cell
        density_sum = np.zeros((n_cells_z, n_cells_x))
        density_cnt = np.zeros((n_cells_z, n_cells_x), dtype=np.int32)

        def _add_edge(ni_arr: np.ndarray, nj_arr: np.ndarray) -> None:
            """Add density contributions for a whole grid of edges."""
            valid = (ni_arr >= 0) & (nj_arr >= 0)
            izs, ixs = np.nonzero(valid)
            for iz, ix in zip(izs, ixs):
                ni, nj = int(ni_arr[iz, ix]), int(nj_arr[iz, ix])
                d = dens_lookup.get((ni, nj))
                if d is not None:
                    density_sum[iz, ix] += d
                    density_cnt[iz, ix] += 1

        # Corner node-id sub-grids (shape = n_cells_z × n_cells_x)
        tl = id_grid[:-1, :-1]   # top-left
        tr = id_grid[:-1, 1:]    # top-right
        bl = id_grid[1:, :-1]    # bottom-left
        br = id_grid[1:, 1:]     # bottom-right

        _add_edge(tl, tr)  # top horizontal
        _add_edge(bl, br)  # bottom horizontal
        _add_edge(tl, bl)  # left vertical
        _add_edge(tr, br)  # right vertical

        with np.errstate(invalid="ignore"):
            density_img = np.where(
                density_cnt > 0, density_sum / density_cnt, 0.0
            )

        if ax is None:
            aspect = n_cells_x / max(n_cells_z, 1)
            fig_w = min(12, max(6, aspect * 4))
            fig_h = fig_w / max(aspect, 0.3)
            fig, ax = plt.subplots(figsize=(fig_w, min(fig_h, 8)))
        else:
            fig = ax.figure

        ax.imshow(
            density_img,
            cmap="gray_r",
            vmin=0,
            vmax=1,
            aspect="equal",
            interpolation="none",
            origin="upper",
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")
        fig.tight_layout()
        return fig

    # Internal forces
    @classmethod
    def plot_internal_forces(
        cls,
        structure: Structure,
        spring_forces: dict[tuple[int, int], dict],
        title: str = "Internal Forces",
        ax: plt.Axes | None = None,
        densities: dict[tuple[int, int], float] | None = None,
    ) -> Figure:
        """Visualise internal axial forces in the structure.

        Springs are coloured and sized by their internal axial force:
        **red** = tension, **blue** = compression.

        Parameters
        ----------
        structure : Structure
        spring_forces : dict
            Per-spring internal forces (from
            :meth:`FEMSolver.compute_internal_forces`).
        title : str
        ax : Axes, optional
        densities : dict, optional
            Per-spring SIMP densities.  When given, springs below
            threshold are hidden.

        Returns
        -------
        Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7))
        else:
            fig = ax.figure

        nodes = structure.get_nodes()
        springs = structure.get_springs()

        # Density helper
        _DENS_THRESHOLD = 0.01
        def _key_density(ni, nj):
            if densities is None:
                return 1.0
            return densities.get((ni, nj), densities.get((nj, ni), 1.0))

        # Filter spring_forces by density
        if densities is not None:
            spring_forces = {
                k: v for k, v in spring_forces.items()
                if _key_density(k[0], k[1]) >= _DENS_THRESHOLD
            }

        # Active nodes (only those connected to visible springs)
        if densities is not None:
            active_ids = set()
            for (ni, nj) in spring_forces:
                active_ids.add(ni)
                active_ids.add(nj)
            active_nodes = [n for n in nodes if n.id in active_ids]
        else:
            active_nodes = nodes

        # springs coloured & sized by internal force
        f_max = 1.0
        if spring_forces:
            abs_forces = [info["abs_force"] for info in spring_forces.values()]
            f_max = max(abs_forces) if abs_forces else 1.0
            if f_max < cls.FORCE_EPSILON:
                f_max = 1.0

            # build segments + colours for LineCollection
            segments = []
            colours = []
            widths = []
            for (ni, nj), info in spring_forces.items():
                xi, zi = info["node_i"]
                xj, zj = info["node_j"]
                segments.append([(xi, zi), (xj, zj)])

                axial = info["axial_force"]
                # map: -1 (compression/blue) -> 0, 0 (neutral) -> 0.5, +1 (tension/red) -> 1
                norm_val = 0.5 + 0.5 * np.clip(axial / f_max, -1, 1)
                colours.append(norm_val)

                # Width proportional to |force|, min 0.3, max 4.5
                w = 0.3 + 4.2 * (info["abs_force"] / f_max)
                widths.append(w)

            cmap = plt.cm.coolwarm  # blue (compression) <-> red (tension)
            lc = LineCollection(
                segments,
                array=np.array(colours),
                cmap=cmap,
                linewidths=widths,
                clim=(0, 1),
                zorder=1,
            )
            ax.add_collection(lc)

            # colourbar
            sm = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=mcolors.Normalize(vmin=-f_max, vmax=f_max),
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, label="Axial Force  (- compress. / + tension)")
            cbar.ax.tick_params(labelsize=8)

        # supports & loads markers
        support_plotted = False
        for n in active_nodes:
            if n.is_fixed:
                marker = "^" if n.is_pinned else (">" if n.fixed_z else "v")
                label = "Support" if not support_plotted else None
                ax.plot(
                    n.x, n.z, marker=marker, color=cls.COLOR_SUPPORT,
                    markersize=11, zorder=6, linestyle="None", label=label,
                )
                support_plotted = True

        force_plotted = False
        for n in nodes:
            if n.has_load:
                mag = max(abs(n.fx), abs(n.fz), cls.FORCE_EPSILON)
                dx = n.fx / mag * 0.7
                dz = n.fz / mag * 0.7
                ax.annotate(
                    "",
                    xy=(n.x + dx, n.z + dz),
                    xytext=(n.x, n.z),
                    arrowprops=dict(arrowstyle="->,head_width=0.35", color=cls.COLOR_LOAD, lw=2.5),
                    zorder=7,
                )
                if not force_plotted:
                    ax.plot(
                        [], [], color=cls.COLOR_LOAD, marker=r"$\rightarrow$",
                        markersize=12, linestyle="None", label="Applied Force",
                    )
                    force_plotted = True

        # legend
        from matplotlib.lines import Line2D
        legend_elements = []
        legend_elements.append(Line2D([0], [0], color="#b91c1c", lw=2, label="Tension"))
        legend_elements.append(Line2D([0], [0], color="#1d4ed8", lw=2, label="Compression"))
        if support_plotted:
            legend_elements.append(Line2D(
                [0], [0], marker="^", color=cls.COLOR_SUPPORT,
                linestyle="None", markersize=9, label="Support",
            ))
        if force_plotted:
            legend_elements.append(Line2D(
                [0], [0], marker=r"$\rightarrow$", color=cls.COLOR_LOAD,
                linestyle="None", markersize=10, label="Applied Force",
            ))

        # axes configuration
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.invert_yaxis()
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("z")

        ax.legend(
            handles=legend_elements, loc="upper right", fontsize=7,
            framealpha=0.85, edgecolor="#cccccc", fancybox=True,
        )

        fig.tight_layout()
        return fig

    # Export helpers
    @staticmethod
    def fig_to_png_bytes(fig: Figure) -> bytes:
        """Render a matplotlib figure to PNG bytes (for download).

        Raises
        ------
        RuntimeError
            If the figure cannot be rendered.
        """
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            return buf.read()
        except Exception:
            logger.exception("Failed to render figure to PNG")
            raise RuntimeError("Could not render the figure to PNG") from None

    @classmethod
    def create_animation_gif(
        cls,
        history: list[Structure],
        initial_structure: Structure | None = None,
        mode: str = "bw",
        duration_ms: int = 300,
        loop: int = 0,
    ) -> bytes:
        """Create an animated GIF from a list of structure snapshots.

        Parameters
        ----------
        history : list[Structure]
            Ordered list of structure snapshots (e.g. from
            ``OptimizationResult.history``).
        initial_structure : Structure, optional
            The original grid structure, used as reference for the B/W
            density plot.
        mode : str
            ``"bw"`` for black-and-white density frames,
            ``"structure"`` for wireframe structure frames.
        duration_ms : int
            Duration of each frame in milliseconds.
        loop : int
            Number of times to loop (0 = infinite).

        Returns
        -------
        bytes
            The GIF file content.

        Raises
        ------
        ValueError
            If *history* is empty.
        RuntimeError
            If the GIF cannot be rendered.
        """
        if not history:
            raise ValueError("history must contain at least one snapshot")

        ref = initial_structure if initial_structure is not None else history[0]
        frames: list[Image.Image] = []

        # Cap total frames to avoid excessive rendering time
        max_frames = 60
        total = len(history)
        if total > max_frames:
            step = total / max_frames
            indices = [int(i * step) for i in range(max_frames)]
            if indices[-1] != total - 1:
                indices.append(total - 1)
        else:
            indices = list(range(total))

        try:
            # Reuse a single figure/axes across all frames
            reuse_fig, reuse_ax = plt.subplots(figsize=(10, 6))

            for frame_num, idx in enumerate(indices):
                reuse_ax.clear()
                snap = history[idx]
                if mode == "bw":
                    fig = cls.plot_bw_density(
                        snap,
                        initial_structure=ref,
                        title=f"Iteration {idx}",
                        ax=reuse_ax,
                    )
                else:
                    fig = cls.plot_structure(
                        snap,
                        title=f"Iteration {idx}",
                        ax=reuse_ax,
                    )

                # Render figure to PIL Image — lower dpi and skip
                # bbox_inches='tight' (avoids costly trial render).
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=72)
                buf.seek(0)
                img = Image.open(buf).convert("RGBA")
                # Composite onto white background for GIF compatibility
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                background.paste(img, mask=img)
                frames.append(background.convert("RGB"))

            plt.close(reuse_fig)

            # Hold the last frame longer
            durations = [duration_ms] * len(frames)
            if len(durations) > 1:
                durations[-1] = duration_ms * 4

            out = io.BytesIO()
            frames[0].save(
                out,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=loop,
            )
            out.seek(0)
            logger.info("Created animation GIF with %d frames", len(frames))
            return out.read()
        except Exception:
            logger.exception("Failed to create animation GIF")
            raise RuntimeError("Could not create the animation GIF") from None

    @classmethod
    def create_simp_animation_gif(
        cls,
        structure: Structure,
        density_history: list[dict[tuple[int, int], float]],
        initial_structure: Structure | None = None,
        mode: str = "bw",
        duration_ms: int = 300,
        loop: int = 0,
    ) -> bytes:
        """Create an animated GIF from SIMP density evolution.

        Parameters
        ----------
        structure : Structure
            The structure (unchanged throughout SIMP).
        density_history : list[dict]
            Per-iteration density snapshots.
        initial_structure : Structure, optional
            The original grid structure for reference dimensions.
        mode : str
            ``"bw"`` for black-and-white density frames,
            ``"structure"`` for wireframe with density-based thickness.
        duration_ms : int
            Duration of each frame in milliseconds.
        loop : int
            Number of times to loop (0 = infinite).

        Returns
        -------
        bytes
            The GIF file content.

        Raises
        ------
        ValueError
            If *density_history* is empty.
        RuntimeError
            If the GIF cannot be rendered.
        """
        if not density_history:
            raise ValueError("density_history must contain at least one snapshot")

        ref = initial_structure if initial_structure is not None else structure
        frames: list[Image.Image] = []

        # Sample frames if there are too many iterations (cap at ~60 frames)
        max_frames = 60
        total = len(density_history)
        if total > max_frames:
            step = total / max_frames
            indices = [int(i * step) for i in range(max_frames)]
            # Always include the last frame
            if indices[-1] != total - 1:
                indices.append(total - 1)
        else:
            indices = list(range(total))

        try:
            # Reuse a single figure/axes across all frames to avoid
            # repeated figure-creation overhead.
            reuse_fig, reuse_ax = plt.subplots(figsize=(10, 6))

            for frame_num, idx in enumerate(indices):
                reuse_ax.clear()
                dens = density_history[idx]
                iteration = idx + 1
                if mode == "bw":
                    fig = cls.plot_bw_density_from_springs(
                        structure,
                        dens,
                        initial_structure=ref,
                        title=f"Iteration {iteration}",
                        ax=reuse_ax,
                    )
                else:
                    fig = cls.plot_structure(
                        structure,
                        title=f"Iteration {iteration}",
                        densities=dens,
                        ax=reuse_ax,
                    )

                # Render figure to PIL Image — use lower dpi and skip
                # bbox_inches='tight' (requires a costly trial render).
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=72)
                buf.seek(0)
                img = Image.open(buf).convert("RGBA")
                # Composite onto white background for GIF compatibility
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                background.paste(img, mask=img)
                frames.append(background.convert("RGB"))

            plt.close(reuse_fig)

            # Hold the last frame longer
            durations = [duration_ms] * len(frames)
            if len(durations) > 1:
                durations[-1] = duration_ms * 4

            out = io.BytesIO()
            frames[0].save(
                out,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=loop,
            )
            out.seek(0)
            logger.info("Created SIMP animation GIF with %d frames", len(frames))
            return out.read()
        except Exception:
            logger.exception("Failed to create SIMP animation GIF")
            raise RuntimeError("Could not create the SIMP animation GIF") from None

    @staticmethod
    def export_image(fig: Figure, path: str) -> None:
        """Save figure to disk.

        Raises
        ------
        OSError
            If the file cannot be written.
        """
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Exported image to %s", path)
        except OSError:
            logger.exception("Failed to export image to %s", path)
            raise

    # SIMP density-field visualisation

    @classmethod
    def plot_density_field(
        cls,
        structure: Structure,
        densities: dict[tuple[int, int], float],
        title: str = "Density Field (SIMP)",
        ax: plt.Axes | None = None,
    ) -> Figure:
        """Render springs coloured/thickened by their SIMP density.

        Parameters
        ----------
        structure : Structure
        densities : dict[(ni, nj), float]
            Per-spring density values in [0, 1].
        title : str
        ax : Axes, optional

        Returns
        -------
        Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        springs = structure.get_springs()
        nodes = structure.get_nodes()

        if not springs:
            ax.text(0.5, 0.5, "No springs", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        segments = []
        vals = []
        for sp in springs:
            key = sp.node_ids
            rkey = (key[1], key[0])
            xe = densities.get(key, densities.get(rkey, 1.0))
            segments.append([(sp.node_i.x, sp.node_i.z),
                             (sp.node_j.x, sp.node_j.z)])
            vals.append(xe)

        vals_arr = np.array(vals)
        # Width proportional to density (thin = void, thick = solid)
        widths = 0.3 + 3.0 * vals_arr
        cmap = plt.cm.gray_r

        lc = LineCollection(
            segments,
            array=vals_arr,
            cmap=cmap,
            linewidths=widths,
            clim=(0, 1),
            zorder=1,
        )
        ax.add_collection(lc)

        # Colourbar
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=mcolors.Normalize(vmin=0, vmax=1),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label="Density x_e")
        cbar.ax.tick_params(labelsize=8)

        # Supports & forces (reuse same markers as plot_structure)
        support_plotted = False
        for n in nodes:
            if n.is_fixed:
                marker = "^" if n.is_pinned else (">" if n.fixed_z else "v")
                label = "Support" if not support_plotted else None
                ax.plot(n.x, n.z, marker=marker, color=cls.COLOR_SUPPORT,
                        markersize=10, zorder=4, linestyle="None", label=label)
                support_plotted = True

        force_plotted = False
        for n in nodes:
            if n.has_load:
                mag = max(abs(n.fx), abs(n.fz), cls.FORCE_EPSILON)
                dx = n.fx / mag * 0.5
                dz = n.fz / mag * 0.5
                ax.annotate("", xy=(n.x + dx, n.z + dz), xytext=(n.x, n.z),
                            arrowprops=dict(arrowstyle="->", color=cls.COLOR_LOAD, lw=2),
                            zorder=5)
                if not force_plotted:
                    ax.plot([], [], color=cls.COLOR_LOAD, marker=r"$\rightarrow$",
                            markersize=10, linestyle="None", label="Applied Force")
                    force_plotted = True

        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.invert_yaxis()
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("z")

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper right", fontsize=7,
                      framealpha=0.85, edgecolor="#cccccc", fancybox=True)

        info = f"Springs: {len(springs)}"
        ax.text(0.01, 0.01, info, transform=ax.transAxes, fontsize=8,
                verticalalignment="bottom", color="#555555",
                zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.7, edgecolor="#cccccc"))

        fig.tight_layout()
        return fig

    # Comparison panel

    @classmethod
    def plot_comparison_structures(
        cls,
        results: dict[str, OptimizationResult],
    ) -> Figure:
        """Side-by-side final node-spring structure for each algorithm.

        Parameters
        ----------
        results : dict[str, OptimizationResult]
            Algorithm name -> result.

        Returns
        -------
        Figure
        """
        n = len(results)
        if n == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No results", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        fig, axes = plt.subplots(
            1, n, figsize=(7 * n, 5),
            constrained_layout=True,
        )
        if n == 1:
            axes = [axes]

        # Collect axis limits across all results so both panels
        # show the same coordinate range (avoids misleading whitespace).
        all_x: list[float] = []
        all_z: list[float] = []
        for res in results.values():
            struct = res.history[-1] if res.history else None
            if struct is None:
                continue
            for nd in struct.get_nodes():
                all_x.append(nd.x)
                all_z.append(nd.z)

        if all_x and all_z:
            pad_x = max(1.0, (max(all_x) - min(all_x)) * 0.04)
            pad_z = max(1.0, (max(all_z) - min(all_z)) * 0.04)
            shared_xlim = (min(all_x) - pad_x, max(all_x) + pad_x)
            shared_zlim = (min(all_z) - pad_z, max(all_z) + pad_z)
        else:
            shared_xlim = shared_zlim = None

        for ax, (name, res) in zip(axes, results.items()):
            struct = res.history[-1] if res.history else None
            if struct is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            cls.plot_structure(
                struct, title=name, ax=ax,
                densities=res.densities,
            )
            # Apply shared limits so both panels match.
            # ylim is inverted (z positive downward) so pass (max, min).
            if shared_xlim is not None:
                ax.set_xlim(shared_xlim)
                ax.set_ylim(shared_zlim[1], shared_zlim[0])

        fig.suptitle("Algorithm Comparison",
                     fontsize=15, fontweight="bold")
        return fig

    @classmethod
    def plot_comparison(
        cls,
        results: dict[str, OptimizationResult],
        initial_structure: Structure | None = None,
    ) -> Figure:
        """Side-by-side final topology for each algorithm.

        Parameters
        ----------
        results : dict[str, OptimizationResult]
            Algorithm name -> result.
        initial_structure : Structure, optional
            Used as reference for B/W density plots.

        Returns
        -------
        Figure
        """
        n = len(results)
        if n == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No results", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
        if n == 1:
            axes = [axes]

        # Determine best compliance for highlighting
        compliances = {}
        for name, res in results.items():
            c = res.compliance_history[-1] if res.compliance_history else float("inf")
            compliances[name] = c
        best_algo = min(compliances, key=compliances.get) if compliances else None

        for ax, (name, res) in zip(axes, results.items()):
            if res.densities is not None:
                # SIMP - use density field on the last-history structure
                struct = res.history[-1] if res.history else None
                if struct is not None:
                    cls.plot_density_field(struct, res.densities,
                                          title=name, ax=ax)
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes)
            else:
                # Discrete methods - use B/W density
                struct = res.history[-1] if res.history else None
                if struct is not None:
                    cls.plot_bw_density(struct, initial_structure=initial_structure,
                                       title=name, ax=ax)
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes)

            # Highlight the winner with a coloured border
            is_best = (name == best_algo and n > 1)
            border_color = "#16a34a" if is_best else "#cccccc"
            border_width = 3 if is_best else 1
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)

            # Metrics annotation
            c_final = compliances.get(name, 0.0)
            winner_tag = "  ★ Best" if is_best else ""
            ax.text(
                0.5, -0.02,
                f"Compliance: {c_final:.4g}  |  "
                f"Iterations: {res.iterations}{winner_tag}",
                transform=ax.transAxes, fontsize=9, ha="center",
                va="top", color="#333",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#d4edda" if is_best else "#f0f0f0",
                          alpha=0.9, edgecolor="#aaa"),
            )

        fig.suptitle("Algorithm Comparison", fontsize=15, fontweight="bold", y=1.02)
        fig.tight_layout()
        return fig

    @classmethod
    def plot_compliance_comparison(
        cls,
        results: dict[str, OptimizationResult],
    ) -> Figure:
        """Overlay compliance-history curves for multiple algorithms.

        Parameters
        ----------
        results : dict[str, OptimizationResult]

        Returns
        -------
        Figure
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#8b5cf6"]

        for idx, (name, res) in enumerate(results.items()):
            ch = res.compliance_history
            if not ch:
                continue
            iters = list(range(1, len(ch) + 1))
            color = colors[idx % len(colors)]
            ax.plot(iters, ch, color=color, lw=2, marker="o", markersize=3,
                    label=name)
            ax.fill_between(iters, ch, alpha=0.06, color=color)

            # Annotate final value
            ax.annotate(
                f"{ch[-1]:.4g}",
                xy=(len(ch), ch[-1]),
                xytext=(8, 0), textcoords="offset points",
                fontsize=8, color=color, fontweight="bold", va="center",
            )

        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("Compliance (total strain energy)", fontsize=10)
        ax.set_title("Compliance Convergence — Algorithm Comparison",
                     fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=10, framealpha=0.85, loc="best")
        fig.tight_layout()
        return fig