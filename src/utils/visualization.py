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

        # undeformed springs (light grey, thin)
        for sp in springs:
            xi, zi = sp.node_i.x, sp.node_i.z
            xj, zj = sp.node_j.x, sp.node_j.z
            ax.plot([xi, xj], [zi, zj], color=cls.COLOR_SPRING, lw=0.5, zorder=1)

        # deformed springs (orange, solid, prominent)
        if u is not None:
            structure.renumber_dofs()
            for sp in springs:
                dofs = sp.dof_indices
                xi = sp.node_i.x + scale * u[dofs[0]]
                zi = sp.node_i.z + scale * u[dofs[1]]
                xj = sp.node_j.x + scale * u[dofs[2]]
                zj = sp.node_j.z + scale * u[dofs[3]]
                ax.plot(
                    [xi, xj], [zi, zj],
                    color=cls.COLOR_DEFORMED, lw=1.2, zorder=2,
                )

        # deformed nodes
        if u is not None:
            structure.renumber_dofs()
            def_xs = [n.x + scale * u[n.dof_indices[0]] for n in nodes]
            def_zs = [n.z + scale * u[n.dof_indices[1]] for n in nodes]
            ax.scatter(
                def_xs, def_zs, s=14,
                c=cls.COLOR_DEFORMED, edgecolors="none",
                alpha=0.7, zorder=4, label="Deformed",
            )

        # nodes (undeformed)
        xs = [n.x for n in nodes]
        zs = [n.z for n in nodes]
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
        info = f"Nodes: {len(nodes)}  |  Springs: {len(springs)}"
        ax.text(
            0.01, 0.01, info,
            transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom", color="#555555",
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
    ) -> Figure:
        """Colour springs / nodes by strain energy.

        Parameters
        ----------
        structure : Structure
        node_energies : dict[int, float]
            Per-node energy values.
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

        # Spring energy = average of the two end-node energies.
        if not node_energies:
            return fig

        e_min = min(node_energies.values())
        e_max = max(node_energies.values())
        if e_max == e_min:
            e_max = e_min + 1.0

        cmap = plt.cm.RdYlBu_r

        def _normed(val: float) -> float:
            return (val - e_min) / (e_max - e_min)

        for sp in springs:
            ni, nj = sp.node_ids
            ei = node_energies.get(ni, 0.0)
            ej = node_energies.get(nj, 0.0)
            avg = (ei + ej) / 2.0
            colour = cmap(_normed(avg))
            ax.plot(
                [sp.node_i.x, sp.node_j.x],
                [sp.node_i.z, sp.node_j.z],
                color=colour, lw=1.5, zorder=1,
            )

        # Nodes
        xs = [n.x for n in nodes]
        zs = [n.z for n in nodes]
        cs = [_normed(node_energies.get(n.id, 0.0)) for n in nodes]
        sc = ax.scatter(xs, zs, c=cs, cmap=cmap, s=18, zorder=3, edgecolors="k", linewidths=0.3)
        cbar = fig.colorbar(sc, ax=ax, label="Strain Energy")
        cbar.ax.tick_params(labelsize=8)

        # Min / max energy annotation
        ax.text(
            0.01, 0.01,
            f"Min: {e_min:.2e}  |  Max: {e_max:.2e}  |  Nodes: {len(nodes)}",
            transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom", color="#555555",
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

    # Internal forces
    @classmethod
    def plot_internal_forces(
        cls,
        structure: Structure,
        spring_forces: dict[tuple[int, int], dict],
        title: str = "Internal Forces",
        ax: plt.Axes | None = None,
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

            cmap = plt.cm.coolwarm  # blue (compression) ↔ red (tension)
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
        for n in nodes:
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

        try:
            for idx, snap in enumerate(history):
                if mode == "bw":
                    fig = cls.plot_bw_density(
                        snap,
                        initial_structure=ref,
                        title=f"Iteration {idx}",
                    )
                else:
                    fig = cls.plot_structure(
                        snap,
                        title=f"Iteration {idx}",
                    )

                # Render figure to PIL Image
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                plt.close(fig)
                buf.seek(0)
                img = Image.open(buf).convert("RGBA")
                # Composite onto white background for GIF compatibility
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                background.paste(img, mask=img)
                frames.append(background.convert("RGB"))

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