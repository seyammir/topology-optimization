"""Import a black-and-white image as a mass-spring structure.

Black pixels represent material (nodes are created), white pixels
represent void (no nodes).  The image is resized to the requested grid
dimensions and then converted into a :class:`~src.models.structure.Structure`
with the same connectivity pattern used by
:meth:`Structure.create_rectangular`.
"""

from __future__ import annotations

import logging
from typing import IO

import numpy as np
from PIL import Image

from ..models.node import Node
from ..models.structure import Structure

logger = logging.getLogger(__name__)


def structure_from_image(
    source: str | IO[bytes],
    width: int,
    height: int,
    *,
    threshold: int = 128,
) -> Structure:
    """Create a :class:`Structure` from a black-and-white image.

    Parameters
    ----------
    source : str | IO[bytes]
        File path or file-like object (e.g. ``BytesIO`` from an upload).
    width : int
        Number of cells in x-direction (image will be resized to
        ``width + 1`` columns).
    height : int
        Number of cells in z-direction (image will be resized to
        ``height + 1`` rows).
    threshold : int
        Grey-value threshold (0-255).  Pixels **below** this value are
        treated as material (black); pixels at or above are void (white).

    Returns
    -------
    Structure
        The resulting mass-spring structure.

    Raises
    ------
    ValueError
        If *width* or *height* is less than 1, or the image yields
        no material nodes.
    OSError
        If the image cannot be opened.
    """
    if width < 1 or height < 1:
        raise ValueError(
            f"width and height must be >= 1, got width={width}, height={height}"
        )

    cols = width + 1  # number of node columns
    rows = height + 1  # number of node rows

    # 1) Load & convert to greyscale, resize to grid dimensions.
    img = Image.open(source).convert("L")  # 8-bit greyscale
    img = img.resize((cols, rows), Image.LANCZOS)

    # 2) Build boolean mask: True = material (black pixel).
    pixels = np.array(img, dtype=np.uint8)  # shape (rows, cols)
    material: np.ndarray = pixels < threshold  # True where dark

    if not material.any():
        raise ValueError(
            "The image contains no material pixels (all white). "
            "Use a darker image or lower the threshold."
        )

    # 3) Create structure - place nodes only where material is True.
    struct = Structure()
    node_map: dict[tuple[int, int], Node] = {}

    for iz in range(rows):
        for ix in range(cols):
            if material[iz, ix]:
                node = struct.add_node(float(ix), float(iz))
                node_map[(ix, iz)] = node

    # 4) Connect springs (same pattern as Structure.create_rectangular).
    for iz in range(rows):
        for ix in range(cols):
            n = node_map.get((ix, iz))
            if n is None:
                continue
            # Right neighbour (horizontal)
            right = node_map.get((ix + 1, iz))
            if right is not None:
                struct.add_spring(n, right)
            # Bottom neighbour (vertical)
            below = node_map.get((ix, iz + 1))
            if below is not None:
                struct.add_spring(n, below)
            # Bottom-right diagonal
            br = node_map.get((ix + 1, iz + 1))
            if br is not None:
                struct.add_spring(n, br)
            # Bottom-left diagonal
            bl = node_map.get((ix - 1, iz + 1))
            if bl is not None:
                struct.add_spring(n, bl)

    struct.renumber_dofs()
    logger.info(
        "Created structure from image: %dx%d grid, %d/%d material nodes, %d springs",
        width, height, struct.num_nodes, rows * cols,
        struct.graph.number_of_edges(),
    )
    return struct
