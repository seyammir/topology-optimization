"""Save / load structure state as JSON."""

from __future__ import annotations

import json
from pathlib import Path

from ..models.structure import Structure


def save_state(structure: Structure, filepath: str | Path) -> None:
    """Serialise the full structure to a JSON file.

    Parameters
    ----------
    structure : Structure
        The structure to persist.
    filepath : str | Path
        Destination path (will be overwritten if it exists).
    """
    data = structure.to_dict()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def load_state(filepath: str | Path) -> Structure:
    """Load a structure from a JSON file.

    Parameters
    ----------
    filepath : str | Path

    Returns
    -------
    Structure
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return Structure.from_dict(data)


def state_to_json_string(structure: Structure) -> str:
    """Return the structure state as a JSON string (for download)."""
    return json.dumps(structure.to_dict(), indent=2, ensure_ascii=False)


def structure_from_json_string(json_str: str) -> Structure:
    """Parse a JSON string back into a :class:`Structure`."""
    return Structure.from_dict(json.loads(json_str))
