"""Save / load structure state as JSON.

Provides file-based and string-based serialisation for
:class:`~src.models.structure.Structure` objects.  All public functions
raise descriptive exceptions on failure so that callers (e.g. the
Streamlit app) can surface user-friendly messages.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..models.structure import Structure

logger = logging.getLogger(__name__)


def save_state(structure: Structure, filepath: str | Path) -> None:
    """Serialise the full structure to a JSON file.

    Parameters
    ----------
    structure : Structure
        The structure to persist.
    filepath : str | Path
        Destination path (will be overwritten if it exists).

    Raises
    ------
    OSError
        If the file cannot be written (permissions, disk full, …).
    """
    data: dict[str, Any] = structure.to_dict()
    filepath = Path(filepath)
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        logger.info("Saved structure state to %s", filepath)
    except OSError:
        logger.exception("Failed to write state file: %s", filepath)
        raise


def load_state(filepath: str | Path) -> Structure:
    """Load a structure from a JSON file.

    Parameters
    ----------
    filepath : str | Path

    Returns
    -------
    Structure

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    KeyError
        If required fields are missing in the JSON data.
    """
    filepath = Path(filepath)
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
        logger.info("Loaded state file: %s", filepath)
    except FileNotFoundError:
        logger.error("State file not found: %s", filepath)
        raise
    except json.JSONDecodeError:
        logger.exception("Invalid JSON in state file: %s", filepath)
        raise
    return Structure.from_dict(data)


def state_to_json_string(structure: Structure) -> str:
    """Return the structure state as a JSON string (for download).

    Raises
    ------
    TypeError
        If the structure data is not JSON-serialisable.
    """
    try:
        return json.dumps(structure.to_dict(), indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        logger.exception("Failed to serialise structure to JSON string")
        raise


def structure_from_json_string(json_str: str) -> Structure:
    """Parse a JSON string back into a :class:`Structure`.

    Raises
    ------
    json.JSONDecodeError
        If *json_str* is not valid JSON.
    KeyError
        If required fields are missing.
    """
    try:
        data: dict[str, Any] = json.loads(json_str)
    except json.JSONDecodeError:
        logger.exception("Invalid JSON string provided for structure loading")
        raise
    return Structure.from_dict(data)