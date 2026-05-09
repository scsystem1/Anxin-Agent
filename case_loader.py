"""Case loading and validation."""

from __future__ import annotations
import json
from pathlib import Path


def load_case(path: str | Path) -> dict:
    """Load a case JSON. Codex: add basic schema validation."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Case file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    _validate(data)
    return data


def _validate(data: dict) -> None:
    """
    Cheap structural validation. Codex: extend as needed.

    Required top-level keys for the runner:
      case_id, case_name, ground_truth, worker, financial,
      evidence_database, npcs, action_space, terminal_conditions
    """
    required = [
        "case_id", "case_name", "ground_truth", "worker", "financial",
        "evidence_database", "npcs", "action_space", "terminal_conditions",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Case JSON missing required keys: {missing}")
