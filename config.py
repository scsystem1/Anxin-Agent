"""
Central configuration.

Reads environment variables (typically populated from .env via python-dotenv).
Exposes a typed view for the rest of the code.
"""

from __future__ import annotations
import os
from dataclasses import dataclass


# Load .env if present (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class RunConfig:
    case_path: str
    max_turns: int
    out_dir: str
    verbose: bool


def get_run_config() -> RunConfig:
    return RunConfig(
        case_path=os.getenv("ANXIN_CASE_PATH", "cases/tianjiao_mingyuan.json"),
        max_turns=int(os.getenv("ANXIN_MAX_TURNS", "30")),
        out_dir=os.getenv("ANXIN_OUT_DIR", "./out"),
        verbose=os.getenv("ANXIN_VERBOSE", "1") == "1",
    )
