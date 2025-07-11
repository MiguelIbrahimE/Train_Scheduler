"""
BCPC – Bring Cities back to the People, not the Cars
Top-level package.  Exposes a tiny public API and
configures logging early so every sub-module inherits it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

__all__ = ["logger", "DATA_DIR", "INPUT_DIR", "OUTPUT_DIR", "PROJECT_ROOT"]

# ---------- paths ----------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
INPUT_DIR: Final[Path] = PROJECT_ROOT / "input"
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "output"

# ---------- logging ----------
LOG_LEVEL = os.getenv("BCPC_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("bcpc")
logger.debug("Logging initialised (level=%s)", LOG_LEVEL)

# ---------- runtime self-check ----------
for _path in (DATA_DIR, INPUT_DIR, OUTPUT_DIR):
    if not _path.exists():
        try:
            _path.mkdir(parents=True, exist_ok=True)
            logger.info("Created missing directory %s", _path)
        except OSError as exc:
            logger.error("Cannot create %s – %s", _path, exc, exc_info=True)
            raise
