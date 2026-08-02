"""Cross-platform path bootstrap for the standalone Aris sidecar modules."""

from __future__ import annotations

import os
import sys
from pathlib import Path


SIDECAR_DIR = Path(__file__).resolve().parent
LAAP_ROOT = Path(os.environ.get("LAAP_ROOT", SIDECAR_DIR.parents[2])).expanduser().resolve()

if str(LAAP_ROOT) not in sys.path:
    sys.path.insert(1, str(LAAP_ROOT))

from laap.config.paths import (  # noqa: E402
    get_aris_brain_dir,
    get_hana_home,
    get_laap_home,
    get_state_dir,
)


LAAP_HOME = get_laap_home()
HANA_HOME = get_hana_home()
ARIS_BRAIN_DIR = get_aris_brain_dir()
LAAP_AGI_DIR = LAAP_ROOT / "laap" / "agi"
SIDECAR_STATE_DIR = get_state_dir() / "aris-sidecar"


def ensure_runtime_dirs() -> None:
    """Create only writable runtime directories, never source directories."""
    for directory in (LAAP_HOME, HANA_HOME, ARIS_BRAIN_DIR, SIDECAR_STATE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
