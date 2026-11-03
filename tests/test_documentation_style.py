"""Regression test for the repository documentation style policy."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def test_documentation_style_policy() -> None:
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import check_docs_style

        assert check_docs_style.main() == 0
    finally:
        sys.path.remove(str(SCRIPTS_DIR))
