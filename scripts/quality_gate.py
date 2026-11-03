#!/usr/bin/env python3
"""Run the release-quality gate used by Git pre-push and GitHub Actions.

This gate is offline and non-training: it compiles the source, validates the
production examples, runs every test, and enforces coverage floors measured
from the current suite. It is deliberately separate from training and remote
dataset checks.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(command: list[str], env: dict[str, str]) -> None:
    """Run one gate command from the repository root and stop on failure."""
    print("\n[quality]", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def main() -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/noesis_pycache")

    run([PYTHON, "-m", "compileall", "-q", "core", "scripts", "examples", "tests"], env)
    run([PYTHON, "scripts/check_docs_style.py"], env)
    run([PYTHON, "examples/validate_examples.py"], env)
    run(
        [
            PYTHON,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "--cov=core",
            "--cov=scripts",
            "--cov-config=.coveragerc",
            "--cov-report=term-missing",
            "--cov-fail-under=80",
        ],
        env,
    )
    run(
        [
            PYTHON,
            "-m",
            "coverage",
            "report",
            "--include=core/*",
            "--fail-under=99",
        ],
        env,
    )
    print("\n[quality] PASS — all tests passed; overall coverage >=80%; core coverage >=99%.")


if __name__ == "__main__":
    main()
