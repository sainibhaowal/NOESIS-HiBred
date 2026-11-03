#!/usr/bin/env python3
"""Enforce objective documentation rules for repository Markdown files."""

from __future__ import annotations

import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = [PROJECT_ROOT / "README.md", *sorted((PROJECT_ROOT / "docs").glob("*.md"))]
FORBIDDEN_TEXT = {
    "em or en dash": re.compile(r"[—–]"),
    "first- or second-person prose": re.compile(
        r"\b(?:[Ww]e|[Oo]ur|[Yy]ou|[Yy]our)\b"
    ),
    "removed promotional claim": re.compile(
        r"100% Ready|unrejectable|Goldilocks|scrappy|will love|trend #1|toy model",
        re.IGNORECASE,
    ),
    "obsolete configuration reference": re.compile(r"config_10m\.json|core/config\.json"),
}


def prose_lines(path: Path) -> list[tuple[int, str]]:
    """Return Markdown prose lines while excluding fenced code blocks."""
    lines: list[tuple[int, str]] = []
    in_fence = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append((number, line))
    return lines


def main() -> int:
    errors: list[str] = []
    for path in DOCUMENTS:
        for number, line in prose_lines(path):
            for label, pattern in FORBIDDEN_TEXT.items():
                if pattern.search(line):
                    relative = path.relative_to(PROJECT_ROOT)
                    errors.append(f"{relative}:{number}: {label}: {line.strip()}")

    if errors:
        print("Documentation style check failed:", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        return 1

    print(f"Documentation style check passed: {len(DOCUMENTS)} Markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
