#!/usr/bin/env python3
"""Dependency-free validation for the checked-in inference examples.

This validates fixture integrity and schema consistency only. It does not load a
model, run inference, or train anything.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IMAGE_TAG = re.compile(r"<image>([^<]+)</image>")


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def require_jpeg(path: Path) -> None:
    require(path.is_file(), f"missing image: {path}")
    data = path.read_bytes()
    require(data[:3] == b"\xff\xd8\xff", f"not a JPEG file: {path}")
    require(b"git-lfs.github.com" not in data[:512], f"unhydrated Git-LFS pointer: {path}")


def walk_image_urls(value):
    if isinstance(value, dict):
        if isinstance(value.get("image_url"), dict) and "url" in value["image_url"]:
            yield value["image_url"]["url"]
        for child in value.values():
            yield from walk_image_urls(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_image_urls(child)


def main() -> None:
    text_path = ROOT / "example.txt"
    harmony_path = ROOT / "example_harmony.json"
    eval_path = ROOT / "production_eval.jsonl"

    require(text_path.is_file(), "example.txt is missing")
    require(harmony_path.is_file(), "example_harmony.json is missing")
    require(eval_path.is_file(), "production_eval.jsonl is missing")

    text = text_path.read_text(encoding="utf-8")
    tagged_images = IMAGE_TAG.findall(text)
    require(len(tagged_images) == 2, "example.txt must contain exactly two image tags")

    cases = json.loads(harmony_path.read_text(encoding="utf-8"))
    require(isinstance(cases, list) and len(cases) == 4, "example_harmony.json must contain four cases")

    referenced = list(tagged_images)
    for case in cases:
        referenced.extend(walk_image_urls(case))
    for relative in sorted(set(referenced)):
        image_path = ROOT.parent / relative
        require_jpeg(image_path)

    eval_cases = []
    for line_number, line in enumerate(eval_path.read_text(encoding="utf-8").splitlines(), 1):
        require(line.strip(), f"empty production_eval.jsonl line {line_number}")
        case = json.loads(line)
        require(case.get("id"), f"missing id on production_eval.jsonl line {line_number}")
        require(case.get("messages"), f"missing messages on production_eval.jsonl line {line_number}")
        require("expected" in case, f"missing expected result on production_eval.jsonl line {line_number}")
        eval_cases.append(case)
    require(len(eval_cases) >= 10, "production_eval.jsonl must contain at least ten evaluation cases")

    print(f"PASS: {len(cases)} harmony cases, {len(eval_cases)} production evaluation cases")
    print(f"PASS: {len(set(referenced))} referenced JPEG assets are present and hydrated")


if __name__ == "__main__":
    main()
