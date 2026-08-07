#!/usr/bin/env python3
"""Fail when LaTeX/BibTeX logs contain unresolved references or missing assets."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PATTERNS = (
    re.compile(r"LaTeX Warning: (?:Reference|Citation).+undefined", re.I),
    re.compile(r"There were undefined (?:references|citations)", re.I),
    re.compile(r"I didn't find a database entry", re.I),
    re.compile(r"undefined control sequence", re.I),
    re.compile(r"File [`'].+[`'] not found", re.I),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--blg", type=Path)
    args = parser.parse_args()
    paths = [args.log]
    if args.blg and args.blg.exists():
        paths.append(args.blg)
    failures: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in PATTERNS):
                failures.append(f"{path}:{line_number}: {line.strip()}")
    if failures:
        print("Unresolved build diagnostics:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("LaTeX/BibTeX log check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
