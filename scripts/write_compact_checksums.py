#!/usr/bin/env python3
"""Write SHA-256 checksums for committed compact publication data."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = DATA / "CHECKSUMS.sha256"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    paths = sorted(path for path in DATA.rglob("*") if path.is_file() and path != OUTPUT)
    lines = [f"{digest(path)}  {path.relative_to(ROOT)}" for path in paths]
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(paths)} compact-data checksums")


if __name__ == "__main__":
    main()
