#!/usr/bin/env python3
"""Lint compiled manuscript sources, release fields, labels, and citations."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compiled_sources() -> list[Path]:
    queue = [ROOT / "main.tex"]
    seen: set[Path] = set()
    while queue:
        path = queue.pop(0).resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\\(?:input|include)\{([^}]+)\}", text):
            candidate = (path.parent / target)
            if not candidate.suffix:
                candidate = candidate.with_suffix(".tex")
            queue.append(candidate)
    return sorted(seen)


def main() -> int:
    sources = compiled_sources()
    issues: list[str] = []
    banned = {
        r"\bTODO\b": "unresolved TODO",
        r"repository URL pending|immutable commit pending|immutable data archive pending": "placeholder release field",
        r"Evidence Against Train--Test Circuit Separability": "stale title",
        r"\bsharedness\b|shared dependence": "unmeasured sharedness language",
        r"Sharedness precedes compactness|Shared dependence is present first": "stale causal event claim",
        r"The same circuit supports both splits": "unrestricted same-circuit claim",
        r"universally falsified|all possible circuits are inseparable": "unrestricted causal claim",
        r"keep.only projection": "obsolete Fourier-operator name",
    }
    combined_parts: list[str] = []
    for path in sources:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            code = re.sub(r"(?<!\\)%.*$", "", line)
            combined_parts.append(code)
            for pattern, message in banned.items():
                if re.search(pattern, code, re.I):
                    issues.append(f"{path.relative_to(ROOT)}:{line_number}: {message}")
    combined = "\n".join(combined_parts)

    labels = re.findall(r"\\label\{([^}]+)\}", combined)
    for label, count in Counter(labels).items():
        if count > 1:
            issues.append(f"duplicate label {label!r} ({count})")
    refs = {
        ref.strip()
        for group in re.findall(r"\\(?:ref|cref|Cref|autoref)\{([^}]+)\}", combined)
        for ref in group.split(",")
    }
    for ref in sorted(refs - set(labels)):
        issues.append(f"unresolved reference target {ref!r}")

    bib_text = (ROOT / "references.bib").read_text(encoding="utf-8")
    bib_keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib_text)
    for key, count in Counter(bib_keys).items():
        if count > 1:
            issues.append(f"duplicate bibliography key {key!r} ({count})")
    cited = {
        key.strip()
        for group in re.findall(r"\\cite[tp]?\{([^}]+)\}", combined)
        for key in group.split(",")
    }
    for key in sorted(cited - set(bib_keys)):
        issues.append(f"citation absent from references.bib: {key!r}")

    release_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "README.md", ROOT / "ARXIV_RELEASE_CHECKLIST.md")
        if path.exists()
    )
    placeholder = re.compile(r"\[(?:repository URL|immutable (?:commit|data archive)) pending public release\]", re.I)
    for match in placeholder.finditer(release_text):
        issues.append(f"release placeholder remains at character {match.start()}")

    if issues:
        print("Manuscript lint failed:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1
    print(f"Manuscript lint: PASS ({len(sources)} TeX files, {len(cited)} citations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
