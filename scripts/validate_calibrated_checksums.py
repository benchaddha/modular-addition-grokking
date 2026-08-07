#!/usr/bin/env python3
"""Validate checkpoint and learned-basis checksums for calibrated experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibrated_separability import tensor_checksum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact-root")
    parser.add_argument("--artifact-manifest")
    parser.add_argument("--skip-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    real_dir = Path(args.real_dir)
    with Path(args.checkpoint_manifest).open(encoding="utf-8", newline="") as handle:
        checkpoints = list(csv.DictReader(handle))
    run_metadata = sorted(real_dir.glob("jobs/*/runs/*.json"))
    artifact_files: List[Path] = []
    if bool(args.artifact_root) != bool(args.artifact_manifest):
        raise ValueError("--artifact-root and --artifact-manifest must be supplied together.")
    if args.artifact_root:
        artifact_root = Path(args.artifact_root)
        excluded = {
            Path(args.output).resolve(),
            Path(args.artifact_manifest).resolve(),
        }
        artifact_files = sorted(
            path
            for path in artifact_root.rglob("*")
            if path.is_file() and path.resolve() not in excluded
        )
    plan = {
        "checkpoint_files": 0 if args.skip_checkpoints else len(checkpoints),
        "real_run_records": len(run_metadata),
        "artifact_files": len(artifact_files),
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return

    failures: List[Dict[str, Any]] = []
    validated_checkpoints = 0
    if not args.skip_checkpoints:
        for row in checkpoints:
            path = REPO_ROOT / row["path"]
            observed = file_sha256(path)
            if observed != row["sha256"]:
                failures.append(
                    {
                        "kind": "checkpoint",
                        "path": str(path),
                        "expected": row["sha256"],
                        "observed": observed,
                    }
                )
            validated_checkpoints += 1

    validated_run_records = 0
    validated_bases = 0
    for metadata_path in run_metadata:
        basis_path = metadata_path.with_suffix(".pt")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not basis_path.exists():
            failures.append({"kind": "missing_basis_file", "path": str(basis_path)})
            continue
        tensors = torch.load(basis_path, map_location="cpu", weights_only=True)
        for metadata_key, tensor_key in (
            ("selected", "selected_basis"),
            ("capacity", "capacity_basis"),
        ):
            expected_record = metadata.get(metadata_key)
            if expected_record is None:
                if tensor_key in tensors:
                    failures.append(
                        {
                            "kind": "unexpected_basis",
                            "path": str(basis_path),
                            "tensor_key": tensor_key,
                        }
                    )
                continue
            if tensor_key not in tensors:
                failures.append(
                    {
                        "kind": "missing_basis",
                        "path": str(basis_path),
                        "tensor_key": tensor_key,
                    }
                )
                continue
            observed = tensor_checksum(tensors[tensor_key])
            expected = expected_record["basis_checksum"]
            if observed != expected:
                failures.append(
                    {
                        "kind": "basis_checksum",
                        "path": str(basis_path),
                        "tensor_key": tensor_key,
                        "expected": expected,
                        "observed": observed,
                    }
                )
            validated_bases += 1
        validated_run_records += 1

    if artifact_files:
        artifact_root = Path(args.artifact_root)
        artifact_rows = [
            {
                "path": str(path.relative_to(artifact_root)),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in artifact_files
        ]
        artifact_manifest = Path(args.artifact_manifest)
        artifact_manifest.parent.mkdir(parents=True, exist_ok=True)
        with artifact_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["path", "size_bytes", "sha256"]
            )
            writer.writeheader()
            writer.writerows(artifact_rows)

    result = {
        "status": "passed" if not failures else "failed",
        "validated_checkpoints": validated_checkpoints,
        "validated_real_run_records": validated_run_records,
        "validated_real_bases": validated_bases,
        "recorded_artifact_files": len(artifact_files),
        "failures": failures,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
