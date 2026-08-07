#!/usr/bin/env python3
"""Fail if committed summaries disagree with manuscript headline claims."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, tol: float = 5e-4) -> None:
    assert math.isclose(actual, expected, abs_tol=tol), (actual, expected)


def main() -> None:
    summary = json.loads((ROOT / "data/calibrated/calibrated_summary.json").read_text())
    assert summary["synthetic_gate_successes"] == 40
    assert summary["synthetic_gate_total"] == 40
    synthetic = rows("data/calibrated/synthetic_detection_summary.csv")
    true_or_higher = [r for r in synthetic if r["control"] == "positive" and r["rank_relation"] in {"true", "higher"}]
    assert sum(int(r["passes"]) for r in true_or_higher) == 480
    assert sum(int(r["n"]) for r in true_or_higher) == 480
    assert summary["standard_criterion_pass_90"] == 0
    assert summary["standard_criterion_pass_80"] == 0
    assert summary["standard_criterion_pass_95"] == 0
    assert summary["standard_final_rows"] == 70
    assert summary["rank64_criterion_pass_90"] == 0
    assert summary["rank64_criterion_pass_80"] == 0
    assert summary["rank64_criterion_pass_95"] == 0
    assert summary["rank64_selected_rows"] + summary["rank64_no_feasible_rows"] == 30
    assert summary["expanded_rank64_grid_selected_rank64"] == 0
    assert summary["standard_capacity_damages_both"] == 68

    phase = rows("data/calibrated/real_checkpoint_summary.csv")
    lookup = {(r["setting"], r["milestone_label"]): r for r in phase}
    close(float(lookup[("p97_1layer", "80")]["median_final_selective_gap"]), .251)
    close(float(lookup[("p113_2layer", "80")]["median_final_selective_gap"]), .177)

    real = rows("data/calibrated/real_learned_vs_best_control.csv")
    maximum = max(real, key=lambda r: float(r["learned_final_selective_gap"]))
    close(float(maximum["learned_final_selective_gap"]), .284)
    close(float(maximum["learned_final_train_retention"]), .909)
    close(float(maximum["learned_final_test_damage"]), .375)
    assert maximum["job_id"] == "p97_1layer-seed104-80"

    selected = rows("data/calibrated/selected_final_candidates.csv")
    standard = [r for r in selected if r["comparison"] == "standard" and r["status"] == "selected"]
    assert len(standard) == 70
    assert sum(r["criterion_pass_90"] == "True" for r in standard) == 0
    pristine = [r for r in standard if r["job_id"] != "p97_1layer-seed100-20"]
    assert len(pristine) == 69
    target = next(r for r in standard if r["job_id"] == maximum["job_id"])
    close(float(target["final_test_accuracy"]), .432)

    manuscript = (ROOT / "main.tex").read_text()
    required_literals = (
        "Calibrated Residual-Stream Interventions Reveal Partial but Not Strong",
        "0/69",
        "0/70",
        "40/40",
        "480/480",
        "0.251",
        "0.177",
        "0.284",
        "0.909",
        "0.375",
        "0.432",
        "68/70",
        "0/30",
    )
    for literal in required_literals:
        assert literal in manuscript, f"headline literal absent from manuscript: {literal}"
    print("headline-number consistency: PASS")


if __name__ == "__main__":
    main()
