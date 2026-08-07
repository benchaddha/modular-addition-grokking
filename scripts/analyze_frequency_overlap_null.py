#!/usr/bin/env python3
"""Exact overlap null for three independent top-k subsets.

The calculation fixes the first subset without loss of generality, enumerates
the overlap of the second subset with it, and then enumerates the third subset
over the four regions induced by the first two. This avoids simulation noise.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path


def choose(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def exact_overlap_null(
    universe_size: int,
    subset_size: int,
    observed_pair_intersections: tuple[int, int, int],
    observed_union_size: int,
    reference_size: int,
) -> dict[str, object]:
    n = universe_size
    k = subset_size
    denominator = choose(n, k) ** 2
    observed_mean_jaccard = sum(
        Fraction(overlap, 2 * k - overlap)
        for overlap in observed_pair_intersections
    ) / 3

    triple_empty_weight = 0
    mean_jaccard_le_weight = 0
    union_eq_weight = 0
    union_ge_weight = 0
    total_weight = 0
    expected_mean_jaccard_numerator = Fraction(0, 1)

    # Fix set A. Let x=|A∩B|. The four categories available to C are
    # A∩B, A\B, B\A, and the complement of A∪B.
    for x in range(k + 1):
        b_weight = choose(k, x) * choose(n - k, k - x)
        category_sizes = (x, k - x, k - x, n - (2 * k - x))
        for c_ab in range(k + 1):
            for c_a_only in range(k + 1 - c_ab):
                for c_b_only in range(k + 1 - c_ab - c_a_only):
                    c_neither = k - c_ab - c_a_only - c_b_only
                    choices = (
                        choose(category_sizes[0], c_ab)
                        * choose(category_sizes[1], c_a_only)
                        * choose(category_sizes[2], c_b_only)
                        * choose(category_sizes[3], c_neither)
                    )
                    if choices == 0:
                        continue
                    weight = b_weight * choices
                    total_weight += weight
                    ab = x
                    ac = c_ab + c_a_only
                    bc = c_ab + c_b_only
                    mean_jaccard = (
                        Fraction(ab, 2 * k - ab)
                        + Fraction(ac, 2 * k - ac)
                        + Fraction(bc, 2 * k - bc)
                    ) / 3
                    union_size = 3 * k - (ab + ac + bc) + c_ab

                    expected_mean_jaccard_numerator += weight * mean_jaccard
                    if c_ab == 0:
                        triple_empty_weight += weight
                    if mean_jaccard <= observed_mean_jaccard:
                        mean_jaccard_le_weight += weight
                    if union_size == observed_union_size:
                        union_eq_weight += weight
                    if union_size >= observed_union_size:
                        union_ge_weight += weight

    if sum(
        choose(k, x) * choose(n - k, k - x) for x in range(k + 1)
    ) != choose(n, k):
        raise AssertionError("Second-subset overlap enumeration is incomplete")
    if total_weight != denominator:
        raise AssertionError("Three-subset overlap enumeration is incomplete")

    expected_mean_jaccard = expected_mean_jaccard_numerator / denominator
    reference_disjoint_given_union = Fraction(
        choose(n - reference_size, observed_union_size),
        choose(n, observed_union_size),
    )

    def probability(weight: int) -> float:
        return weight / denominator

    return {
        "universe_size": n,
        "subset_size": k,
        "num_subsets": 3,
        "observed_pair_intersections": list(observed_pair_intersections),
        "observed_mean_pairwise_jaccard": float(observed_mean_jaccard),
        "observed_union_size": observed_union_size,
        "expected_mean_pairwise_jaccard": float(expected_mean_jaccard),
        "probability_mean_pairwise_jaccard_le_observed": probability(
            mean_jaccard_le_weight
        ),
        "probability_empty_three_way_intersection": probability(triple_empty_weight),
        "probability_union_size_equals_observed": probability(union_eq_weight),
        "probability_union_size_at_least_observed": probability(union_ge_weight),
        "reference_size": reference_size,
        "probability_reference_disjoint_given_observed_union_size": float(
            reference_disjoint_given_union
        ),
        "method": (
            "Exact combinatorial enumeration with the first subset fixed; "
            "no Monte Carlo sampling."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-size", type=int, default=56)
    parser.add_argument("--subset-size", type=int, default=5)
    parser.add_argument(
        "--observed-pair-intersections",
        type=int,
        nargs=3,
        default=(0, 0, 2),
        metavar=("AB", "AC", "BC"),
    )
    parser.add_argument("--observed-union-size", type=int, default=13)
    parser.add_argument("--reference-size", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = exact_overlap_null(
        universe_size=args.universe_size,
        subset_size=args.subset_size,
        observed_pair_intersections=tuple(args.observed_pair_intersections),
        observed_union_size=args.observed_union_size,
        reference_size=args.reference_size,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
