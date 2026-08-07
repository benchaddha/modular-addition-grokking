import unittest

import torch

from src.calibrated_separability import (
    AccessAudit,
    Candidate,
    DataSubset,
    MetricSet,
    PairMetrics,
    deterministic_three_way_split,
    select_validation_candidate,
)


def metric(accuracy: float, n: int = 20) -> MetricSet:
    return MetricSet(n, accuracy, 0.0, 1.0, 1.0)


def pair(train_acc: float, test_acc: float) -> PairMetrics:
    clean_train = metric(1.0)
    clean_test = metric(1.0)
    train_damage = 1.0 - train_acc
    test_damage = 1.0 - test_acc
    return PairMetrics(
        clean_train=clean_train,
        clean_test=clean_test,
        intervened_train=metric(train_acc),
        intervened_test=metric(test_acc),
        train_damage=train_damage,
        test_damage=test_damage,
        selective_gap=test_damage - train_damage,
        relative_train_retention=train_acc,
        relative_test_damage=test_damage,
        attainable_ceiling=0.8,
        recovered_ceiling_fraction=test_damage / 0.8,
        operational_pass=train_acc >= 0.9 and test_acc <= 0.2,
    )


def candidate(name: str, step: int, train_acc: float, test_acc: float) -> Candidate:
    return Candidate(
        candidate_id=name,
        step=step,
        rank=1,
        lam=1.0,
        restart=0,
        basis=torch.tensor([[1.0], [0.0]]),
        fit_metrics=pair(train_acc, test_acc),
        validation_metrics=pair(train_acc, test_acc),
    )


class ThreeWayProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokens = torch.arange(240).reshape(80, 3)
        self.labels = torch.arange(80) % 7

    def test_subsets_are_disjoint_and_exhaustive(self) -> None:
        split = deterministic_three_way_split(
            self.tokens, self.labels, seed=123, namespace="train"
        )
        split.assert_valid()
        fit = set(split.fit.indices.tolist())
        validation = set(split.validation.indices.tolist())
        final = set(split.final.indices.tolist())
        self.assertFalse(fit & validation)
        self.assertFalse(fit & final)
        self.assertFalse(validation & final)
        self.assertEqual(fit | validation | final, set(range(80)))
        self.assertEqual((len(fit), len(validation), len(final)), (40, 20, 20))

    def test_same_seed_gives_identical_split_and_selection(self) -> None:
        first = deterministic_three_way_split(
            self.tokens, self.labels, seed=91, namespace="test"
        )
        second = deterministic_three_way_split(
            self.tokens, self.labels, seed=91, namespace="test"
        )
        self.assertEqual(first.split_checksum, second.split_checksum)
        self.assertTrue(torch.equal(first.fit.indices, second.fit.indices))
        choices_a = [candidate("late", 20, 0.95, 0.40), candidate("early", 10, 0.95, 0.40)]
        choices_b = list(reversed(choices_a))
        self.assertEqual(
            select_validation_candidate(choices_a).candidate_id,
            select_validation_candidate(choices_b).candidate_id,
        )
        self.assertEqual(select_validation_candidate(choices_a).candidate_id, "early")

    def test_changing_final_labels_cannot_change_selection(self) -> None:
        split = deterministic_three_way_split(
            self.tokens, self.labels, seed=77, namespace="train"
        )
        choices = [candidate("winner", 10, 0.92, 0.10), candidate("runner_up", 0, 0.98, 0.30)]
        selected_before = select_validation_candidate(choices).candidate_id
        changed_final = split.final.with_labels(torch.flip(split.final.labels, dims=[0]))
        self.assertFalse(torch.equal(changed_final.labels, split.final.labels))
        selected_after = select_validation_candidate(choices).candidate_id
        self.assertEqual(selected_before, selected_after)
        self.assertEqual(selected_after, "winner")

    def test_no_feasible_candidate_is_explicit(self) -> None:
        choices = [candidate("bad", 0, 0.89, 0.0)]
        self.assertIsNone(select_validation_candidate(choices))

    def test_final_data_are_rejected_by_optimizer_audit(self) -> None:
        split = deterministic_three_way_split(
            self.tokens, self.labels, seed=3, namespace="train"
        )
        audit = AccessAudit()
        audit.record_optimizer(split.fit)
        self.assertEqual(audit.optimizer_roles, ["train_fit"])
        with self.assertRaisesRegex(ValueError, "non-fit"):
            audit.record_optimizer(split.final)
        self.assertNotIn("train_final", audit.optimizer_roles)

    def test_validation_tie_break_order(self) -> None:
        # Equal gap: prefer greater absolute test damage, then lower train
        # damage, then the earlier step.
        lower_damage = candidate("lower_test_damage", 0, 0.95, 0.25)  # gap .70
        greater_damage = candidate("greater_test_damage", 20, 0.90, 0.20)  # gap .70
        self.assertEqual(
            select_validation_candidate([lower_damage, greater_damage]).candidate_id,
            "greater_test_damage",
        )


if __name__ == "__main__":
    unittest.main()
