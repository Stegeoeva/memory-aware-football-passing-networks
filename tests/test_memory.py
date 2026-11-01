import unittest
import numpy as np
import pandas as pd
from src.memory import MemoryCounts
from src.evaluation import (fold_map, split_records, assert_split, aligned_probabilities,
                            probability_metrics, context_features, paired_match_interval)


def synthetic(n, seed, memory):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 3, n)
    target = a if memory else rng.integers(0, 3, n)
    return pd.DataFrame({"count_group": 0, "state_a": a, "state_b": 0, "target": target,
                         "match_id": rng.integers(0, 10, n)})


class MemoryTests(unittest.TestCase):
    def test_known_memory_predicts_heldout_paths(self):
        train, test = synthetic(6000, 1, True), synthetic(3000, 2, True)
        candidates = np.tile(np.arange(3), (len(test), 1))
        probabilities = MemoryCounts.probabilities(MemoryCounts(train).prepare(test, candidates), 10, 10)
        score = [probability_metrics(p, candidates, test.target.to_numpy()).log_loss_bits.mean() for p in probabilities]
        self.assertGreater(score[1]-score[2], 1.4)

    def test_first_order_null_does_not_invent_heldout_gain(self):
        train, test = synthetic(9000, 3, False), synthetic(6000, 4, False)
        candidates = np.tile(np.arange(3), (len(test), 1))
        probabilities = MemoryCounts.probabilities(MemoryCounts(train).prepare(test, candidates), 10, 10)
        score = [probability_metrics(p, candidates, test.target.to_numpy()).log_loss_bits.mean() for p in probabilities]
        self.assertLess(score[1]-score[2], .02)

    def test_unseen_players_get_positive_mass_on_common_candidates(self):
        raw = np.array([[.2, .8]])
        candidates = np.array([[10, 11, 12, -1]])
        prior = np.array([[1/3, 1/3, 1/3, 0]])
        prob = aligned_probabilities(raw, [10, 11], candidates, prior, .1)
        self.assertGreater(prob[0, 2], 0)
        self.assertEqual(prob[0, 3], 0)
        self.assertAlmostEqual(prob.sum(), 1)

    def test_unknown_memory_state_backs_off_exactly(self):
        train = synthetic(100, 1, True)
        test = pd.DataFrame({"count_group": [999], "state_a": [999], "state_b": [999], "target": [10], "match_id": [11]})
        allowed = np.array([[10, 11, -1]])
        probabilities = MemoryCounts.probabilities(MemoryCounts(train).prepare(test, allowed), 5, 50)
        for p in probabilities:
            np.testing.assert_allclose(p, [[.5, .5, 0]])

    def test_predictions_do_not_depend_on_heldout_target_labels(self):
        train, test = synthetic(1000, 1, True), synthetic(200, 2, True)
        allowed = np.tile(np.arange(3), (len(test), 1))
        model = MemoryCounts(train)
        before = MemoryCounts.probabilities(model.prepare(test, allowed))
        test.target = (test.target+1)%3
        after = MemoryCounts.probabilities(model.prepare(test, allowed))
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)

    def test_nested_splits_group_whole_matches_and_exclude_outer_test(self):
        config = {"repeats": 3, "outer_folds": 5, "inner_folds": 3, "seed": 20261101}
        records = split_records(range(51), config)
        self.assertEqual(len(records), 15)
        for record in records:
            assert_split(record)
        for repeat in range(3):
            seen = [m for r in records if r["repeat"] == repeat for m in r["test_match_ids"]]
            self.assertEqual(sorted(seen), list(range(51)))
        self.assertEqual(fold_map(range(51), 5, 1), fold_map(range(51), 5, 1))

    def test_receipt_features_do_not_read_future_outcomes_or_passer_identity(self):
        f = pd.DataFrame({"receipt_x": [60.], "receipt_y": [40.], "receipt_zone_6x3": [9],
            "b_role": ["midfielder"], "b_pressure_state": ["not_flagged"], "team_id": [1], "opponent_id": [2],
            "period": [1], "receipt_clock_seconds": [10.], "score_difference_at_receipt": [0], "b_id": [20],
            "a_id": [10], "c_id": [30], "receiver_resolution_type": ["Shot"]})
        before = context_features(f)
        f["c_id"], f["receiver_resolution_type"], f["a_id"] = 999, "Miscontrol", 999
        pd.testing.assert_frame_equal(before, context_features(f))
        self.assertNotIn("receiver_B", before)
        self.assertIn("receiver_B", context_features(f, include_b=True))

    def test_match_cluster_interval_not_false_independent_rows(self):
        f = pd.DataFrame({"match_id": [1]*100+[2]*100, "gain_bits": [1.]*100+[-1.]*100})
        result = paired_match_interval(f, repetitions=1000, seed=1)
        self.assertLess(result["ci95_low"], -.9)
        self.assertGreater(result["ci95_high"], .9)


if __name__ == "__main__": unittest.main()
