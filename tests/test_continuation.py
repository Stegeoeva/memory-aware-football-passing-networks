import unittest
import numpy as np
import pandas as pd
from test_sequences import build, event
from src.continuation import outcome_material, values_from_material, xt_statistics, fit_fold_xt
from src.evaluation import hierarchical_weights


FAKE_XT = {"values": np.arange(96)/100}
CONFIG = {"xt_transition_prior": 5, "xt_shot_prior": 10}


class ContinuationTests(unittest.TestCase):
    def test_first_shot_uses_source_xg_before_action_horizon(self):
        tables = build([event(1, "Pass"), event(2, "Shot", player=2), event(3, "Half End")])
        material = outcome_material(tables)
        values = values_from_material(material, FAKE_XT, ["1:e1"])
        self.assertAlmostEqual(values["R1"][0], .2)
        self.assertAlmostEqual(values["R3"][0], .2)
        self.assertAlmostEqual(values["R5"][0], .2)

    def test_failed_pass_and_confirmed_opponent_control_has_zero_value(self):
        tables = build([event(1, "Pass"), event(2, "Pass", player=2, recipient=3, outcome="Incomplete"),
                        event(3, "Carry", player=9, team=2)])
        values = values_from_material(outcome_material(tables), FAKE_XT, ["1:e1"])
        self.assertEqual(values["R1"][0], 0)
        self.assertEqual(values["R3"][0], 0)

    def test_retained_third_action_uses_actual_endpoint(self):
        tables = build([event(1, "Pass"), event(2, "Carry", player=2, end_x=70),
                        event(3, "Pass", player=2, recipient=3, end_x=90),
                        event(4, "Carry", player=3, end_x=100), event(5, "Half End")])
        values = values_from_material(outcome_material(tables), FAKE_XT, ["1:e1"])
        self.assertAlmostEqual(values["R3"][0], .58)
        self.assertTrue(np.isnan(values["R5"][0]))

    def test_period_end_is_censored_and_not_assigned_zero(self):
        tables = build([event(1, "Pass"), event(2, "Ball Receipt*", player=2), event(3, "Half End")])
        values = values_from_material(outcome_material(tables), FAKE_XT, ["1:e1"])
        self.assertTrue(np.isnan(values["R3"][0]))

    def test_own_goal_without_source_shot_xg_not_misclassified_as_failed_chain(self):
        failed = event(2, "Pass", player=2, outcome="Incomplete")
        tables = build([event(1, "Pass"), failed, event(3, "Own Goal Against", team=2),
                        event(4, "Pass", player=9, team=2, possession=2)])
        material = outcome_material(tables)
        row = material.loc[material.sequence_id.eq("1:e1") & material.horizon.eq(3)].iloc[0]
        self.assertEqual(row.reason, "own_goal_without_source_shot_xg")
        self.assertEqual(row.status, "censored")

    def test_state_value_fit_cannot_read_heldout_goals(self):
        train = build([event(1, "Pass"), event(2, "Carry", player=2),
                       event(3, "Shot", player=2, outcome="Goal")])["events"]
        heldout = train.copy()
        heldout["match_id"] = 2
        all_events = pd.concat([train, heldout], ignore_index=True)
        before = fit_fold_xt(xt_statistics(all_events), [1], CONFIG)
        all_events.loc[all_events.match_id.eq(2), "outcome"] = "Off T"
        after = fit_fold_xt(xt_statistics(all_events), [1], CONFIG)
        np.testing.assert_array_equal(before["values"], after["values"])
        self.assertEqual(before["training_match_ids"], [1])
        self.assertLess(before["bellman_residual"], 1e-10)

    def test_match_possession_bootstrap_keeps_overlapping_passes_together(self):
        sequences = pd.DataFrame({"match_id": [1, 1, 1, 2], "possession_key": ["a", "a", "b", "c"]})
        blocks, weights = hierarchical_weights(sequences, 200, 1)
        self.assertEqual(len(blocks), 3)
        self.assertEqual(weights.shape, (200, 3))
        mapping = dict(zip(blocks.possession_key, range(len(blocks))))
        per_pass = weights[:, sequences.possession_key.map(mapping)]
        np.testing.assert_array_equal(per_pass[:, 0], per_pass[:, 1])


if __name__ == "__main__": unittest.main()
