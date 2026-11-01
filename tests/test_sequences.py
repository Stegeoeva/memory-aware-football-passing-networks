"""Temporal, context-availability and censoring invariants for the 51-match data."""
import unittest
import pandas as pd

from src.sequences import build_match_sequences, dataframe, pitch_zone, role_group


MATCH = {"match_id": 1, "home_team": {"home_team_id": 1, "home_team_name": "A"},
         "away_team": {"away_team_id": 2, "away_team_name": "B"}}


def event(i, kind, player=1, team=1, possession=1, period=1, time=None,
          recipient=2, x=40, end_x=60, outcome=None, position=None):
    row = {"id": f"e{i}", "index": i, "period": period, "timestamp": f"00:00:{(time if time is not None else i):06.3f}",
           "type": {"name": kind}, "team": {"id": team, "name": "A" if team == 1 else "B"},
           "possession": possession, "possession_team": {"id": 1, "name": "A"},
           "player": {"id": player, "name": str(player)}, "location": [x, 40], "duration": .5}
    if position:
        row["position"] = {"id": position, "name": f"Position {position}"}
    if kind in {"Pass", "Carry", "Dribble", "Ball Receipt*", "Shot"}:
        key = "ball_receipt" if kind == "Ball Receipt*" else kind.lower()
        detail = {"end_location": [end_x, 40]}
        if kind == "Pass":
            if recipient is not None:
                detail["recipient"] = {"id": recipient, "name": str(recipient)}
            detail.update(length=abs(end_x-x), height={"name": "Ground Pass"})
        if kind == "Shot":
            detail.update(statsbomb_xg=.2, outcome={"name": "Saved"}, type={"name": "Open Play"})
        if outcome:
            detail["outcome"] = {"name": outcome}
        row[key] = detail
    return row


def build(events):
    return {k: dataframe(v) for k, v in build_match_sequences(events, MATCH).items()}


def initial(tables):
    return tables["receiver_sequences"].loc[lambda f: f.a_event_id.eq("e1")].iloc[0]


class SequenceTests(unittest.TestCase):
    def test_optional_carry_pressure_and_successful_dribble_keep_real_link(self):
        t = build([event(1, "Pass"), event(2, "Ball Receipt*", player=2),
                   event(3, "Pressure", player=9, team=2), event(4, "Carry", player=2),
                   event(5, "Dribble", player=2, outcome="Complete"), event(6, "Pass", player=2, recipient=3)])
        b = t["two_pass_sequences"].iloc[0]
        self.assertEqual(b.b_event_id, "e6")
        self.assertEqual(b.c_id, 3)
        self.assertEqual((b.receiver_carries_before_release, b.receiver_dribbles_before_release), (1, 1))
        self.assertTrue(b.memory_target_eligible)
        self.assertEqual(initial(t).future_actions_before_boundary, 3)

    def test_other_teammate_ball_receipt_prevents_manufactured_motif(self):
        t = build([event(1, "Pass"), event(2, "Ball Receipt*", player=3), event(3, "Pass", player=2, recipient=4)])
        self.assertTrue(t["two_pass_sequences"].empty)
        self.assertEqual(initial(t).receiver_status, "other_teammate_handler")

    def test_opponent_control_stops_chain_even_with_same_possession_id(self):
        t = build([event(1, "Pass"), event(2, "Ball Receipt*", player=9, team=2), event(3, "Pass", player=2, recipient=4)])
        self.assertEqual(initial(t).stop_reason, "opponent_control")
        self.assertEqual(initial(t).future_actions_before_boundary, 0)

    def test_incomplete_opponent_receipt_is_not_confirmed_control(self):
        t = build([event(1, "Pass"), event(2, "Ball Receipt*", player=9, team=2, outcome="Incomplete"),
                   event(3, "Pass", player=2, recipient=4)])
        self.assertTrue(t["two_pass_sequences"].a_event_id.eq("e1").any())

    def test_opposing_miscontrol_at_successful_receipt_is_flagged_not_assumed_control(self):
        t = build([event(1, "Pass"), event(2, "Ball Receipt*", player=2),
                   event(3, "Miscontrol", player=9, team=2, time=2),
                   event(4, "Pass", player=2, recipient=3)])
        b = t["two_pass_sequences"].iloc[0]
        self.assertTrue(b.memory_target_eligible)
        self.assertEqual(b.opponent_noncontrol_annotations, 1)
        self.assertEqual(b.opponent_noncontrol_event_ids, "e3")

    def test_period_and_provider_possession_boundaries_are_respected(self):
        for extra in [{"period": 2}, {"possession": 2}]:
            t = build([event(1, "Pass"), event(2, "Pass", player=2, recipient=3, **extra)])
            self.assertFalse(initial(t).b_next_pass_observed)

    def test_restart_with_same_possession_does_not_become_next_open_play_pass(self):
        restart = event(2, "Pass", player=2, recipient=3)
        restart["pass"]["type"] = {"name": "Throw-in"}
        t = build([event(1, "Pass"), restart])
        self.assertEqual(initial(t).stop_reason, "restart")
        self.assertFalse(initial(t).b_next_pass_observed)

    def test_foul_advantage_continues_but_stoppage_does_not(self):
        for advantage in [False, True]:
            foul = event(2, "Foul Committed", player=9, team=2)
            foul["foul_committed"] = {"advantage": advantage}
            t = build([event(1, "Pass"), foul, event(3, "Pass", player=2, recipient=3)])
            self.assertEqual(bool(initial(t).b_next_pass_observed), advantage)

    def test_receiver_foul_won_with_advantage_keeps_actual_next_pass(self):
        foul = event(2, "Foul Won", player=2)
        foul["foul_won"] = {"advantage": True}
        t = build([event(1, "Pass"), foul, event(3, "Pass", player=2, recipient=3)])
        self.assertTrue(initial(t).b_next_pass_observed)

    def test_failed_terminal_out_pass_is_included_before_boundary(self):
        failed = event(2, "Pass", player=2, outcome="Incomplete")
        failed["out"] = True
        t = build([event(1, "Pass"), failed, event(3, "Carry", player=9, team=2)])
        s = initial(t)
        self.assertTrue(s.b_next_pass_observed)
        self.assertTrue(s.stop_action_included)
        self.assertEqual(s.stop_reason, "ball_out")
        self.assertEqual(s.future_actions_before_boundary, 1)

    def test_own_goal_outcome_distinguishes_for_and_against_initial_team(self):
        for team, own_goal_for in [(1, False), (2, True)]:
            t = build([event(1, "Pass"), event(2, "Own Goal Against", team=team)])
            w = t["future_windows"].loc[lambda f: f.horizon_kind.eq("seconds") & f.horizon_value.eq(10)].iloc[0]
            self.assertEqual(bool(w.goal_for_team_in_window), own_goal_for)
            self.assertEqual(bool(w.goal_against_team_in_window), not own_goal_for)

    def test_estimated_receipt_after_half_end_remains_flagged_and_censored(self):
        a = event(1, "Pass")
        a["duration"] = 5.
        t = build([a, event(2, "Half End")])
        self.assertTrue(initial(t).receipt_after_boundary)
        w = t["future_windows"].loc[lambda f: f.horizon_kind.eq("seconds")].iloc[0]
        self.assertTrue(w.censored)
        self.assertFalse(w.shot_outcome_known)

    def test_failed_b_pass_keeps_intended_recipient_without_inventing_actual_c(self):
        t = build([event(1, "Pass"), event(2, "Pass", player=2, recipient=3, outcome="Incomplete"),
                   event(3, "Carry", player=9, team=2)])
        b = t["two_pass_sequences"].iloc[0]
        self.assertTrue(pd.isna(b.c_id))
        self.assertEqual(b.b_intended_recipient_id, 3)
        self.assertFalse(b.memory_target_eligible)
        self.assertTrue(initial(t).receiver_loss_confirmed)

    def test_failed_b_pass_retained_by_teammate_is_not_a_confirmed_loss(self):
        t = build([event(1, "Pass"), event(2, "Pass", player=2, recipient=3, outcome="Incomplete"),
                   event(3, "Carry", player=3)])
        self.assertFalse(initial(t).receiver_loss_confirmed)

    def test_failed_dribble_is_resolution_not_skipped_to_later_pass(self):
        t = build([event(1, "Pass"), event(2, "Dribble", player=2, outcome="Incomplete"),
                   event(3, "Pass", player=2, recipient=3)])
        self.assertFalse(initial(t).b_next_pass_observed)
        self.assertEqual(initial(t).receiver_resolution_type, "Dribble")

    def test_failed_initial_passes_and_restarts_are_preserved_in_attempts(self):
        restart = event(3, "Pass")
        restart["pass"]["type"] = {"name": "Kick Off"}
        t = build([event(1, "Pass", outcome="Incomplete"), event(2, "Pass", recipient=None), restart])
        self.assertEqual(len(t["pass_attempts"]), 3)
        self.assertEqual(len(t["receiver_sequences"]), 0)

    def test_receipt_related_event_has_priority_over_duration_estimate(self):
        a = event(1, "Pass", time=1)
        a["related_events"] = ["e2"]
        a["duration"] = 5.
        receipt = event(2, "Ball Receipt*", player=2, time=2, position=14)
        receipt["under_pressure"] = True
        t = build([a, receipt, event(3, "Pass", player=2, recipient=3, time=3)])
        s = initial(t)
        self.assertEqual(s.receipt_clock_seconds, 2)
        self.assertEqual(s.receipt_source, "related_event")
        self.assertTrue(s.b_under_pressure_at_receipt)

    def test_missing_receipt_does_not_borrow_future_release_role_or_pressure(self):
        release = event(2, "Pass", player=2, recipient=3, position=23)
        release["under_pressure"] = True
        t = build([event(1, "Pass"), release])
        s = initial(t)
        self.assertTrue(pd.isna(s.b_position_id))
        self.assertTrue(pd.isna(s.b_under_pressure_at_receipt))
        self.assertEqual(s.b_pressure_state, "unobserved_receipt")

    def test_starting_xi_role_fallback_uses_past_tactical_snapshot(self):
        start = event(0, "Starting XI", time=0)
        start["tactics"] = {"lineup": [{"player": {"id": 2, "name": "2"},
                                       "position": {"id": 14, "name": "Centre Midfield"}}]}
        t = build([start, event(1, "Pass"), event(2, "Pass", player=2, recipient=3, position=23)])
        self.assertEqual(initial(t).b_position_id, 14)
        self.assertEqual(initial(t).b_role_source_event_id, "e0")

    def test_local_receipt_fallback_stops_at_receivers_first_decision(self):
        t = build([event(1, "Pass"), event(2, "Pass", player=2, recipient=3),
                   event(3, "Ball Receipt*", player=2, time=2.5)])
        self.assertFalse(initial(t).receipt_observed)

    def test_time_window_period_end_is_censored_not_a_known_negative(self):
        t = build([event(1, "Pass"), event(2, "Ball Receipt*", player=2), event(3, "Half End", time=3)])
        w = t["future_windows"].loc[lambda f: f.horizon_kind.eq("seconds") & f.horizon_value.eq(10)].iloc[0]
        self.assertTrue(w.censored)
        self.assertFalse(w.shot_outcome_known)

    def test_early_opponent_control_resolves_time_window_without_full_clock(self):
        t = build([event(1, "Pass"), event(2, "Carry", player=9, team=2)])
        w = t["future_windows"].loc[lambda f: f.horizon_kind.eq("seconds") & f.horizon_value.eq(10)].iloc[0]
        self.assertTrue(w.confirmed_loss_in_window)
        self.assertTrue(w.observation_complete)

    def test_source_shot_xg_and_failed_entries_are_separate(self):
        t = build([event(1, "Pass", end_x=70), event(2, "Pass", player=2, recipient=3, x=70, end_x=110, outcome="Incomplete"),
                   event(3, "Shot", player=3, time=3)])
        w = t["future_windows"].loc[lambda f: f.sequence_id.eq("1:e1") & f.horizon_kind.eq("actions") & f.horizon_value.eq(3)].iloc[0]
        self.assertEqual(w.source_xg_sum, .2)
        self.assertFalse(w.box_entry_observed)
        self.assertEqual(w.action_event_ids, "e2;e3")

    def test_goal_and_own_goal_pair_update_score_only_once(self):
        shot = event(1, "Shot", outcome="Goal")
        own_against = event(2, "Own Goal Against", team=2)
        own_for = event(3, "Own Goal For", team=1)
        t = build([shot, own_against, own_for, event(4, "Pass"), event(5, "Ball Receipt*", player=2)])
        rows = t["events"]
        self.assertEqual(rows.home_score_after.tolist(), [1, 2, 2, 2, 2])
        self.assertEqual(rows.scoring_team_id.notna().sum(), 2)

    def test_penalty_shootout_is_excluded_but_extra_time_is_retained(self):
        t = build([event(1, "Pass", period=3), event(2, "Shot", period=5, outcome="Goal")])
        self.assertEqual(len(t["events"]), 1)
        self.assertEqual(t["events"].scope.iloc[0], "extra_time")

    def test_zones_handle_boundaries_missing_and_invalid_coordinates(self):
        self.assertEqual(pitch_zone(120, 80), 17)
        self.assertEqual(pitch_zone(120, 80, 4, 3), 11)
        self.assertEqual(pitch_zone(20, 0), 1)
        for x, y in [(None, 40), (float("nan"), 40), (-.1, 40), (121, 40)]:
            self.assertIsNone(pitch_zone(x, y))
        self.assertEqual(role_group(1), "goalkeeper")
        self.assertIsNone(role_group(None))


if __name__ == "__main__":
    unittest.main()
