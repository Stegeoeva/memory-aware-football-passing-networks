import unittest
import numpy as np
import pandas as pd
from test_sequences import build, event
from src.motifs import canonical_pattern, build_motifs, shot15_labels, motif_features

CONFIG={"time_tolerance_seconds":.001,"maximum_gap_seconds":5}


def three_pass(players, extra=None):
    events=[event(1,"Carry",player=players[0],time=0),
            event(2,"Pass",player=players[0],recipient=players[1],time=1),
            event(3,"Ball Receipt*",player=players[1],time=1.5),
            event(4,"Pass",player=players[1],recipient=players[2],time=3),
            event(5,"Ball Receipt*",player=players[2],time=3.5),
            event(6,"Pass",player=players[2],recipient=players[3],time=5),
            event(7,"Ball Receipt*",player=players[3],time=5.5)]
    events.extend(extra or [event(8,"Half End",time=7)])
    return build(events)


class MotifTests(unittest.TestCase):
    def test_all_five_patterns_are_identity_invariant(self):
        for players, pattern in [([10,20,10,20],"ABAB"),([10,20,10,30],"ABAC"),
                ([10,20,30,10],"ABCA"),([10,20,30,20],"ABCB"),([10,20,30,40],"ABCD")]:
            self.assertEqual(canonical_pattern(players),pattern)
            rows=build_motifs(three_pass(players),CONFIG)
            self.assertEqual(rows.loc[rows.order.eq(3)].pattern.tolist(),[pattern])
            self.assertEqual(len(rows.loc[rows.order.eq(2)]),2)

    def test_unknown_recipient_cannot_be_filled_from_intention(self):
        with self.assertRaises(ValueError): canonical_pattern([1,2,np.nan,3])
        t=three_pass([1,2,3,4])
        t["two_pass_sequences"].loc[t["two_pass_sequences"].b_event_id.eq("e6"),"memory_target_eligible"]=False
        self.assertTrue(build_motifs(t,CONFIG).order.eq(2).all())

    def test_intervening_opponent_control_breaks_three_passes(self):
        t=build([event(1,"Pass"),event(2,"Pass",player=2,recipient=3),
                 event(3,"Carry",player=9,team=2),event(4,"Pass",player=3,recipient=4)])
        rows=build_motifs(t,CONFIG)
        self.assertTrue(rows.order.eq(2).all())

    def test_failed_second_pass_does_not_create_third_pass_motif(self):
        t=build([event(1,"Pass"),event(2,"Pass",player=2,recipient=3,outcome="Incomplete"),
                 event(3,"Pass",player=3,recipient=4)])
        self.assertTrue(build_motifs(t,CONFIG).empty)

    def test_carry_links_preserved_and_direct_pass_policy_is_separate(self):
        t=build([event(1,"Pass"),event(2,"Carry",player=2),event(3,"Pass",player=2,recipient=3),
                 event(4,"Pass",player=3,recipient=4)])
        rows=build_motifs(t,CONFIG)
        triple=rows.loc[rows.order.eq(3)].iloc[0]
        self.assertEqual(triple.intervening_carries,1)
        self.assertFalse(triple.no_carry_dribble)

    def test_previous_event_five_seconds_is_not_between_passes_five_seconds(self):
        t=build([event(1,"Carry",time=0),event(2,"Pass",time=1),
                 event(3,"Ball Receipt*",player=2,time=1.5),event(4,"Carry",player=2,time=14),
                 event(5,"Pass",player=2,recipient=3,time=15),event(6,"Pass",player=3,recipient=4,time=17)])
        triple=build_motifs(t,CONFIG).loc[lambda f:f.order.eq(3)].iloc[0]
        self.assertTrue(triple.previous_event_5s)
        self.assertFalse(triple.between_passes_5s)

    def test_outcome_clock_follows_final_pass_not_initial_pass(self):
        t=three_pass([1,2,3,4],[event(8,"Shot",player=4,time=19),event(9,"Half End",time=22)])
        motifs=build_motifs(t,CONFIG)
        last=motifs.loc[motifs.order.eq(3)].end_sequence_id.iloc[0]
        labels=shot15_labels(t).set_index("sequence_id")
        self.assertEqual(last,"1:e6")
        self.assertEqual(labels.loc[last,"shot15_pass_start"],1)
        self.assertEqual(labels.loc["1:e2","shot15_pass_start"],0)

    def test_period_end_unknown_is_not_zero_shot(self):
        t=three_pass([1,2,3,4])
        labels=shot15_labels(t).set_index("sequence_id")
        self.assertTrue(np.isnan(labels.loc["1:e6","shot15_pass_start"]))

    def test_paper_own_goal_convention_is_separate_from_source_shots(self):
        t=three_pass([1,2,3,4],[event(8,"Own Goal Against",player=9,team=2,time=18),
                              event(9,"Half End",time=22)])
        row=shot15_labels(t).set_index("sequence_id").loc["1:e6"]
        self.assertEqual(row.shot15_pass_start,1)
        self.assertEqual(row.shot15_source_only,0)
        self.assertTrue(row.shot15_own_goal_counted)
        self.assertEqual(row.shot15_event_ids,"")

    def test_completion_features_never_read_pattern_or_future_outcomes(self):
        t=three_pass([1,2,3,4])
        rows=build_motifs(t,CONFIG)
        a=motif_features(rows,t["receiver_sequences"],"completion_geometry")
        changed=rows.copy()
        changed["pattern"]="ABAB";changed["R3"]=1;changed["player_ids"]="99;98;99;98"
        b=motif_features(changed,t["receiver_sequences"],"completion_geometry")
        pd.testing.assert_frame_equal(a,b)
        self.assertAlmostEqual(a.rx.iloc[0],1/3)  # Actual receipt x=40, not pass endpoint x=60.


if __name__=="__main__": unittest.main()
