import unittest
import numpy as np
import pandas as pd
from test_sequences import build,event
from test_motifs import three_pass
from src.motifs import build_motifs
from src.motif_xg import sequence_windows,bootstrap_ratio,fit_q


def extract(events,horizon="possession"):
    tables=build(events)
    windows,links=sequence_windows(tables)
    row=windows.loc[windows.sequence_id.eq("1:e1")&windows.horizon.eq(horizon)].iloc[0]
    return row,links.loc[links.sequence_id.eq("1:e1")&links.horizon.eq(horizon)]


class MotifXGTests(unittest.TestCase):
    def test_no_shot_in_closed_possession_has_observed_zero(self):
        row,_=extract([event(1,"Pass"),event(2,"Ball Receipt*",player=2),event(3,"Carry",player=9,team=2)])
        self.assertTrue(row.xg_observed)
        self.assertEqual((row.xg_sum,row.shot_any,row.shot_count),(0,0,0))

    def test_observed_period_end_closes_remaining_possession(self):
        row,_=extract([event(1,"Pass"),event(2,"Ball Receipt*",player=2),event(3,"Half End")])
        self.assertTrue(row.observation_complete)
        self.assertEqual(row.xg_sum,0)

    def test_unresolved_file_tail_not_automatic_zero(self):
        row,_=extract([event(1,"Pass"),event(2,"Ball Receipt*",player=2)])
        self.assertFalse(row.observation_complete)
        self.assertTrue(np.isnan(row.xg_sum))

    def test_multiple_shots_are_summed_without_capping_at_one(self):
        a=event(3,"Shot",player=2,time=4);a["shot"]["statsbomb_xg"]=.7
        b=event(4,"Shot",player=2,time=5);b["shot"]["statsbomb_xg"]=.6
        row,links=extract([event(1,"Pass"),event(2,"Ball Receipt*",player=2),a,b,event(5,"Half End",time=7)])
        self.assertEqual(row.shot_count,2)
        self.assertAlmostEqual(row.xg_sum,1.3)
        self.assertAlmostEqual(links.shot_xg.mean(),.65)

    def test_shots_before_receipt_do_not_enter_later_window(self):
        passed=event(1,"Pass",time=1);passed["related_events"]=["e3"]
        row,links=extract([passed,event(2,"Shot",player=2,time=1.5),
            event(3,"Ball Receipt*",player=2,time=2),event(4,"Half End",time=4)])
        self.assertEqual(row.xg_sum,0)
        self.assertTrue(links.empty)

    def test_confirmed_opponent_control_stops_even_when_provider_id_stays(self):
        row,links=extract([event(1,"Pass"),event(2,"Ball Receipt*",player=2),
            event(3,"Carry",player=9,team=2),event(4,"Shot",player=2),event(5,"Half End")])
        self.assertEqual(row.xg_sum,0)
        self.assertTrue(links.empty)

    def test_new_provider_possession_stops_same_team_later_shot(self):
        row,_=extract([event(1,"Pass"),event(2,"Ball Receipt*",player=2),
            event(3,"Shot",player=2,possession=2),event(4,"Half End",possession=2)])
        self.assertEqual(row.xg_sum,0)

    def test_own_restart_same_provider_possession_is_flagged(self):
        restart=event(3,"Pass",player=2,recipient=3);restart["pass"]["type"]={"name":"Free Kick"}
        row,_=extract([event(1,"Pass"),event(2,"Ball Receipt*",player=2),restart,
            event(4,"Shot",player=3),event(5,"Half End")])
        self.assertTrue(row.contains_restart)
        self.assertAlmostEqual(row.xg_sum,.2)

    def test_missing_shot_xg_is_not_imputed_as_zero(self):
        shot=event(3,"Shot",player=2);del shot["shot"]["statsbomb_xg"]
        row,_=extract([event(1,"Pass"),event(2,"Ball Receipt*",player=2),shot,event(4,"Half End")])
        self.assertEqual(row.missing_shot_xg,1)
        self.assertFalse(row.xg_observed)
        self.assertTrue(np.isnan(row.xg_sum))

    def test_own_goal_is_separate_from_source_shot_xg(self):
        row,_=extract([event(1,"Pass"),event(2,"Ball Receipt*",player=2),
            event(3,"Own Goal Against",player=9,team=2),event(4,"Half End")])
        self.assertEqual(row.own_goal_for_count,1)
        self.assertEqual((row.shot_any,row.shot_count,row.xg_sum),(0,0,0))

    def test_ten_second_cutoff_is_measured_from_final_receipt(self):
        events=[event(1,"Pass",time=1),event(2,"Ball Receipt*",player=2,time=2),
            event(3,"Shot",player=2,time=12.1),event(4,"Half End",time=15)]
        short,_=extract(events,"10s");remaining,_=extract(events)
        self.assertEqual(short.xg_sum,0)
        self.assertAlmostEqual(remaining.xg_sum,.2)

    def test_shot_quality_is_ratio_of_sums_not_mean_of_sequence_ratios(self):
        frame=pd.DataFrame({"pattern":["ABA","ABA"],"possession_key":["a","b"],"xg_observed":[True,True],
                            "xg_sum":[.2,.9],"shot_count":[2,1]})
        blocks=pd.DataFrame({"match_id":[1,2],"possession_key":["a","b"]})
        weights=np.array([[1,1],[2,0],[0,2]])
        point,draws=bootstrap_ratio(frame,pd.Series([True,True]),["ABA"],blocks,weights,"xg_sum","shot_count")
        self.assertAlmostEqual(point[0],1.1/3)
        np.testing.assert_allclose(draws[0],[1.1/3,.1,.9])

    def test_xg_expectation_model_is_not_capped_at_one(self):
        source=three_pass([1,2,3,4])
        frame=build_motifs(source,{"time_tolerance_seconds":.001,"maximum_gap_seconds":5})
        prediction,_,_,_=fit_q(frame,frame,np.full(len(frame),1.3),np.full(len(frame),1.3),
            source["receiver_sequences"],"initial_receipt",[1],False,600)
        np.testing.assert_allclose(prediction[1],1.3)


if __name__=="__main__":unittest.main()
