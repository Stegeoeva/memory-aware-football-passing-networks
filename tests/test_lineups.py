"""Boundary cases for the on-field membership reconstruction."""
import unittest
import pandas as pd
from src.lineups import reconstruct_stints, change_groups, interval_graph, period_bounds, presentation_status


def event(index,kind,player=None,replacement=None,period=1,time="00:10:00.000"):
    row={"id":str(index),"index":index,"type":{"name":kind},"team":{"name":"A"},
         "period":period,"timestamp":time}
    if player is not None: row["player"]={"id":player}
    if kind=="Starting XI":
        row["timestamp"]="00:00:00.000"
        row["tactics"]={"lineup":[{"player":{"id":p},"position":{"id":p}} for p in range(1,12)]}
    if replacement is not None:
        row["substitution"]={"replacement":{"id":replacement}}
        row["position"]={"id":5}
    return row


class LineupTests(unittest.TestCase):
    def test_same_stoppage_multiple_substitutions_are_atomic(self):
        events=[event(1,"Starting XI"),event(10,"Substitution",10,12),
                event(11,"Tactical Shift"),event(12,"Substitution",11,13,time="00:10:12.000"),event(20,"Half End")]
        # Keep the intervening administrative event on the other team, since the
        # event is intentionally only a stub and has no tactics payload.
        events[2]["team"]["name"]="B"
        groups=change_groups(events,"A")
        self.assertEqual([len(g) for g in groups],[2])
        stints=reconstruct_stints(events,"A")
        self.assertEqual(len(stints),2)
        self.assertEqual(stints[1]["roster"],set(range(1,10))|{12,13})

    def test_football_event_between_changes_preserves_intermediate_lineup(self):
        events=[event(1,"Starting XI"),event(10,"Substitution",10,12),
                event(11,"Pass",12),event(12,"Substitution",11,13),event(20,"Half End")]
        stints=reconstruct_stints(events,"A")
        self.assertEqual(len(stints),3)
        self.assertIn(12,stints[1]["roster"])
        self.assertIn(11,stints[1]["roster"])
        self.assertNotIn(13,stints[1]["roster"])

    def test_temporary_absence_and_return_preserve_both_eleven_player_intervals(self):
        events=[event(1,"Starting XI"),event(5,"Player Off",7),event(8,"Player On",7),event(20,"Half End")]
        stints=reconstruct_stints(events,"A")
        self.assertEqual([len(s["roster"]) for s in stints],[11,10,11])
        self.assertEqual(stints[0]["roster"],stints[2]["roster"])

    def test_second_yellow_reduces_roster(self):
        card=event(10,"Foul Committed",7)
        card["foul_committed"]={"card":{"name":"Second Yellow"}}
        stints=reconstruct_stints([event(1,"Starting XI"),card,event(20,"Half End")],"A")
        self.assertEqual(len(stints[-1]["roster"]),10)
        self.assertNotIn(7,stints[-1]["roster"])

    def test_nominal_lineup_retains_temporary_absence_without_extra_window(self):
        events=[event(1,"Starting XI"),event(5,"Player Off",7),event(8,"Player On",7),event(20,"Half End")]
        stints=reconstruct_stints(events,"A",nominal=True)
        self.assertEqual(len(stints),1)
        self.assertEqual(stints[0]["roster"],set(range(1,12)))

    def test_injured_off_player_replacement_changes_nominal_lineup(self):
        events=[event(1,"Starting XI"),event(5,"Player Off",7),event(8,"Substitution",7,12),event(20,"Half End")]
        stints=reconstruct_stints(events,"A",nominal=True)
        self.assertEqual(len(stints),2)
        self.assertEqual(stints[0]["end"]["index"],8)
        self.assertEqual(stints[1]["roster"],set(range(1,12))-{7}|{12})

    def test_dismissal_while_temporarily_off_still_changes_nominal_lineup(self):
        card=event(8,"Bad Behaviour",7)
        card["bad_behaviour"]={"card":{"name":"Red Card"}}
        stints=reconstruct_stints([event(1,"Starting XI"),event(5,"Player Off",7),card,event(20,"Half End")],"A",nominal=True)
        self.assertEqual([len(s["roster"]) for s in stints],[11,10])

    def test_display_thresholds_do_not_merge_short_real_lineups(self):
        events=[event(1,"Starting XI"),event(10,"Substitution",10,12),event(11,"Pass",12),
                event(12,"Substitution",11,13),event(20,"Half End")]
        stints=reconstruct_stints(events,"A",nominal=True)
        self.assertEqual(len(stints),3)
        self.assertNotEqual(stints[0]["roster"],stints[1]["roster"])
        self.assertEqual(presentation_status({"duration_seconds":5,"network_passes":2}),
                         (False,"duration_below_threshold;passes_below_threshold"))
        self.assertEqual(presentation_status({"duration_seconds":300,"network_passes":20}),(True,""))
        self.assertFalse(presentation_status({"duration_seconds":500,"network_passes":0})[0])

    def test_period_clock_includes_added_time_but_no_half_time_break(self):
        events=[event(1,"Starting XI"),event(5,"Half End",time="00:47:12.000"),
                event(6,"Half Start",period=2,time="00:00:00.000"),event(20,"Half End",period=2,time="00:49:03.000")]
        bounds=period_bounds(events)
        self.assertEqual(bounds[2]["offset"],47*60+12)
        self.assertEqual(bounds[2]["duration"],49*60+3)

    def test_event_indices_assign_same_timestamp_passes_and_keep_zero_players(self):
        metadata={(1,p):{"name":str(p),"full_name":str(p),"team":"A","jersey":p} for p in range(1,13)}
        rows=[]
        for index,source in [(9,10),(11,12)]:
            rows.append({"match_id":1,"event_id":str(index),"index":index,"team":"A","period":1,
                         "regulation":True,"open_play":True,"completed":True,"has_recipient":True,
                         "self_pass":False,"network_pass":True,"passer_id":source,"recipient_id":1,
                         "x":50.,"y":40.,"end_x":20.,"end_y":40.})
        dataset={"passes":pd.DataFrame(rows),"players":metadata,"events":{1:[]}}
        base={"match_id":1,"team":"A","periods":[1],"roles":{p:5 for p in range(1,13)},
              "scope":"regulation","start_label":"start","end_label":"end","duration_seconds":10}
        before={**base,"segment_id":"before","start_index":1,"end_index":10,"roster":set(range(1,12))}
        after={**base,"segment_id":"after","start_index":10,"end_index":20,"roster":set(range(1,10))|{11,12}}
        for segment,source in [(before,10),(after,12)]:
            graph,passes=interval_graph(dataset,segment)
            self.assertEqual(len(graph),11)
            self.assertEqual(list(graph.edges),[(source,1)])
            self.assertEqual(len(passes),1)
            self.assertTrue(all("x" in a for _,a in graph.nodes(data=True)))
        empty={**base,"segment_id":"empty","start_index":15,"end_index":20,"roster":set(range(1,12))}
        graph,passes=interval_graph(dataset,empty)
        self.assertEqual(len(graph),11)
        self.assertEqual(graph.number_of_edges(),0)
        self.assertTrue(all(a["position_source"].startswith("tactical reference") for _,a in graph.nodes(data=True)))


if __name__=="__main__": unittest.main()
