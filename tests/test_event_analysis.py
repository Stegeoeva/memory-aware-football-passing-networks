"""Temporal sequence and credit-assignment boundary cases."""
import unittest
import pandas as pd
from src.event_analysis import normalise_events,add_progression,turnover_tables,followups,possession_tables
from src.threat import value_moves

class LinearModel:
    def value(self,x,y): return x/120

def raw(i,kind,player=1,team='A',poss=1,period=1,x=40,end_x=50,recipient=2,outcome=None,xg=.2,key=None):
    e={'id':f'e{i}','index':i,'period':period,'timestamp':f'00:00:{i:06.3f}',
       'type':{'name':kind},'team':{'id':1 if team=='A' else 2,'name':team},
       'possession':poss,'possession_team':{'id':1,'name':'A'},
       'player':{'id':player,'name':str(player)},'location':[x,40],'duration':0.5}
    if kind in ['Pass','Carry','Shot','Dribble','Ball Receipt*']:
        key_name={'Ball Receipt*':'ball_receipt'}.get(kind,kind.lower())
        detail={'end_location':[end_x,40]}
        if kind=='Pass':detail.update(recipient={'id':recipient,'name':str(recipient)},length=abs(end_x-x))
        if kind=='Shot':detail.update(statsbomb_xg=xg,type={'name':'Open Play'},outcome={'name':'Saved'},key_pass_id=key)
        if outcome:detail['outcome']={'name':outcome}
        e[key_name]=detail
    return e

def fixture(raw_events):
    d={'events':{1:raw_events},'players':{}}
    e=normalise_events(d);m=value_moves(add_progression(e),LinearModel())
    losses,_=turnover_tables(e)
    f,motifs,summary=followups(e,m,LinearModel(),losses)
    return e,m,f,motifs

class TemporalTests(unittest.TestCase):
    def test_pressure_and_receiver_carry_do_not_break_real_two_pass_motif(self):
        e,m,f,motifs=fixture([raw(1,'Pass',recipient=2),raw(2,'Ball Receipt*',player=2),
            raw(3,'Pressure',player=9,team='B'),raw(4,'Carry',player=2,x=50,end_x=60),
            raw(5,'Pass',player=2,recipient=3,x=60,end_x=80),raw(6,'Shot',player=3,x=80)])
        a=f.loc[f.pass_event_id.eq('e1')].iloc[0]
        self.assertEqual(a.c_id,3)
        self.assertTrue(a.future_possession_shot)
        self.assertAlmostEqual(a.through_receiver_release_xt_delta,(80-40)/120)

    def test_intervening_teammate_action_is_not_skipped(self):
        _,_,f,motifs=fixture([raw(1,'Pass',recipient=2),raw(2,'Carry',player=3),raw(3,'Pass',player=2,recipient=4)])
        self.assertFalse(motifs.a_event_id.eq('e1').any())

    def test_intervening_opponent_control_is_not_skipped(self):
        _,_,f,motifs=fixture([raw(1,'Pass',recipient=2),raw(2,'Carry',player=9,team='B'),raw(3,'Pass',player=2,recipient=3)])
        self.assertFalse(motifs.a_event_id.eq('e1').any())
        self.assertEqual(f.loc[f.pass_event_id.eq('e1'),'next_3_actions_observed'].iloc[0],0)

    def test_possession_and_period_boundaries_are_respected(self):
        for keyword in [{'poss':2},{'period':2}]:
            _,_,f,motifs=fixture([raw(1,'Pass',recipient=2),raw(2,'Pass',player=2,recipient=3,**keyword)])
            self.assertFalse(motifs.a_event_id.eq('e1').any())

    def test_failed_receiver_pass_is_loss_only_when_opponent_controls_next(self):
        for team,expected in [('B',True),('A',False)]:
            _,_,f,_=fixture([raw(1,'Pass',recipient=2),raw(2,'Pass',player=2,recipient=3,outcome='Incomplete'),
                             raw(3,'Carry',player=9,team=team)])
            a=f.loc[f.pass_event_id.eq('e1')].iloc[0]
            self.assertEqual(bool(a.receiver_loss_observed),expected)
            self.assertFalse(a.b_next_pass_completed)

    def test_backward_release_can_have_positive_later_progression_and_shot(self):
        _,_,f,_=fixture([raw(1,'Pass',recipient=2,x=30,end_x=60),raw(2,'Pass',player=2,recipient=3,x=60,end_x=40),
            raw(3,'Carry',player=3,x=40,end_x=100),raw(4,'Shot',player=3,x=100)])
        a=f.loc[f.pass_event_id.eq('e1')].iloc[0]
        self.assertEqual(a.b_next_pass_direction,'backward')
        self.assertTrue(a.future_possession_shot)
        self.assertGreater(a.next_3_actions_max_xt_gain,0)

    def test_failed_move_does_not_count_as_entry(self):
        _,m,_,_=fixture([raw(1,'Pass',x=70,end_x=110,outcome='Incomplete')])
        self.assertFalse(m.final_third_entry.iloc[0]);self.assertFalse(m.box_entry.iloc[0])

    def test_xg_prefix_credits_are_full_and_never_include_later_participants(self):
        e,m,_,_=fixture([raw(1,'Pass',player=1,recipient=2),raw(2,'Shot',player=2,xg=.2,key='e1'),
            raw(3,'Pass',player=2,recipient=3),raw(4,'Shot',player=3,xg=.4,key='e3')])
        _,credits=possession_tables(e,m)
        first=credits.loc[credits.shot_event_id.eq('e2')]
        self.assertEqual(set(first.player_id),{1,2})
        self.assertTrue(first.xg_chain_prefix.eq(.2).all())
        self.assertTrue(first.xg_buildup_prefix.eq(0).all())
        self.assertEqual(credits.loc[credits.shot_event_id.eq('e4')&credits.player_id.eq(1),'xg_buildup_prefix'].iloc[0],.4)

if __name__=='__main__':unittest.main()
