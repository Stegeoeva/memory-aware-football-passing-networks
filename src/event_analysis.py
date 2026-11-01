"""Event-level exports, progression, pressure, possessions and future pass outcomes.

These are descriptive temporal associations, not causal ratings of passers.
All sequences respect period boundaries and the provider's possession identifiers.
"""
from collections import defaultdict
import math
import numpy as np
import pandas as pd
from .analysis import RESTARTS
from .lineups import seconds,period_bounds,clock_label,change_groups

DECISIONS={'Pass','Carry','Dribble','Shot','Dispossessed','Miscontrol','Clearance','Foul Won'}

def scope(period): return 'regulation' if period in [1,2] else 'extra_time'
def inside_box(x,y): return pd.notna(x) and pd.notna(y) and x>=102 and 18<=y<=62
def direction(dx): return 'forward' if dx>5 else 'backward' if dx< -5 else 'lateral'
def zone(x,y): return int(min(5,max(0,x//20)))+6*int(min(2,max(0,y//(80/3))))

def normalise_events(dataset):
    rows=[]
    for mid,events in dataset['events'].items():
        bounds=period_bounds(events)
        for e in events:
            if e['period'] not in [1,2,3,4]: continue
            kind=e['type']['name']; detail=e.get({'Pass':'pass','Carry':'carry','Shot':'shot','Dribble':'dribble',
                'Ball Receipt*':'ball_receipt','Interception':'interception','Goal Keeper':'goalkeeper'}.get(kind,''),{})
            location=e.get('location',[np.nan,np.nan]); end=detail.get('end_location',[np.nan,np.nan])
            pid=e.get('player',{}).get('id')
            metadata=dataset['players'].get((mid,pid),{})
            outcome=detail.get('outcome',{}).get('name')
            if outcome is None and kind in ['Pass','Carry','Ball Receipt*']: outcome='Complete'
            t=seconds(e)
            row={'match_id':mid,'event_id':e['id'],'index':e['index'],'period':e['period'],'scope':scope(e['period']),
                 'clock_seconds':t,'elapsed_seconds':bounds[e['period']]['offset']+t,'clock_label':clock_label(e),
                 'team':e['team']['name'],'team_id':e['team']['id'],
                 'possession_id':e['possession'],'possession_team':e['possession_team']['name'],
                 'player_id':pid,'player':metadata.get('name',e.get('player',{}).get('name')),
                 'position':e.get('position',{}).get('name'),'event_type':kind,
                 'x':location[0],'y':location[1],'end_x':end[0],'end_y':end[1],
                 'duration':e.get('duration',0.),'under_pressure':bool(e.get('under_pressure',False)),
                 'pressure_flag_present':'under_pressure' in e,'counterpress':bool(e.get('counterpress',False)),
                 'play_pattern':e.get('play_pattern',{}).get('name'),'outcome':outcome,
                 'recipient_id':detail.get('recipient',{}).get('id'),
                 'recipient':dataset['players'].get((mid,detail.get('recipient',{}).get('id')),{}).get('name',detail.get('recipient',{}).get('name')),
                 'pass_type':detail.get('type',{}).get('name') if kind=='Pass' else None,
                 'height':detail.get('height',{}).get('name'),'body_part':detail.get('body_part',{}).get('name'),
                 'length':detail.get('length',np.nan),'xg':detail.get('statsbomb_xg',0.),
                 'key_pass_id':detail.get('key_pass_id'),'shot_type':detail.get('type',{}).get('name') if kind=='Shot' else None,
                 'recovery_failed':bool(e.get('ball_recovery',{}).get('recovery_failure',False))}
            row['completed']=kind=='Carry' or (kind=='Pass' and outcome=='Complete')
            row['open_play']=kind!='Pass' or row['pass_type'] not in RESTARTS
            row['decision']=kind in DECISIONS
            row['controlled_touch']=kind in {'Pass','Carry','Dribble','Shot'} or (
                kind=='Ball Recovery' and not row['recovery_failed']) or (
                kind=='Interception' and outcome in {'Won','Success In Play'}) or (
                kind=='Goal Keeper' and detail.get('type',{}).get('name') in {'Collected','Keeper Sweeper','Smother'}
                and outcome in {'Success','Claim','Collected Twice','Won'})
            rows.append(row)
    return pd.DataFrame(rows).sort_values(['match_id','index']).reset_index(drop=True)

def add_progression(events):
    moves=events.loc[events.event_type.isin(['Pass','Carry'])].copy()
    moves['dx']=moves.end_x-moves.x
    moves['dy']=moves.end_y-moves.y
    moves['coordinate_distance']=np.hypot(moves.dx,moves.dy)
    moves['distance_to_goal_start']=np.hypot(120-moves.x,40-moves.y)
    moves['distance_to_goal_end']=np.hypot(120-moves.end_x,40-moves.end_y)
    moves['goal_distance_reduction']=moves.distance_to_goal_start-moves.distance_to_goal_end
    moves['direction']=moves.dx.map(direction)
    moves['progressive_10_units']=moves.completed & moves.dx.ge(10)
    moves['final_third_entry']=moves.completed & moves.x.lt(80) & moves.end_x.ge(80)
    moves['box_entry']=[bool(c and not inside_box(x,y) and inside_box(ex,ey)) for c,x,y,ex,ey in
                         zip(moves.completed,moves.x,moves.y,moves.end_x,moves.end_y)]
    moves['start_zone']=[zone(x,y) for x,y in zip(moves.x,moves.y)]
    moves['end_zone']=[zone(x,y) for x,y in zip(moves.end_x,moves.end_y)]
    return moves

def shot_tables(events,dataset):
    shots=events.loc[events.event_type.eq('Shot')].copy()
    shots['goal']=shots.outcome.eq('Goal')
    frames=[]
    for mid,raw in dataset['events'].items():
        for e in raw:
            if e['type']['name']!='Shot': continue
            for point in e['shot'].get('freeze_frame',[]):
                frames.append({'match_id':mid,'shot_event_id':e['id'],'period':e['period'],
                    'shooter_id':e['player']['id'],'shooter_team':e['team']['name'],
                    'player_id':point.get('player',{}).get('id'),'player':point.get('player',{}).get('name'),
                    'teammate':point['teammate'],'position':point.get('position',{}).get('name'),
                    'x':point['location'][0],'y':point['location'][1]})
    return shots,pd.DataFrame(frames)

def possession_tables(events,moves):
    possessions=[];credits=[]
    raw_by_id=events.set_index('event_id').to_dict('index')
    for (mid,p,poss),group in events.groupby(['match_id','period','possession_id'],sort=False):
        team=group.possession_team.iloc[0]
        own=group.loc[group.team.eq(team)]
        actions=own.loc[own.decision]
        if actions.empty: continue
        keys=set(actions.event_id)
        movement=moves.loc[moves.event_id.isin(keys)]
        passes=actions.loc[actions.event_type.eq('Pass')]
        shots=actions.loc[actions.event_type.eq('Shot')]
        first,last=actions.iloc[0],actions.iloc[-1]
        later=events.loc[events.match_id.eq(mid)&events.period.eq(p)&events['index'].gt(group['index'].max())]
        next_controller=later.possession_team.iloc[0] if not later.empty else None
        end_reason='period_end' if next_controller is None else 'opponent_possession' if next_controller!=team else 'same_team_new_possession'
        possessions.append({'match_id':mid,'period':p,'scope':scope(p),'possession_id':poss,'team':team,
            'start_index':int(group['index'].min()),'end_index':int(group['index'].max()),
            'start_clock':first.clock_label,'end_clock':last.clock_label,
            'duration_seconds':max(0.,last.clock_seconds+float(last.duration or 0)-first.clock_seconds),
            'start_x':first.x,'start_y':first.y,'end_x':last.end_x if pd.notna(last.end_x) else last.x,
            'end_y':last.end_y if pd.notna(last.end_y) else last.y,
            'actions':len(actions),'pass_attempts':len(passes),'completed_passes':int(passes.completed.sum()),
            'carries':int(actions.event_type.eq('Carry').sum()),'shots':len(shots),'xg':float(shots.xg.sum()),
            'goals':int(shots.outcome.eq('Goal').sum()),'final_third_entries':int(movement.final_third_entry.sum()),
            'box_entries':int(movement.box_entry.sum()),'end_reason':end_reason})
        # Temporal, per-shot prefix version. Each participant receives the full
        # shot xG once; credits overlap and must never be summed across players.
        for shot in shots.itertuples():
            participation=(actions.event_type.isin(['Carry','Shot'])
                           | (actions.event_type.eq('Pass')&actions.completed)
                           | (actions.event_type.eq('Dribble')&actions.outcome.eq('Complete')))
            prefix=actions.loc[actions['index'].le(shot.index)&actions.player_id.notna()&participation]
            participants=prefix.drop_duplicates('player_id')
            assister=raw_by_id.get(shot.key_pass_id,{}).get('player_id')
            for player in participants.itertuples():
                excluded=player.player_id in [shot.player_id,assister]
                credits.append({'match_id':mid,'period':p,'scope':scope(p),'possession_id':poss,
                    'shot_event_id':shot.event_id,'player_id':int(player.player_id),'player':player.player,
                    'team':team,'shot_xg':shot.xg,'xg_chain_prefix':shot.xg,
                    'xg_buildup_prefix':0. if excluded else shot.xg,'excluded_shooter_or_key_passer':excluded})
    return pd.DataFrame(possessions),pd.DataFrame(credits)

def wilson(success,n):
    if not n: return np.nan,np.nan
    z=1.96; phat=success/n; den=1+z*z/n
    center=(phat+z*z/(2*n))/den
    half=z*math.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/den
    return 100*(center-half),100*(center+half)

def pressure_table(events):
    passes=events.loc[events.event_type.eq('Pass')&events.open_play].copy()
    passes['origin_third']=pd.cut(passes.x,[-.001,40,80,120],labels=['defensive','middle','attacking'],right=False)
    passes.loc[passes.x.eq(120),'origin_third']='attacking'
    passes['pressure_group']=np.where(passes.under_pressure,'flagged pressure','not flagged')
    summary=[]
    for grouping in [['match_id','team','scope','pressure_group'],['match_id','team','scope','pressure_group','origin_third']]:
        for key,g in passes.groupby(grouping,observed=True):
            n=len(g); successes=int(g.completed.sum()); lo,hi=wilson(successes,n)
            summary.append({**dict(zip(grouping,key)),'stratification':'third' if 'origin_third' in grouping else 'overall',
                'attempts':n,'completed':successes,'completion_pct':100*successes/n,
                'wilson95_low_pct':lo,'wilson95_high_pct':hi,'mean_length':g.length.mean()})
    return passes,pd.DataFrame(summary)

def turnover_tables(events):
    rows=[];regains=[]
    for (mid,p),g in events.groupby(['match_id','period'],sort=False):
        controls=g.loc[g.controlled_touch]
        for row in g.itertuples():
            candidate=(row.event_type=='Pass' and not row.completed) or row.event_type in {'Miscontrol','Dispossessed'} or (row.event_type=='Dribble' and row.outcome=='Incomplete')
            if candidate:
                later=controls.loc[controls['index'].gt(row.index)]
                next_event=later.iloc[0] if not later.empty else None
                status='censored_period_end' if next_event is None else 'opponent_control' if next_event.team!=row.team else 'same_team_control'
                lost=next_event is not None and next_event.team!=row.team
                recover=later.loc[later.team.eq(row.team)] if lost else later.iloc[:0]
                recovery=recover.iloc[0] if not recover.empty else None
                rows.append({'match_id':mid,'period':p,'scope':scope(p),'event_id':row.event_id,'index':row.index,
                    'team':row.team,'player_id':row.player_id,'player':row.player,'event_type':row.event_type,
                    'clock_seconds':row.clock_seconds,'x':row.x,'y':row.y,'outcome':row.outcome,'status':status,
                    'next_control_event_id':None if next_event is None else next_event.event_id,
                    'next_control_team':None if next_event is None else next_event.team,
                    'seconds_to_next_control':np.nan if next_event is None else next_event.clock_seconds-row.clock_seconds,
                    'opponent_restart':bool(lost and next_event.event_type=='Pass' and not next_event.open_play),
                    'seconds_to_same_team_regain':np.nan if recovery is None else recovery.clock_seconds-row.clock_seconds,
                    'regain_within_10s':bool(recovery is not None and recovery.clock_seconds-row.clock_seconds<=10),
                    'opponent_next10s_xg':float(g.loc[g.team.ne(row.team)&g.event_type.eq('Shot')
                        &g['index'].gt(row.index)&g.clock_seconds.le(row.clock_seconds+10),'xg'].sum()),
                    'next10s_full_period_available':row.clock_seconds+10<=g.clock_seconds.max()})
            successful_regain=(row.event_type=='Ball Recovery' and not row.recovery_failed) or (row.event_type=='Interception' and row.outcome in ['Won','Success In Play'])
            if successful_regain:
                regains.append({'match_id':mid,'period':p,'scope':scope(p),'event_id':row.event_id,'index':row.index,
                    'team':row.team,'player_id':row.player_id,'player':row.player,'event_type':row.event_type,
                    'clock_seconds':row.clock_seconds,'x':row.x,'y':row.y,
                    'same_team_next10s_xg':float(g.loc[g.team.eq(row.team)&g.event_type.eq('Shot')
                        &g['index'].gt(row.index)&g.clock_seconds.le(row.clock_seconds+10),'xg'].sum()),
                    'same_team_next10s_pass_attempts':int((g.team.eq(row.team)&g.event_type.eq('Pass')
                        &g['index'].gt(row.index)&g.clock_seconds.le(row.clock_seconds+10)).sum()),
                    'next10s_full_period_available':row.clock_seconds+10<=g.clock_seconds.max()})
    return pd.DataFrame(rows),pd.DataFrame(regains)

def zone_tables(moves):
    passes=moves.loc[moves.event_type.eq('Pass')&moves.open_play&moves.completed]
    edges=passes.groupby(['match_id','team','scope','start_zone','end_zone']).agg(
        weight=('event_id','size'),mean_dx=('dx','mean'),mean_xt=('xt_delta_completed','mean')).reset_index()
    edges['within_zone']=edges.start_zone.eq(edges.end_zone)
    nodes=pd.DataFrame([{'zone_id':z,'column':z%6,'row':z//6,'center_x':(z%6+.5)*20,
                         'center_y':(z//6+.5)*80/3} for z in range(18)])
    return nodes,edges

def followups(events,moves,model,turnovers):
    """One row per completed open-play A->B; 1/3/5 actions and 5/10/20s futures."""
    eligible=moves.loc[moves.event_type.eq('Pass')&moves.open_play&moves.completed&moves.recipient_id.notna()&moves.player_id.ne(moves.recipient_id)]
    records=[];motifs=[]
    turnover_status=turnovers.set_index('event_id').status.to_dict() if not turnovers.empty else {}
    groups={key:g.sort_values('index') for key,g in events.groupby(['match_id','period','possession_id'],sort=False)}
    for a in eligible.itertuples():
        g=groups[(a.match_id,a.period,a.possession_id)]
        suffix=g.loc[g['index'].gt(a.index)]
        opposing_control=suffix.loc[suffix.controlled_touch&suffix.team.ne(a.team)]
        if not opposing_control.empty:
            suffix=suffix.loc[suffix['index'].lt(opposing_control.iloc[0]['index'])]
        future=suffix.loc[suffix.team.eq(a.team)&suffix.decision]
        decisions=future.to_dict('records')
        first=decisions[0] if decisions else None
        b=int(a.recipient_id)
        receipt=events.loc[events.match_id.eq(a.match_id)&events.period.eq(a.period)&events['index'].gt(a.index)
              &events.event_type.eq('Ball Receipt*')&events.player_id.eq(b)&events.team.eq(a.team)
              &events.clock_seconds.le(a.clock_seconds+max(float(a.duration),0)+2)]
        receiver_pressure=bool(receipt.iloc[0].under_pressure) if not receipt.empty else None
        reception_time=float(receipt.iloc[0].clock_seconds) if not receipt.empty else a.clock_seconds+max(float(a.duration),0)
        resolution=None
        # Carries and successful take-ons may precede B's release. Do not skip
        # a different teammate's action to manufacture an A->B->C sequence.
        for item in decisions:
            if item['player_id']!=b: break
            if item['event_type']=='Carry' or (item['event_type']=='Dribble' and item['outcome']=='Complete'): continue
            resolution=item; break
        row={'match_id':a.match_id,'period':a.period,'scope':a.scope,'possession_id':a.possession_id,
             'pass_event_id':a.event_id,'index':a.index,'team':a.team,'passer_id':int(a.player_id),'passer':a.player,
             'recipient_id':b,'recipient':a.recipient,'pass_under_pressure':a.under_pressure,
             'receiver_pressure_observed':receiver_pressure,'reception_x':a.end_x,'reception_y':a.end_y,
             'pass_length':a.length,'height':a.height,'initial_xt_delta':a.xt_delta_completed,
             'pass_start_seconds':a.clock_seconds,'reception_time_seconds':reception_time,
             'first_followup_actor_is_receiver':first is not None and first['player_id']==b,
             'receiver_resolution':'unobserved_or_censored' if resolution is None else resolution['event_type'],
             'receiver_resolution_event_id':None if resolution is None else resolution['event_id'],
             'receiver_resolution_outcome':None if resolution is None else resolution['outcome'],
             'receiver_loss_observed':False if resolution is None else turnover_status.get(resolution['event_id'])=='opponent_control',
             'b_next_pass_observed':resolution is not None and resolution['event_type']=='Pass',
             'b_next_pass_completed':False,'b_next_pass_direction':None,'b_next_pass_dx':np.nan,
             'b_next_pass_xt_delta':np.nan,'c_id':None,'c_name':None,
             'through_receiver_release_xt_delta':np.nan,
             'future_possession_xg':float(future.loc[future.event_type.eq('Shot'),'xg'].sum()),
             'future_possession_shot':bool(future.event_type.eq('Shot').any())}
        if resolution is not None and resolution['event_type']=='Pass':
            r=resolution; completed=bool(r['completed']); dx=r['end_x']-r['x']
            row.update(b_next_pass_completed=completed,b_next_pass_direction=direction(dx),b_next_pass_dx=dx,
                       b_next_pass_xt_delta=model.value(r['end_x'],r['end_y'])-model.value(r['x'],r['y']) if completed else np.nan,
                       c_id=r['recipient_id'],c_name=r['recipient'])
            if completed:
                row['through_receiver_release_xt_delta']=model.value(r['end_x'],r['end_y'])-model.value(a.x,a.y)
            motifs.append({'match_id':a.match_id,'period':a.period,'scope':a.scope,'possession_id':a.possession_id,
                'a_event_id':a.event_id,'b_event_id':r['event_id'],'team':a.team,'a_id':int(a.player_id),'a':a.player,
                'b_id':b,'b':a.recipient,'c_id':r['recipient_id'],'c':r['recipient'],
                'b_completed':completed,'b_direction':direction(dx),'b_dx':dx,
                'initial_xt_delta':a.xt_delta_completed,'b_xt_delta':row['b_next_pass_xt_delta'],
                'seconds_between_pass_starts':r['clock_seconds']-a.clock_seconds,
                'future_possession_xg':row['future_possession_xg']})
        for n in [1,3,5]:
            subset=decisions[:n]
            prefix=f'next_{n}_actions'
            row[prefix+'_observed']=len(subset)
            row[prefix+'_complete_horizon']=len(subset)==n
            row[prefix+'_xg']=sum(r['xg'] for r in subset if r['event_type']=='Shot')
            row[prefix+'_shot']=any(r['event_type']=='Shot' for r in subset)
            endpoints=[(r['end_x'],r['end_y']) if r['completed'] else (r['x'],r['y']) for r in subset]
            valid=[xy for xy in endpoints if pd.notna(xy[0]) and pd.notna(xy[1])]
            row[prefix+'_max_forward_gain']=max([0.]+[x-a.end_x for x,y in valid])
            row[prefix+'_max_xt_gain']=max([0.]+[model.value(x,y)-model.value(a.end_x,a.end_y) for x,y in valid])
        for t in [5,10,20]:
            subset=[r for r in decisions if r['clock_seconds']<=reception_time+t]
            row[f'next_{t}s_xg']=sum(r['xg'] for r in subset if r['event_type']=='Shot')
            row[f'next_{t}s_shot']=any(r['event_type']=='Shot' for r in subset)
            row[f'next_{t}s_actions']=len(subset)
            period_end=events.loc[events.match_id.eq(a.match_id)&events.period.eq(a.period),'clock_seconds'].max()
            row[f'next_{t}s_full_period_available']=reception_time+t<=period_end
        records.append(row)
    frame=pd.DataFrame(records)
    if frame.empty:
        return frame,pd.DataFrame(columns=['a_event_id','b_event_id']),pd.DataFrame()
    summaries=[]
    for grouping in [['match_id','team','scope','passer_id','passer'],['team','scope','passer_id','passer']]:
        for key,g in frame.groupby(grouping,dropna=False):
            observed=g.loc[g.b_next_pass_observed]; complete=observed.loc[observed.b_next_pass_completed]
            resolved=g.receiver_resolution.ne('unobserved_or_censored').sum()
            summaries.append({**dict(zip(grouping,key)),'aggregation':'per_match' if 'match_id' in grouping else 'all_three_matches',
                'completed_initial_passes':len(g),'receiver_resolved_n':int(resolved),
                'receiver_loss_n':int(g.receiver_loss_observed.sum()),
                'receiver_loss_pct_of_resolved':100*g.receiver_loss_observed.sum()/resolved if resolved else np.nan,
                'b_next_pass_attempts':len(observed),'b_next_pass_completed_n':len(complete),
                'b_next_pass_completion_pct':100*len(complete)/len(observed) if len(observed) else np.nan,
                'b_forward_pct_of_completed':100*complete.b_next_pass_direction.eq('forward').mean(),
                'b_backward_pct_of_completed':100*complete.b_next_pass_direction.eq('backward').mean(),
                'mean_initial_xt_delta':g.initial_xt_delta.mean(),'mean_b_next_pass_xt_delta':complete.b_next_pass_xt_delta.mean(),
                'subsequent_shot_pct':100*g.future_possession_shot.mean(),
                'mean_subsequent_possession_xg':g.future_possession_xg.mean()})
    motif_frame=pd.DataFrame(motifs) if motifs else pd.DataFrame(columns=['a_event_id','b_event_id'])
    return frame,motif_frame,pd.DataFrame(summaries)

def before_after(events,moves,turnovers,dataset):
    triggers=[]
    for mid,raw in dataset['events'].items():
        for team in {e['team']['name'] for e in raw}:
            for group in change_groups(raw,team):
                if group[0]['type']['name']=='Substitution': triggers.append((mid,group[0],'substitution_batch',team,len(group)))
        for e in raw:
            if e['type']['name']=='Shot' and e['shot']['outcome']['name']=='Goal':
                triggers.append((mid,e,'goal',e['team']['name'],1))
    rows=[]
    for mid,e,kind,trigger_team,batch in triggers:
        p=e['period']; t=seconds(e)
        group=events.loc[events.match_id.eq(mid)&events.period.eq(p)]
        if group.empty: continue
        end=group.clock_seconds.max()
        for team in group.team.unique():
            for side,lo,hi in [('before',max(0,t-300),t),('after',t,min(end,t+300))]:
                subset=group.loc[group.team.eq(team)&group.clock_seconds.ge(lo)&group.clock_seconds.le(hi)]
                subset=subset.loc[subset['index'].lt(e['index']) if side=='before' else subset['index'].gt(e['index'])]
                passes=subset.loc[subset.event_type.eq('Pass')&subset.open_play]
                shots=subset.loc[subset.event_type.eq('Shot')]
                movement=moves.loc[moves.event_id.isin(subset.event_id)]
                losses=turnovers.loc[turnovers.event_id.isin(subset.event_id)&turnovers.status.eq('opponent_control')]
                duration=hi-lo
                rows.append({'trigger_event_id':e['id'],'match_id':mid,'period':p,'scope':scope(p),
                    'trigger':kind,'trigger_team':trigger_team,'batch_size':batch,'trigger_clock':clock_label(e),
                    'team':team,'side':side,'window_seconds':duration,'open_play_attempts':len(passes),
                    'completed_open_play':int(passes.completed.sum()),
                    'completed_passes_per_minute':60*passes.completed.sum()/duration if duration else np.nan,
                    'shots':len(shots),'xg':float(shots.xg.sum()),'progressive_actions':int(movement.progressive_10_units.sum()),
                    'final_third_entries':int(movement.final_third_entry.sum()),'box_entries':int(movement.box_entry.sum()),
                    'confirmed_opponent_control_losses':len(losses)})
    return pd.DataFrame(rows)
