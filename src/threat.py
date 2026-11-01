"""Smoothed 12x8 event-only xT baseline, fitted on 48 disjoint EURO matches."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .analysis import MATCH_IDS,RESTARTS,ROOT

class ThreatModel:
    def __init__(self,values,metadata): self.values=np.asarray(values);self.metadata=metadata
    @staticmethod
    def cell(x,y):
        return int(np.clip(y/80*8,0,7))*12+int(np.clip(x/120*12,0,11))
    def value(self,x,y):
        return float(self.values[self.cell(x,y)]) if np.isfinite([x,y]).all() else np.nan

def fit_threat(root=ROOT):
    manifest=json.loads((root/'data/xt_training_manifest.json').read_text(encoding='utf-8'))
    assert set(manifest['training_match_ids']).isdisjoint(MATCH_IDS)
    n=96;move=np.zeros(n);shot=np.zeros(n);goal=np.zeros(n);transition=np.zeros((n,n));count=0
    audit=[]
    for mid in manifest['training_match_ids']:
        path=root/f'data/xt_training/events/{mid}.json'
        content=path.read_bytes()
        assert hashlib.sha256(content).hexdigest()==manifest['files'][str(mid)]['sha256']
        events=json.loads(content);before=count
        for e in events:
            if e['period'] not in [1,2,3,4] or 'location' not in e:continue
            kind=e['type']['name'];x,y=e['location'][:2];a=ThreatModel.cell(x,y)
            if kind=='Shot' and e['shot'].get('type',{}).get('name')=='Open Play':
                shot[a]+=1;goal[a]+=e['shot']['outcome']['name']=='Goal';count+=1
            elif kind in ['Pass','Carry']:
                d=e[kind.lower()]
                if kind=='Pass' and d.get('type',{}).get('name') in RESTARTS:continue
                move[a]+=1;count+=1
                if 'outcome' not in d and 'end_location' in d:
                    b=ThreatModel.cell(*d['end_location'][:2]);transition[a,b]+=1
        audit.append({'match_id':mid,'model_actions':count-before})
    # Explicit prior strengths fixed before viewing the held-out games.
    alpha=5.;beta=10.;total=move+shot
    p_shot=(shot+alpha*shot.sum()/total.sum())/(total+alpha)
    p_goal=(goal+beta*goal.sum()/shot.sum())/(shot+beta)
    destination=transition.sum(axis=0)/transition.sum()
    success_rate=transition.sum()/move.sum()
    T=(transition+alpha*success_rate*destination[None,:])/(move[:,None]+alpha)
    Q=(1-p_shot)[:,None]*T
    reward=p_shot*p_goal
    values=np.linalg.solve(np.eye(n)-Q,reward)
    assert np.all(values>=0) and np.all(values<=1)
    residual=float(np.abs(values-(reward+Q@values)).max())
    assert residual<1e-10
    metadata={'model':'Smoothed event-only xT baseline','grid_columns':12,'grid_rows':8,
        'training_matches':len(audit),'training_match_ids':manifest['training_match_ids'],
        'held_out_match_ids':MATCH_IDS,'training_actions':count,'move_attempts':int(move.sum()),
        'split_type':'Retrospective match-disjoint holdout, not a chronological forecasting split.',
        'open_play_shots':int(shot.sum()),'open_play_goals':int(goal.sum()),
        'action_transition_prior_strength':alpha,'shot_conversion_prior_strength':beta,
        'bellman_residual':residual,'equation':'V = p_shot*p_goal + diag(1-p_shot)*T*V',
        'failed_move_assumption':'Failed pass attempts transition to zero future scoring value in the baseline.',
        'limits':'Event-only 48-match tournament baseline; not calibrated VAEP/OBV, not causal, no off-ball state or conceding model.'}
    cells=pd.DataFrame([{'cell_id':i,'column':i%12,'row':i//12,'center_x':(i%12+.5)*10,
        'center_y':(i//12+.5)*10,'xt':values[i],'move_attempts':move[i],
        'shots':shot[i],'goals':goal[i],'p_shot_smoothed':p_shot[i],
        'p_goal_given_shot_smoothed':p_goal[i],'move_success_probability_smoothed':T[i].sum()} for i in range(n)])
    return ThreatModel(values,metadata),cells,pd.DataFrame(audit),T

def value_moves(moves,model):
    moves=moves.copy()
    moves['xt_start']=[model.value(x,y) for x,y in zip(moves.x,moves.y)]
    moves['xt_end']=[model.value(x,y) for x,y in zip(moves.end_x,moves.end_y)]
    moves['xt_delta_completed']=np.where(moves.completed,moves.xt_end-moves.xt_start,np.nan)
    moves['xt_attempt_value_zero_on_failure']=np.where(moves.completed,moves.xt_end-moves.xt_start,-moves.xt_start)
    return moves
