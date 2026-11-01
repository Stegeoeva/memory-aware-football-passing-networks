"""Complete event-data extension and a temporal follow-up study of A->B->C."""
import html
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import networkx as nx

from .analysis import ROOT,MATCH_IDS,COLORS,INK,load_dataset,configure_style,pitch,save_figure
from .event_analysis import (normalise_events,add_progression,shot_tables,possession_tables,pressure_table,
                             turnover_tables,zone_tables,followups,before_after)
from .threat import fit_threat,value_moves

TABLE_UNITS={
 'events':'One row per original event; all periods 1-4, both teams. Defensive events are not ball-control actions.',
 'shots':'One row per shot, including source xG and key-pass ID.',
 'shot_freeze_frames':'One row per recorded player in a shot snapshot; not a complete tracking frame.',
 'possessions':'One row per provider possession, period and controlling team; only that team\'s actions are counted.',
 'xg_chain_credits':'One player/shot-prefix credit. Full shot xG once per participant, never divided among players.',
 'xg_player_credits':'Aggregated overlapping xGChain-prefix and xGBuildup-prefix credits; not additive across players.',
 'progression_actions':'One pass attempt or carry, including failures and restart flags. Units are StatsBomb coordinates.',
 'progression_players':'Per-player movement totals; open-play passes and carries only.',
 'pressure_passes':'All open-play attempts, including unsuccessful passes. Not flagged does not guarantee absence of pressure.',
 'pressure_summary':'Completion rates and Wilson 95% intervals, overall and by origin third.',
 'turnover_candidates':'Failed passes/take-ons, dispossessions and miscontrols; actual next control identifies confirmed losses.',
 'regains':'Successful ball recoveries and interceptions, with locations.',
 'zone_nodes':'18 pitch zones: 6 columns and 3 rows.',
 'zone_edges':'Completed open-play passes between zones; within-zone passes retained.',
 'before_after_windows':'Five-minute windows around goals/substitution batches, clipped at period boundaries; trigger shot excluded.',
 'followup_passes':'One completed open-play A->B, with B\'s resolution, next 1/3/5 actions and 5/10/20-second futures.',
 'two_pass_motifs':'Observed A->B (optional B carries/take-ons) -> B->C attempt. A different teammate\'s action ends the motif.',
 'higher_order_edges':'Observed successful transitions (A,B)->(B,C), aggregated by match, team and phase.',
 'passer_followup_summary':'Per-passer descriptive downstream outcomes, with explicit denominators and coverage.',
 'followup_context_summary':'Descriptive follow-up outcomes stratified by reception third, pressure flag and initial passer.',
 'xt_cells':'96 cells of the 12x8 smoothed event-only expected-threat baseline.',
 'xt_training_matches':'48 non-evaluation EURO 2024 games and their training action counts.',
 'xt_transitions':'Conditional move transition probabilities; row sums may be <1 because failures have zero future value.',
}

def compute_extended(root=ROOT):
    dataset=load_dataset(root);events=normalise_events(dataset)
    model,cells,train,transitions=fit_threat(root)
    moves=value_moves(add_progression(events),model)
    shots,frames=shot_tables(events,dataset)
    possessions,credits=possession_tables(events,moves)
    pressure,pressure_summary=pressure_table(events)
    losses,regains=turnover_tables(events)
    follow,motifs,passer_summary=followups(events,moves,model,losses)
    windows=before_after(events,moves,losses,dataset)
    nodes,edges=zone_tables(moves)
    movement=moves.loc[moves.open_play]
    player_progress=movement.groupby(['match_id','team','scope','player_id','player'],dropna=False).agg(
        movement_actions=('event_id','size'),completed_actions=('completed','sum'),
        progressive_10_units=('progressive_10_units','sum'),final_third_entries=('final_third_entry','sum'),
        box_entries=('box_entry','sum'),completed_xt_sum=('xt_delta_completed','sum'),
        xt_with_zero_on_failure_sum=('xt_attempt_value_zero_on_failure','sum')).reset_index()
    credit_summary=credits.groupby(['match_id','team','scope','player_id','player']).agg(
        shot_prefix_participations=('shot_event_id','size'),xg_chain_prefix=('xg_chain_prefix','sum'),
        xg_buildup_prefix=('xg_buildup_prefix','sum')).reset_index()
    successful=motifs.loc[motifs.b_completed&motifs.c_id.notna()]
    higher=successful.groupby(['match_id','team','scope','a_id','a','b_id','b','c_id','c']).agg(
        count=('a_event_id','size'),mean_b_dx=('b_dx','mean'),mean_b_xt=('b_xt_delta','mean'),
        mean_following_possession_xg=('future_possession_xg','mean')).reset_index()
    context=follow.copy()
    context['reception_third']=pd.cut(context.reception_x,[-.001,40,80,120.001],labels=['defensive','middle','attacking'],right=False)
    context['receiver_pressure_group']=context.receiver_pressure_observed.map({True:'flagged',False:'not flagged'}).fillna('unobserved')
    context_summary=context.groupby(['match_id','team','scope','passer_id','passer','recipient_id','recipient',
        'reception_third','receiver_pressure_group'],observed=True).agg(
        initial_passes=('pass_event_id','size'),receiver_loss_count=('receiver_loss_observed','sum'),
        next_b_pass_attempts=('b_next_pass_observed','sum'),next_b_pass_completions=('b_next_pass_completed','sum'),
        future_shot_count=('future_possession_shot','sum'),mean_future_possession_xg=('future_possession_xg','mean')).reset_index()
    tables={'events':events,'shots':shots,'shot_freeze_frames':frames,'possessions':possessions,
        'xg_chain_credits':credits,'xg_player_credits':credit_summary,'progression_actions':moves,
        'progression_players':player_progress,'pressure_passes':pressure,'pressure_summary':pressure_summary,
        'turnover_candidates':losses,'regains':regains,'zone_nodes':nodes,'zone_edges':edges,
        'before_after_windows':windows,'followup_passes':follow,'two_pass_motifs':motifs,
        'higher_order_edges':higher,'passer_followup_summary':passer_summary,'followup_context_summary':context_summary,
        'xt_cells':cells,'xt_training_matches':train,'xt_transitions':pd.DataFrame(transitions)}
    return dataset,tables,model

def validate_extended(dataset,tables,model):
    e=tables['events'];m=tables['progression_actions'];f=tables['followup_passes'];motifs=tables['two_pass_motifs']
    assert len(e)==11097 and len(tables['shots'])==82
    assert e.event_type.eq('Carry').sum()==2597
    assert e.event_id.is_unique
    assert len(f)==2401 and f.pass_event_id.is_unique
    assert int(f.scope.eq('regulation').sum())==2160
    assert tables['shots'].xg.between(0,1).all()
    assert set(model.metadata['training_match_ids']).isdisjoint(MATCH_IDS)
    assert model.metadata['training_matches']==48
    assert model.metadata['bellman_residual']<1e-10
    assert not tables['xg_chain_credits'].duplicated(['shot_event_id','player_id']).any()
    assert tables['xg_chain_credits'].xg_buildup_prefix.le(tables['xg_chain_credits'].xg_chain_prefix).all()
    lookup=e.set_index('event_id')
    for row in motifs.itertuples():
        a,b=lookup.loc[row.a_event_id],lookup.loc[row.b_event_id]
        assert a.recipient_id==b.player_id and a.team==b.team
        assert a.match_id==b.match_id and a.period==b.period and a.possession_id==b.possession_id
        assert a['index']<b['index']
        between=e.loc[e.match_id.eq(a.match_id)&e['index'].gt(a['index'])&e['index'].lt(b['index'])]
        assert not (between.controlled_touch&between.team.ne(a.team)).any()
    for row in f.itertuples():
        assert row.next_1_actions_observed<=row.next_3_actions_observed<=row.next_5_actions_observed
        assert row.next_1_actions_xg<=row.next_3_actions_xg+1e-12<=row.next_5_actions_xg+1e-12
    for _,p in tables['pressure_summary'].iterrows():
        assert p.completed<=p.attempts and 0<=p.completion_pct<=100
    assert int(tables['zone_edges'].weight.sum())==int((m.event_type.eq('Pass')&m.open_play&m.completed).sum())
    return {'status':'passed','source_events':len(e),'shots':len(tables['shots']),
        'carries':int(e.event_type.eq('Carry').sum()),'followup_initial_passes':len(f),
        'observed_two_pass_motifs':len(motifs),'shot_snapshot_player_records':len(tables['shot_freeze_frames']),
        'training_matches_excluding_evaluation':48,'checks':['unique source and initial-pass IDs','exact regulation pass count',
        'same-period same-possession motifs','no skipped opponent control','no duplicate player-shot credits',
        'non-overlapping training/evaluation IDs','xT Bellman residual','zone pass accounting','pressure denominators','nested future horizons']}

def create_extended_figures(dataset,tables,model,out):
    configure_style();figures=[];matches=list(dataset['matches'].itertuples())
    def save(fig,stem,title,note):
        path=save_figure(fig,out/stem,title,note,footer='Data: StatsBomb Open Data | EURO 2024 | Scope labelled per figure')
        figures.append({'file':path.name,'title':title,'note':note})
    shots=tables['shots'];moves=tables['progression_actions'];events=tables['events']
    fig,axes=plt.subplots(3,2,figsize=(17,15))
    for i,match in enumerate(matches):
        for ax,team in zip(axes[i],[match.home,match.away]):
            pitch(ax);s=shots.loc[shots.match_id.eq(match.match_id)&shots.team.eq(team)&shots.scope.eq('regulation')]
            ax.scatter(s.x,s.y,s=50+1200*s.xg,c=COLORS[team],alpha=.65,edgecolor='white')
            goal=s.loc[s.goal];ax.scatter(goal.x,goal.y,s=80+1200*goal.xg,facecolor='none',edgecolor=INK,lw=1.8)
            ax.set_title(f'{team} vs {match.away if team==match.home else match.home} | {len(s)} shots | {s.xg.sum():.2f} xG',loc='left',fontsize=11)
    save(fig,'01_shot_maps','Shot locations and chance quality','Regulation only. Bubble area increases with source StatsBomb xG; dark rings mark goals. xG is supplied by StatsBomb.')
    fig,axes=plt.subplots(3,1,figsize=(15,11))
    for ax,match in zip(axes,matches):
        for team in [match.home,match.away]:
            s=shots.loc[shots.match_id.eq(match.match_id)&shots.team.eq(team)].sort_values('index')
            duration=events.loc[events.match_id.eq(match.match_id),'elapsed_seconds'].max()/60
            x=np.r_[0,s.elapsed_seconds.to_numpy()/60,duration];y=np.r_[0,s.xg.cumsum().to_numpy(),s.xg.sum()]
            ax.step(x,y,where='post',color=COLORS[team],label=team,lw=2)
        ax.set_title(f'Spain vs {match.away}',loc='left',weight='bold');ax.set_ylabel('Cumulative xG');ax.legend(frameon=False,ncol=2)
    axes[-1].set_xlabel('Elapsed period time, including added time; half-time breaks excluded')
    save(fig,'02_xg_timeline','When were chances created?','All periods, including extra time in Spain-Germany. Steps show shots, not a continuous probability of scoring.')
    fig,axes=plt.subplots(3,2,figsize=(17,15))
    poss=tables['possessions']
    for i,match in enumerate(matches):
        selected=poss.loc[poss.match_id.eq(match.match_id)&poss.shots.gt(0)].nlargest(2,'xg')
        for ax,(_,p) in zip(axes[i],selected.iterrows()):
            pitch(ax);ms=moves.loc[moves.match_id.eq(match.match_id)&moves.period.eq(p.period)&moves.possession_id.eq(p.possession_id)&moves.team.eq(p.team)&moves.completed]
            for j,r in enumerate(ms.itertuples()):
                ax.annotate('',xy=(r.end_x,r.end_y),xytext=(r.x,r.y),arrowprops={'arrowstyle':'->','lw':1.4,'color':COLORS[p.team],'alpha':.65,'linestyle':'--' if r.event_type=='Carry' else '-'})
                ax.text(r.x,r.y,str(j+1),fontsize=7)
            ax.set_title(f'{p.team} | possession {p.possession_id} | {p.scope} | {p.xg:.2f} xG',loc='left',fontsize=10)
    save(fig,'03_possession_examples','Actual possession sequences ending in shots','Top two shot-producing possessions by total xG per match. Solid: completed pass; dashed: carry. Numbers preserve event order.')
    fig,axes=plt.subplots(1,3,figsize=(17,7))
    for ax,match in zip(axes,matches):
        for j,team in enumerate([match.home,match.away]):
            g=moves.loc[moves.match_id.eq(match.match_id)&moves.team.eq(team)&moves.scope.eq('regulation')&moves.open_play]
            values=[g.progressive_10_units.sum(),g.final_third_entry.sum(),g.box_entry.sum()]
            x=np.arange(3)+(j-.5)*.35;ax.bar(x,values,width=.33,color=COLORS[team],label=team)
        ax.set_xticks(range(3),['Forward 10+ units','Final-third entries','Box entries'],rotation=25,ha='right',fontsize=9)
        ax.set_title(f'Spain vs {match.away}',loc='left');ax.legend(frameon=False);ax.set_ylabel('Completed actions')
    save(fig,'04_progression','Who moves the ball into advanced positions?','Regulation; open-play passes and carries. Forward 10+ units is our explicit operational threshold, not a universal progressive-pass definition.')
    fig,axes=plt.subplots(3,2,figsize=(16,12))
    ps=tables['pressure_summary']
    for i,match in enumerate(matches):
        for ax,team in zip(axes[i],[match.home,match.away]):
            g=ps.loc[ps.match_id.eq(match.match_id)&ps.team.eq(team)&ps.scope.eq('regulation')&ps.stratification.eq('overall')].sort_values('pressure_group')
            ax.bar(g.pressure_group,g.completion_pct,color=COLORS[team],alpha=.7)
            for j,r in enumerate(g.itertuples()):
                ax.errorbar(j,r.completion_pct,yerr=[[r.completion_pct-r.wilson95_low_pct],[r.wilson95_high_pct-r.completion_pct]],color=INK,capsize=4)
                ax.text(j,r.completion_pct+5,f'n={r.attempts}',ha='center',fontsize=9)
            ax.set_ylim(0,108);ax.set_title(f'{team} vs {match.away if team==match.home else match.home}',loc='left');ax.set_ylabel('Completion (%)')
    save(fig,'05_pressure','Passing completion with and without a pressure flag','All open-play attempts in regulation. Error bars: Wilson 95% intervals. Groups differ in location and difficulty; this is not an adjusted pressure effect.')
    fig,axes=plt.subplots(3,2,figsize=(17,15))
    loss=tables['turnover_candidates'];reg=tables['regains']
    for i,match in enumerate(matches):
        for ax,team in zip(axes[i],[match.home,match.away]):
            pitch(ax);a=loss.loc[loss.match_id.eq(match.match_id)&loss.team.eq(team)&loss.scope.eq('regulation')&loss.status.eq('opponent_control')]
            b=reg.loc[reg.match_id.eq(match.match_id)&reg.team.eq(team)&reg.scope.eq('regulation')]
            ax.scatter(a.x,a.y,c='#bf5d50',marker='x',s=30,alpha=.65,label='Loss candidate followed by opponent control')
            ax.scatter(b.x,b.y,c='#358a74',s=25,alpha=.55,label='Successful recovery/interception')
            ax.set_title(f'{team} | {len(a)} confirmed losses | {len(b)} regains',loc='left',fontsize=10)
    save(fig,'06_losses_regains','Where is control lost and recovered?','Regulation. Red crosses: failed actions followed by opponent control (including opponent restarts). Green: successful recoveries/interceptions. Neither is a complete pressing model.')
    fig,axes=plt.subplots(3,2,figsize=(17,15))
    zones=tables['zone_nodes'];edges=tables['zone_edges']
    for i,match in enumerate(matches):
        for ax,team in zip(axes[i],[match.home,match.away]):
            pitch(ax)
            for x in range(20,120,20):ax.plot([x,x],[0,80],color='#ccd6cf',lw=.6,ls='--')
            for y in [80/3,160/3]:ax.plot([0,120],[y,y],color='#ccd6cf',lw=.6,ls='--')
            g=edges.loc[edges.match_id.eq(match.match_id)&edges.team.eq(team)&edges.scope.eq('regulation')]
            G=nx.DiGraph();G.add_nodes_from(range(18));G.add_weighted_edges_from((int(r.start_zone),int(r.end_zone),r.weight) for r in g.itertuples() if not r.within_zone)
            pos={int(r.zone_id):(r.center_x,r.center_y) for r in zones.itertuples()}
            nx.draw_networkx_edges(G,pos,ax=ax,width=[.10*d['weight'] for _,_,d in G.edges(data=True)],alpha=.5,edge_color=COLORS[team],arrowsize=9,connectionstyle='arc3,rad=.12')
            for z,(x,y) in pos.items():
                ax.scatter(x,y,s=70,c=COLORS[team],zorder=3);ax.text(x,y,str(z),ha='center',va='center',fontsize=6,color='white',zorder=4)
            ax.set_title(f'{team} vs {match.away if team==match.home else match.home}',loc='left',fontsize=11)
    save(fig,'07_zone_networks','Passing routes between 18 pitch zones','Regulation, completed open-play passes. Every between-zone link shown. Within-zone passes are retained in the CSV but not drawn as loops.')
    windows=tables['before_after_windows'];fig,axes=plt.subplots(1,2,figsize=(17,8))
    for ax,trigger in zip(axes,['substitution_batch','goal']):
        w=windows.loc[windows.trigger.eq(trigger)&windows.scope.eq('regulation')]
        pivot=w.pivot(index=['trigger_event_id','team','match_id'],columns='side',values=['completed_passes_per_minute','window_seconds'])
        valid=pivot.loc[(pivot['window_seconds']['before']>=120)&(pivot['window_seconds']['after']>=120)]
        for key,r in valid.iterrows():
            team=key[1];ax.scatter(r['completed_passes_per_minute']['before'],r['completed_passes_per_minute']['after'],c=COLORS[team],s=45,alpha=.7)
        ax.plot([0,12],[0,12],ls='--',color='#91a0ad');ax.set(xlim=(0,12),ylim=(0,12),xlabel='Before: completed passes / minute',ylabel='After: completed passes / minute')
        ax.set_title(trigger.replace('_',' ').title(),loc='left',weight='bold')
    save(fig,'08_before_after','Passing rate around substitutions and goals','Regulation, up to five minutes per side; windows with <2 minutes on either side excluded from this plot. Goal shot excluded. Overlapping windows and confounding prevent causal attribution.')
    fig,ax=plt.subplots(figsize=(13,9));cells=tables['xt_cells'];matrix=model.values.reshape(8,12)
    im=ax.imshow(matrix,extent=(0,120,80,0),cmap='YlOrRd',aspect='equal')
    for r in cells.itertuples():ax.text(r.center_x,r.center_y,f'{r.xt:.3f}',ha='center',va='center',fontsize=7,color='white' if r.xt>matrix.max()*.6 else INK)
    ax.set(xlabel='Attacking direction: left to right',ylabel='StatsBomb y');fig.colorbar(im,ax=ax,label='Model probability of a future goal')
    save(fig,'09_xt_surface','Expected threat learned from 48 other EURO matches','Smoothed 12x8 event-only baseline. The three evaluation games were excluded from training. Grid and priors are documented; this is not a calibrated VAEP/OBV model.')
    follow=tables['followup_passes'];fig,ax=plt.subplots(figsize=(12,9))
    for team in ['Spain','Germany','France','England']:
        g=follow.loc[follow.team.eq(team)&follow.b_next_pass_completed]
        ax.scatter(g.initial_xt_delta,g.b_next_pass_xt_delta,s=10,alpha=.30,c=COLORS[team],label=team)
    ax.axhline(0,color='#9daab6',lw=.8);ax.axvline(0,color='#9daab6',lw=.8)
    ax.set(xlabel='A -> B: immediate spatial xT change',ylabel="B's next completed pass: xT change")
    ax.legend(frameon=False)
    save(fig,'10_a_to_b_to_c','What happens after the initial pass?','All periods, observed B releases only; B may carry before passing. Each point is a real observed sequence.\nDownstream outcome is shared context, not causal credit to A; no paths are inferred from the aggregate graph.')
    summary=tables['passer_followup_summary'];g=summary.loc[summary.team.eq('Spain')&summary.scope.eq('regulation')&summary.aggregation.eq('all_three_matches')].nlargest(10,'completed_initial_passes').sort_values('completed_initial_passes')
    fig,axes=plt.subplots(1,3,figsize=(19,9))
    for ax,column,title in zip(axes,['b_forward_pct_of_completed','receiver_loss_pct_of_resolved','subsequent_shot_pct'],["B next pass forward (%)","Observed receiver loss (%)","Later same-possession shot (%)"]):
        ax.barh(g.passer,g[column],color=COLORS['Spain'],alpha=.8);ax.set_title(title,loc='left',fontsize=11);ax.set_xlabel('Percent')
        for y,r in enumerate(g.itertuples()):ax.text(getattr(r,column)+.4,y,f'n={r.completed_initial_passes}',va='center',fontsize=8)
        ax.set_xlim(0,max(5,float(g[column].max())*1.3))
    save(fig,'11_passer_followups',"Spain: what follows each player's passes?",'Regulation across three matches; top ten passers by initial-pass count. Labels show initial passes; each percentage has its own denominator in the CSV. Descriptive, unadjusted for receiver, zone, pressure or minutes played.')
    fig,axes=plt.subplots(3,2,figsize=(21,15));higher=tables['higher_order_edges']
    for i,match in enumerate(matches):
        for ax,team in zip(axes[i],[match.home,match.away]):
            g=higher.loc[higher.match_id.eq(match.match_id)&higher.team.eq(team)&higher.scope.eq('regulation')].nlargest(5,'count').sort_values('count')
            labels=[f'{r.a} -> {r.b} -> {r.c}' for r in g.itertuples()]
            ax.barh(labels,g['count'],color=COLORS[team]);ax.set_title(f'{team} vs {match.away if team==match.home else match.home}',loc='left',fontsize=11);ax.set_xlabel('Observed successful two-pass sequences')
            ax.tick_params(axis='y',labelsize=8)
    save(fig,'12_temporal_motifs','Common observed A -> B -> C combinations','Regulation. The same player may reappear as C (a return pass). B carries or successful take-ons are allowed between the passes; intervening teammate actions or opponent control end the motif.')
    fig,axes=plt.subplots(1,3,figsize=(17,7))
    for ax,match in zip(axes,matches):
        for team in [match.home,match.away]:
            g=follow.loc[follow.match_id.eq(match.match_id)&follow.team.eq(team)&follow.scope.eq('regulation')]
            values=[100*g[f'next_{n}_actions_shot'].mean() for n in [1,3,5]]
            ax.plot([1,3,5],values,'o-',color=COLORS[team],label=team)
        ax.set(xticks=[1,3,5],xlabel='Up to N later team actions',ylabel='Initial passes followed by a shot (%)');ax.set_title(f'Spain vs {match.away}',loc='left');ax.legend(frameon=False)
    save(fig,'13_future_horizons','Looking beyond the immediately next action','Same possession and period, stopping at opponent control. Horizons end early when the possession ends; observed-action coverage is exported. This is realized association, not a scoring-probability model.')
    fig,axes=plt.subplots(1,2,figsize=(16,8))
    g=follow.loc[follow.b_next_pass_completed]
    for ax,column,label in zip(axes,['future_possession_shot','future_possession_xg'],['Later shot frequency (%)','Mean later same-possession xG']):
        categories=['backward','lateral','forward'];groups=[g.loc[g.b_next_pass_direction.eq(c)] for c in categories]
        values=[v[column].mean()*(100 if column.endswith('shot') else 1) for v in groups]
        ax.bar(categories,values,color=['#587a99','#8da9ba','#cb6255']);ax.set_ylabel(label)
        for i,(v,s) in enumerate(zip(values,groups)):ax.text(i,v+max(values)*.03,f'n={len(s)}',ha='center',fontsize=9)
        ax.set_ylim(0,max(values)*1.2)
    save(fig,'14_backward_is_not_failure',"A backward B pass can still precede an attack",'All periods, completed B passes. Categories describe x displacement; subsequent outcomes depend on context. These unadjusted groups do not measure the causal merit of passing backward or forward.')
    g=tables['xg_player_credits'].loc[lambda x:x.team.eq('Spain')&x.scope.eq('regulation')].groupby('player')[['xg_chain_prefix','xg_buildup_prefix']].sum().nlargest(10,'xg_chain_prefix').sort_values('xg_chain_prefix')
    fig,ax=plt.subplots(figsize=(14,9));y=np.arange(len(g));ax.barh(y-.18,g.xg_chain_prefix,height=.34,label='Shot-prefix xGChain',color='#cb6255');ax.barh(y+.18,g.xg_buildup_prefix,height=.34,label='Shot-prefix xGBuildup',color='#587a99');ax.set_yticks(y,g.index);ax.set_xlabel('Overlapping shot xG credits');ax.legend(frameon=False)
    save(fig,'15_xg_chain','Who participates before the final shot?','Spain, regulation across three matches. Temporal per-shot prefix variant; each participant gets full shot xG once. Buildup excludes the shooter and key-pass provider for that shot. Credits are not additive across players.')
    return figures

def write_extended_report(tables,figures,validation,out):
    catalog=[];schema=[]
    for name,table in tables.items():
        catalog.append(f'<tr><td><a href="tables/{name}.csv">{name}.csv</a></td><td>{len(table):,}</td><td>{len(table.columns)}</td><td>{html.escape(TABLE_UNITS[name])}</td></tr>')
        schema += [{'table':name,'column':str(c),'dtype':str(table[c].dtype),'non_null':int(table[c].notna().sum()),'row_unit':TABLE_UNITS[name]} for c in table.columns]
    pd.DataFrame(schema).to_csv(out/'data_dictionary.csv',index=False)
    cards=''.join(f'<section><h2>{html.escape(item["title"])}</h2><a href="figures/{item["file"]}"><img loading="lazy" src="figures/{item["file"]}" alt="{html.escape(item["title"])}"></a><p>{html.escape(item["note"])}</p><a href="figures/{Path(item["file"]).stem}.svg">SVG</a></section>' for item in figures)
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>From A to C — Event analysis</title><style>
body{{margin:0;background:#f1f5f8;color:#20364c;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1350px;margin:auto;padding:30px 20px}}header,section{{background:white;border-radius:14px;padding:25px;margin:24px 0}}h1{{font-size:38px;line-height:1.15}}img{{width:100%;height:auto}}.brand{{width:190px}}a{{color:#27628d}}table{{border-collapse:collapse;font-size:14px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #dce4eb}}.scroll{{overflow:auto}}code{{background:#edf3f8;padding:2px 5px}}</style><main>

<header><img class="brand" src="../../assets/statsbomb-logo.png" alt="StatsBomb"><h1>From A → B to the future of possession</h1>
<p>Shots/xG, possessions, progression, pressure, losses/recoveries, zone networks, before/after windows, xT and real A → B → C sequences from the original events.</p>
<p><strong>{len(tables)} CSV tables · {len(figures)} PNG/SVG figure pairs · 2,401 initial passes.</strong> Every output retains regulation/extra-time scope.</p>
<p>After each completed initial pass, record the receiver's resolution, their next pass if they remain the handler, and later outcomes over 1/3/5 actions or 5/10/20 seconds. Paths do not bridge possessions, periods or confirmed opponent control.</p>
<p><strong>Backward passes are not automatically penalised.</strong> Direction, xT change and future shots/xG remain separate. Outcomes depend on the receiver, location, pressure and opponents; they do not isolate A's responsibility.</p>
<p>The 12 × 8 xT baseline trains on the <strong>other 48 EURO matches</strong>, excluding the three evaluated matches. Its smoothing and assumptions are explicit. It is not a calibrated VAEP/OBV implementation or causal evaluation.</p>
<p><a href="METHODOLOGY.md">Definitions and literature</a> · <a href="../../notebooks/extended_analysis.ipynb">Executed notebook</a> · <a href="data_dictionary.csv">Column dictionary</a> · <a href="validation.json">Checks</a> · <a href="models/xt_model.json">xT model</a> · <a href="../index.html">All results</a></p>
<h2>All tables</h2><div class="scroll"><table><tr><th>File</th><th>Rows</th><th>Columns</th><th>Observation unit</th></tr>{''.join(catalog)}</table></div></header>{cards}<p>Data: StatsBomb Open Data / hudl.</p>
</main></html>'''
    (out/'report.html').write_text(body,encoding='utf-8')

def run_extended(root=ROOT):
    start=time.perf_counter();dataset,tables,model=compute_extended(root)
    validation=validate_extended(dataset,tables,model)
    out=root/'results/extended'
    for directory in ['tables','figures','models']:(out/directory).mkdir(parents=True,exist_ok=True)
    for name,table in tables.items():table.to_csv(out/f'tables/{name}.csv',index=False)
    (out/'models/xt_model.json').write_text(json.dumps({'values':model.values.tolist(),'metadata':model.metadata},indent=2),encoding='utf-8')
    figures=create_extended_figures(dataset,tables,model,out/'figures')
    (out/'figure_index.json').write_text(json.dumps(figures,indent=2),encoding='utf-8')
    validation['csv_tables']=len(tables);validation['figures']=len(figures);validation['run_seconds']=round(time.perf_counter()-start,2)
    (out/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    write_extended_report(tables,figures,validation,out)
    return dataset,tables,model,figures,validation
