"""Networks for stable nominal lineups, with exact active memberships audited.

Substitution batches are atomic when no football event intervenes. Membership is
reconstructed from Starting XI, Substitution, Player Off/On and dismissal events.
Event indices assign passes; period-relative clocks measure elapsed playing-phase
time without including half-time breaks. Regulation and extra time stay separate.
"""
from __future__ import annotations

from collections import Counter
import html
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

from .analysis import ROOT, COLORS, INK, RESTARTS, build_graph, configure_style, pitch, save_figure

ADMIN = {"Starting XI", "Substitution", "Tactical Shift", "Half Start", "Half End",
         "Injury Stoppage", "Bad Behaviour", "Camera On", "Camera off", "Referee Ball-Drop"}
BASE_MINUTES = {1: 0, 2: 45, 3: 90, 4: 105}
PERIOD_NAMES = {1: "1H", 2: "2H", 3: "ET1", 4: "ET2"}
ROLE_POSITIONS = {
    1:(10,40), 2:(37,68), 3:(28,51), 4:(28,40), 5:(28,29), 6:(37,12),
    7:(48,68), 8:(48,12), 9:(48,51), 10:(48,40), 11:(48,29),
    12:(62,68), 13:(62,51), 14:(62,40), 15:(62,29), 16:(62,12),
    17:(80,66), 18:(80,51), 19:(80,40), 20:(80,29), 21:(80,14),
    22:(98,51), 23:(98,40), 24:(98,29), 25:(98,40),
}
MIN_DURATION_SECONDS = 5 * 60
MIN_NETWORK_PASSES = 20


def seconds(event):
    hh, mm, ss = event["timestamp"].split(":")
    return int(hh)*3600 + int(mm)*60 + float(ss)


def clock_label(event):
    p = event["period"]
    t = seconds(event)
    nominal = 45*60 if p <= 2 else 15*60
    base = BASE_MINUTES[p]
    if t >= nominal:
        extra = t - nominal
        return f"{PERIOD_NAMES[p]} {base + nominal//60:02d}+{int(extra)//60:02d}:{int(extra)%60:02d}"
    return f"{PERIOD_NAMES[p]} {base + int(t)//60:02d}:{int(t)%60:02d}"


def dismissal(event):
    return any(event.get(key,{}).get("card",{}).get("name") in {"Red Card","Second Yellow"}
               for key in ["foul_committed","bad_behaviour"])


def change_groups(events, team):
    """Group same-stoppage substitutions even when timestamps differ slightly."""
    events = sorted((e for e in events if e["period"] in BASE_MINUTES),key=lambda e:e["index"])
    candidates = [e for e in events if e.get("team",{}).get("name")==team and
                  (e["type"]["name"] in {"Substitution","Player Off","Player On"} or dismissal(e))]
    groups = []
    for event in candidates:
        if groups:
            previous = groups[-1][-1]
            between = [e for e in events if previous["index"] < e["index"] < event["index"]]
            if (event["type"]["name"] == previous["type"]["name"] == "Substitution"
                and event["period"] == previous["period"]
                and all(e["type"]["name"] in ADMIN for e in between)):
                groups[-1].append(event)
                continue
        groups.append([event])
    return groups


def reconstruct_stints(events, team, *, nominal=False):
    """Return contiguous memberships; nominal mode retains temporarily off players.

    Substitutions and dismissals always change the nominal lineup. The default
    exact mode remains the source of truth for checking actual pass endpoints.
    """
    events = sorted((e for e in events if e["period"] in BASE_MINUTES),key=lambda e:e["index"])
    start_xi = next(e for e in events if e["type"]["name"]=="Starting XI" and e["team"]["name"]==team)
    active = {p["player"]["id"] for p in start_xi["tactics"]["lineup"]}
    assert len(active)==11, (team,"Starting XI must contain 11")
    temporary = set()
    roles = {p["player"]["id"]:p["position"]["id"] for p in start_xi["tactics"]["lineup"]}
    start = dict(events[0],index=1,timestamp="00:00:00.000",period=1)
    end = dict(events[-1],index=events[-1]["index"]+1)
    result = []
    reason = "Starting XI"
    change_ids = []
    for group in change_groups(events,team):
        boundary = group[0]
        # Tactical shifts update reference slots but do not change the player set.
        snapshot = [e for e in events if start["index"] <= e["index"] < boundary["index"]
                    and e["type"]["name"]=="Tactical Shift" and e["team"]["name"]==team]
        for event in snapshot:
            roles.update({p["player"]["id"]:p["position"]["id"] for p in event["tactics"]["lineup"]})
        next_active, next_temporary, next_roles = set(active), set(temporary), dict(roles)
        reasons = []
        for event in group:
            kind, pid = event["type"]["name"], event.get("player",{}).get("id")
            if kind=="Substitution":
                new = event["substitution"]["replacement"]["id"]
                assert pid in next_active or pid in next_temporary, (team,"outgoing player absent",pid)
                assert new not in next_active, (team,"replacement already active",new)
                next_active.discard(pid); next_temporary.discard(pid)
                next_active.add(new)
                next_roles[new] = event.get("position",{}).get("id",next_roles.get(pid,14))
                reasons.append(f"{pid} -> {new}")
            elif kind=="Player Off":
                assert pid in next_active, (team,"Player Off not active",pid)
                next_active.remove(pid); next_temporary.add(pid)
                reasons.append(f"Temporary off: {pid}")
            elif kind=="Player On":
                assert pid in next_temporary, (team,"Player On without prior off",pid)
                next_temporary.remove(pid); next_active.add(pid)
                reasons.append(f"Returned: {pid}")
            elif dismissal(event) and pid in next_active | next_temporary:
                next_active.discard(pid)
                next_temporary.discard(pid)
                reasons.append(f"Dismissal: {pid}")
        # Bench cards do not change an on-field lineup.
        membership = active | temporary if nominal else active
        next_membership = next_active | next_temporary if nominal else next_active
        if next_membership==membership:
            active,temporary,roles = next_active,next_temporary,next_roles
            continue
        result.append({"start":start,"end":boundary,"roster":set(membership),"roles":dict(roles),
                       "reason":reason,"change_ids":change_ids})
        start,active,temporary,roles = boundary,next_active,next_temporary,next_roles
        reason = "; ".join(reasons)
        change_ids = [e["id"] for e in group]
    result.append({"start":start,"end":end,"roster":set(active | temporary if nominal else active),"roles":dict(roles),
                   "reason":reason,"change_ids":change_ids})
    return result


def period_bounds(events):
    bounds, offset = {}, 0.
    for p in sorted({e["period"] for e in events if e["period"] in BASE_MINUTES}):
        period_events = [e for e in events if e["period"]==p]
        end_event = max(period_events,key=lambda e:(seconds(e),e["index"]))
        bounds[p] = {"offset":offset, "duration":seconds(end_event),
                     "start":dict(period_events[0],timestamp="00:00:00.000"),
                     "end":dict(end_event,index=max(e["index"] for e in period_events)+1)}
        offset += seconds(end_event)
    return bounds


def interval_graph(dataset, segment):
    """All active players are nodes, including those with zero completed passes."""
    mid, team = segment["match_id"],segment["team"]
    sample = dataset["passes"].loc[lambda x: x.match_id.eq(mid) & x.team.eq(team)
             & x["index"].ge(segment["start_index"]) & x["index"].lt(segment["end_index"])
             & x.period.isin(segment["periods"])].copy()
    # Extra time uses exactly the same pass filters, without the regulation flag.
    sample["regulation"] = True
    sample["network_pass"] = sample.open_play & sample.completed & sample.has_recipient & ~sample.self_pass
    graph = build_graph(sample,dataset["players"],mid,team)
    assert set(graph).issubset(segment["roster"]), (segment["segment_id"],set(graph)-segment["roster"])
    for pid in segment["roster"]:
        if pid not in graph:
            graph.add_node(pid,**dataset["players"][(mid,pid)])
        attrs = graph.nodes[pid]
        if "x" in attrs:
            attrs["position_source"] = "completed pass/reception"
            continue
        # Keep zero-involvement players visible without inventing pass positions.
        locations = [e["location"][:2] for e in dataset["events"][mid]
                     if e["period"] in segment["periods"] and segment["start_index"] <= e["index"] < segment["end_index"]
                     and e.get("team",{}).get("name")==team and e.get("player",{}).get("id")==pid
                     and isinstance(e.get("location"),list)]
        if locations:
            x,y = np.mean(locations,axis=0)
            source = "other interval events"
        else:
            x,y = ROLE_POSITIONS[segment["roles"].get(pid,14)]
            source = "tactical reference (not observed position)"
        attrs.update(x=float(x),y=float(y),position_source=source)
    assert set(graph)==segment["roster"]
    graph.graph.update(scope=segment["scope"],segment_id=segment["segment_id"],
                       start=segment["start_label"],end=segment["end_label"],
                       start_index=segment["start_index"],end_index_exclusive=segment["end_index"],
                       duration_seconds=segment["duration_seconds"])
    return graph,sample.loc[sample.network_pass]


def _compute_lineup_segments(dataset, *, nominal):
    segments, graphs, assigned = [], {}, []
    for match in dataset["matches"].itertuples():
        mid = match.match_id
        events = dataset["events"][mid]
        bounds = period_bounds(events)
        for team in [match.home,match.away]:
            for scope,periods in [("regulation",[1,2]),("extra_time",[3,4])]:
                available = [p for p in periods if p in bounds]
                if not available:
                    continue
                phase_start,phase_end = bounds[available[0]]["start"],bounds[available[-1]]["end"]
                number = 0
                for raw in reconstruct_stints(events,team,nominal=nominal):
                    start = max([raw["start"],phase_start],key=lambda e:e["index"])
                    end = min([raw["end"],phase_end],key=lambda e:e["index"])
                    if start["index"] >= end["index"]:
                        continue
                    elapsed = lambda e: bounds[e["period"]]["offset"]+seconds(e)
                    duration = elapsed(end)-elapsed(start)
                    # Initial period-start administration can have zero elapsed
                    # time. Retain a segment if it nevertheless has a pass event.
                    has_pass = any(e["type"]["name"]=="Pass" and e["team"]["name"]==team
                                   and start["index"]<=e["index"]<end["index"] for e in events)
                    if duration<=0 and not has_pass:
                        continue
                    number += 1
                    sid = f"{mid}_{team.lower()}_{scope}_{number:02d}"
                    segment = {"segment_id":sid,"match_id":mid,"team":team,
                               "opponent":match.away if team==match.home else match.home,
                               "scope":scope,"number":number,"start_index":start["index"],
                               "end_index":end["index"],"start_label":clock_label(start),
                               "end_label":clock_label(end),"duration_seconds":round(duration,3),
                               "periods":available,"roster":raw["roster"],"roles":raw["roles"],
                               "reason":raw["reason"],"change_ids":raw["change_ids"]}
                    graph, passes = interval_graph(dataset,segment)
                    segment.update(players=len(graph),network_passes=len(passes),links=graph.number_of_edges(),
                                   density=nx.density(graph),zero_involvement=sum(graph.degree(pid,weight="weight")==0 for pid in graph))
                    segments.append(segment)
                    graphs[sid] = graph
                    for event in passes.itertuples():
                        assigned.append({"match_id":mid,"team":team,"event_id":event.event_id,
                                         "segment_id":sid,"scope":scope,"passer_id":int(event.passer_id),
                                         "recipient_id":int(event.recipient_id)})
    return {"segments":segments,"graphs":graphs,"assignments":pd.DataFrame(assigned),
            "lineup_definition":"nominal" if nominal else "exact_active"}


def presentation_status(segment, min_duration=MIN_DURATION_SECONDS, min_passes=MIN_NETWORK_PASSES):
    """Predeclared display thresholds; never merge rosters or remove source passes."""
    reasons=[]
    if segment["duration_seconds"] < min_duration:
        reasons.append("duration_below_threshold")
    if segment["network_passes"] < min_passes:
        reasons.append("passes_below_threshold")
    return not reasons, ";".join(reasons)


def compute_fixed_lineups(dataset, min_duration=MIN_DURATION_SECONDS, min_passes=MIN_NETWORK_PASSES):
    result=_compute_lineup_segments(dataset,nominal=True)
    audit=_compute_lineup_segments(dataset,nominal=False)
    result["active_audit"]=audit
    result["presentation_policy"]={"min_duration_seconds":min_duration,"min_network_passes":min_passes}
    absences=[]
    for segment in result["segments"]:
        mid,team,scope=segment["match_id"],segment["team"],segment["scope"]
        bounds=period_bounds(dataset["events"][mid])
        elapsed=lambda event:bounds[event["period"]]["offset"]+seconds(event)
        lookup={e["index"]:e for e in dataset["events"][mid]}
        # Segment boundary events themselves, rather than a shared index lookup,
        # avoid the end-of-period / next-period-start index collision.
        def boundary(index, ending=False):
            if ending and index==bounds[segment["periods"][-1]]["end"]["index"]:
                return bounds[segment["periods"][-1]]["end"]
            return lookup[index]
        overlap=[s for s in audit["segments"] if (s["match_id"],s["team"],s["scope"])==(mid,team,scope)
                 and s["start_index"]<segment["end_index"] and s["end_index"]>segment["start_index"]]
        segment["active_audit_segment_ids"]=[s["segment_id"] for s in overlap]
        segment["temporary_absence_seconds"]=0.
        for exact in overlap:
            missing=segment["roster"]-exact["roster"]
            if not missing:
                continue
            start=boundary(max(segment["start_index"],exact["start_index"]))
            end=boundary(min(segment["end_index"],exact["end_index"]),ending=True)
            duration=round(elapsed(end)-elapsed(start),3)
            segment["temporary_absence_seconds"]+=duration
            absences.append({"segment_id":segment["segment_id"],"match_id":mid,"team":team,"scope":scope,
                             "start_index":start["index"],"end_index":end["index"],
                             "start_label":clock_label(start),"end_label":clock_label(end),
                             "duration_seconds":duration,"absent_player_ids":";".join(map(str,sorted(missing))),
                             "absent_player_names":"; ".join(dataset["players"][(mid,p)]["name"] for p in sorted(missing))})
        segment["temporary_absence_seconds"]=round(segment["temporary_absence_seconds"],3)
        eligible,reason=presentation_status(segment,min_duration,min_passes)
        segment.update(display_eligible=eligible,display_exclusion_reason=reason)
        result["graphs"][segment["segment_id"]].graph.update(
            lineup_definition="nominal",temporary_absence_seconds=segment["temporary_absence_seconds"],
            display_eligible=eligible)
    result["temporary_absences"]=pd.DataFrame(absences)
    return result


def validate_fixed_lineups(dataset, result):
    expected = dataset["passes"].loc[lambda x:x.period.isin([1,2,3,4]) & x.open_play & x.completed & x.has_recipient & ~x.self_pass]
    assigned = result["assignments"]
    assert not assigned.duplicated(["match_id","event_id"]).any()
    assert set(zip(expected.match_id,expected.event_id))==set(zip(assigned.match_id,assigned.event_id))
    summed_edges = Counter()
    for segment in result["segments"]:
        graph = result["graphs"][segment["segment_id"]]
        assert set(graph)==segment["roster"]
        assert 1<=len(graph)<=11
        assert all(np.isfinite([a["x"],a["y"]]).all() for _,a in graph.nodes(data=True))
        assert sum(nx.get_edge_attributes(graph,"weight").values())==segment["network_passes"]
        assert graph.number_of_edges()==segment["links"]
        for u,v,a in graph.edges(data=True):
            assert u in segment["roster"] and v in segment["roster"] and a["weight"]>=1
            summed_edges[(segment["match_id"],segment["team"],u,v)] += a["weight"]
    expected_edges = expected.groupby(["match_id","team","passer_id","recipient_id"]).size().to_dict()
    assert dict(summed_edges)==expected_edges
    checks={"status":"passed","segments":len(result["segments"]),
            "regulation_segments":sum(s["scope"]=="regulation" for s in result["segments"]),
            "extra_time_segments":sum(s["scope"]=="extra_time" for s in result["segments"]),
            "eleven_player_segments":sum(s["players"]==11 for s in result["segments"]),
            "short_handed_segments":sum(s["players"]<11 for s in result["segments"]),
            "unique_passes_assigned_once":len(assigned),
            "lineup_definition":result["lineup_definition"],
            "checks":["reconstructed roster","maximum eleven nodes","all roster players retained",
                      "exact pass partition","no inactive pass endpoints","edge-weight conservation",
                      "atomic substitution batches","separate regulation and extra time"]}
    if "active_audit" in result:
        exact=validate_fixed_lineups(dataset,result["active_audit"])
        eligible=[s for s in result["segments"] if s["display_eligible"]]
        assert all(presentation_status(s,**{"min_duration":result["presentation_policy"]["min_duration_seconds"],
                                          "min_passes":result["presentation_policy"]["min_network_passes"]})[0] for s in eligible)
        checks.update(exact_active_segments=exact["segments"],displayed_segments=len(eligible),
                      audit_only_segments=len(result["segments"])-len(eligible),
                      displayed_passes=sum(s["network_passes"] for s in eligible),
                      temporary_absence_records=len(result["temporary_absences"]),
                      presentation_policy=result["presentation_policy"],exact_active_validation=exact)
        checks["checks"].extend(["exact active memberships audited separately","temporary absences disclosed",
                                 "short stints retained in tables without merging different rosters"])
    return checks


def draw_fixed_network(ax,graph,segment,names=True):
    pitch(ax)
    positions = {pid:(a["x"],a["y"]) for pid,a in graph.nodes(data=True)}
    nodes = list(graph)
    sizes = [190+5*graph.degree(pid,weight="weight") for pid in nodes]
    # Only display markers are separated. Original event-derived coordinates in
    # the CSV/GraphML remain unchanged and are marked with a small grey tick.
    points=np.array([positions[pid] for pid in nodes],dtype=float)
    for _ in range(80):
        moved=False
        for i in range(len(nodes)):
            for j in range(i+1,len(nodes)):
                delta=points[j]-points[i]
                distance=np.linalg.norm(delta)
                minimum=4.5+(np.sqrt(sizes[i])+np.sqrt(sizes[j]))/30
                if distance<minimum:
                    direction=delta/distance if distance>1e-8 else np.array([1.,0.])
                    step=direction*(minimum-distance+.01)/2
                    points[i]-=step; points[j]+=step
                    moved=True
        if not moved:
            break
    for pid,point in zip(nodes,points):
        original=positions[pid]
        if np.linalg.norm(point-original)>.15:
            ax.plot([original[0],point[0]],[original[1],point[1]],color="#8a99a5",lw=.5,zorder=1)
            ax.scatter(*original,marker="+",s=18,color="#8a99a5",lw=.6,zorder=2)
        positions[pid]=tuple(point)
    if graph.number_of_edges():
        nx.draw_networkx_edges(graph,positions,ax=ax,
            width=[0.30*a["weight"] for _,_,a in graph.edges(data=True)],
            edge_color=COLORS[segment["team"]],alpha=.60,arrows=True,arrowstyle="-|>",
            arrowsize=10,connectionstyle="arc3,rad=0.13",nodelist=nodes,node_size=sizes,
            min_source_margin=8,min_target_margin=9)
    for pid,size in zip(nodes,sizes):
        a = graph.nodes[pid]
        zero = graph.degree(pid,weight="weight")==0
        x,y=positions[pid]
        ax.scatter(x,y,s=size,facecolor="#fafbfd" if zero else COLORS[segment["team"]],
                   edgecolor="#7e8b98" if zero else "white",lw=1.4,zorder=3)
        ax.text(x,y,str(a["jersey"]),ha="center",va="center",color=INK if zero else "white",
                fontsize=8,weight="bold",zorder=4)
        if names:
            label=ax.annotate(a["name"],(x,y),xytext=(0,12),textcoords="offset points",
                              ha="center",fontsize=7,zorder=5,
                              bbox={"facecolor":"#fafbfd","alpha":.88,"edgecolor":"none","pad":1})
            label.set_gid("player-label")
    ax.set_title(f"{segment['start_label']} to {segment['end_label']}\n"
                 f"{segment['players']} players | {segment['network_passes']} passes | {segment['duration_seconds']/60:.1f} min",
                 loc="left",fontsize=10,weight="bold",pad=9)
    if segment["players"]<11:
        ax.text(.02,.04,"TEMPORARILY SHORT-HANDED" if "Temporary" in segment["reason"] else "AFTER DISMISSAL",
                transform=ax.transAxes,color="#a45039",fontsize=8,weight="bold")
    elif segment.get("temporary_absence_seconds",0)>0:
        ax.text(.02,.04,f"NOMINAL XI | temporary absence: {segment['temporary_absence_seconds']/60:.2f} min",
                transform=ax.transAxes,color="#a45039",fontsize=8,weight="bold")


def add_weight_legend(fig):
    handles=[Line2D([0],[0],color=INK,lw=.30*w,label=f"{w} pass" if w==1 else f"{w} passes") for w in [1,5,10]]
    fig.legend(handles=handles,loc="upper right",bbox_to_anchor=(.97,.977),ncol=3,
               frameon=False,fontsize=8,handlelength=2.4,columnspacing=1.4)


def export_fixed_lineups(dataset,result,root=ROOT):
    configure_style()
    out=root/"results/fixed_lineups"
    for sub in ["figures","graphs","tables"]:
        (out/sub).mkdir(parents=True,exist_ok=True)
    policy=result["presentation_policy"]
    selected_main=[s for s in result["segments"] if s["display_eligible"]]
    old_index=json.loads((out/"figure_index.json").read_text(encoding="utf-8")) if (out/"figure_index.json").exists() else []
    note=("Every directed link with 1+ completed open-play passes. Width = 0.30 pt per pass; identical across all panels.\n"
          "Hollow nodes: zero pass involvement; position from other interval events or a tactical reference if none exists.\n"
          "Close markers are separated for legibility; grey ticks show original locations, retained in CSV/GraphML.\n"
          f"Nominal lineups; temporary absences disclosed. Display: at least {policy['min_duration_seconds']/60:g} min and {policy['min_network_passes']} passes. All stints kept in CSV.")
    figures=[]
    for segment in result["segments"]:
        graph=result["graphs"][segment["segment_id"]]
        stem=segment["segment_id"]
        nx.write_graphml(graph,out/"graphs"/f"{stem}.graphml")
        if not segment["display_eligible"]:
            continue
        fig,ax=plt.subplots(figsize=(12,10))
        draw_fixed_network(ax,graph,segment)
        add_weight_legend(fig)
        stem=f"{segment['segment_id']}"
        title=f"{segment['team']} vs {segment['opponent']} | Lineup {segment['number']:02d}"
        path=save_figure(fig,out/"figures"/stem,title,note,
                         footer="Data: StatsBomb Open Data  |  EURO 2024  |  " + segment["scope"].replace("_"," ").title())
        item={"file":path.name,"title":title,"segment_id":segment["segment_id"],"kind":"individual"}
        figures.append(item)
    # Six primary team pages, plus two separate extra-time pages.
    group_keys=list(dict.fromkeys((s["match_id"],s["team"],s["scope"]) for s in selected_main))
    for mid,team,scope in group_keys:
        selected=[s for s in selected_main if (s["match_id"],s["team"],s["scope"])==(mid,team,scope)]
        rows=math.ceil(len(selected)/2)
        fig,axes=plt.subplots(rows,2,figsize=(17,max(7,rows*5.2+1.6)),squeeze=False)
        for ax,s in zip(axes.flat,selected):
            draw_fixed_network(ax,result["graphs"][s["segment_id"]],s,names=True)
        for ax in list(axes.flat)[len(selected):]:
            ax.axis("off")
        add_weight_legend(fig)
        stem=f"overview_{mid}_{team.lower()}_{scope}"
        title=f"{team} vs {selected[0]['opponent']} | {'Extra time' if scope=='extra_time' else 'Regulation'}"
        path=save_figure(fig,out/"figures"/stem,title,note,
                         footer="Data: StatsBomb Open Data  |  EURO 2024  |  " + scope.replace("_"," ").title())
        figures.append({"file":path.name,"title":title,"kind":"overview","match_id":mid,"team":team,"scope":scope})
    # Machine-readable segment, player, edge, assignment and substitution-batch tables.
    segment_rows,player_rows,edge_rows=[] ,[],[]
    for s in result["segments"]:
        g=result["graphs"][s["segment_id"]]
        row={k:v for k,v in s.items() if k not in {"roster","roles","periods","change_ids","active_audit_segment_ids"}}
        row["active_audit_segment_ids"]=";".join(s["active_audit_segment_ids"])
        row["roster_ids"]=";".join(map(str,sorted(s["roster"])))
        row["roster_names"]="; ".join(g.nodes[pid]["name"] for pid in sorted(g,key=lambda p:g.nodes[p]["jersey"]))
        row["change_event_ids"]=";".join(s["change_ids"])
        segment_rows.append(row)
        for pid,a in g.nodes(data=True):
            player_rows.append({"segment_id":s["segment_id"],"player_id":pid,**a,
                                "passes_sent":g.out_degree(pid,weight="weight"),
                                "passes_received":g.in_degree(pid,weight="weight")})
        for u,v,a in g.edges(data=True):
            edge_rows.append({"segment_id":s["segment_id"],"passer_id":u,"recipient_id":v,**a})
    for name,frame in [("segments",pd.DataFrame(segment_rows)),("players",pd.DataFrame(player_rows)),
                       ("edges",pd.DataFrame(edge_rows)),("pass_assignments",result["assignments"]),
                       ("temporary_absences",result["temporary_absences"]),
                       ("active_pass_assignments",result["active_audit"]["assignments"])]:
        frame.to_csv(out/f"tables/{name}.csv",index=False)
    exact_rows=[]
    for s in result["active_audit"]["segments"]:
        row={k:v for k,v in s.items() if k not in {"roster","roles","periods","change_ids"}}
        row.update(roster_ids=";".join(map(str,sorted(s["roster"]))),change_event_ids=";".join(s["change_ids"]))
        exact_rows.append(row)
    pd.DataFrame(exact_rows).to_csv(out/"tables/active_segments.csv",index=False)
    # Remove only generated files explicitly listed by the previous manifest.
    # This avoids stale short-window figures surviving after a policy change.
    keep={f["file"] for f in figures}
    graph_ids=set(result["graphs"])
    for item in old_index:
        old_file=Path(item["file"])
        assert old_file.name==item["file"]
        if old_file.name not in keep:
            for suffix in [".png",".svg"]:
                (out/"figures"/old_file.with_suffix(suffix)).unlink(missing_ok=True)
        sid=item.get("segment_id")
        if sid and sid not in graph_ids:
            assert Path(sid).name==sid
            (out/"graphs"/f"{sid}.graphml").unlink(missing_ok=True)
    (out/"figure_index.json").write_text(json.dumps(figures,ensure_ascii=False,indent=2),encoding="utf-8")
    validation=validate_fixed_lineups(dataset,result)
    (out/"validation.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
    write_fixed_report(result,figures,out)
    return figures,validation


def write_fixed_report(result,figures,out):
    e=html.escape
    overview=[f for f in figures if f["kind"]=="overview"]
    nav=" · ".join(f'<a href="#{f["file"]}">{e(f["title"])}</a>' for f in overview)
    body=[]
    for f in overview:
        selected=[s for s in result["segments"] if s["display_eligible"] and (s["match_id"],s["team"],s["scope"])==(f["match_id"],f["team"],f["scope"])]
        rows=[]
        for s in selected:
            g=result["graphs"][s["segment_id"]]
            roster=", ".join(f'{g.nodes[p]["jersey"]} {g.nodes[p]["name"]}' for p in sorted(g,key=lambda p:g.nodes[p]["jersey"]))
            rows.append(f'<tr><td>{s["number"]:02d}</td><td>{s["start_label"]} → {s["end_label"]}</td>'
                        f'<td>{s["duration_seconds"]/60:.2f}</td><td>{s["players"]}</td><td>{s["network_passes"]}</td>'
                        f'<td><a href="figures/{s["segment_id"]}.png">PNG</a> · <a href="figures/{s["segment_id"]}.svg">SVG</a>'
                        f'<details><summary>Full roster</summary>{e(roster)}</details></td></tr>')
        body.append(f'<section id="{f["file"]}"><h2>{e(f["title"])}</h2>'
                    f'<a href="figures/{f["file"]}"><img loading="lazy" src="figures/{f["file"]}" alt="{e(f["title"])}"></a>'
                    f'<p><a href="figures/{Path(f["file"]).stem}.svg">Overview SVG</a></p><div class="scroll"><table>'
                    '<tr><th>Interval</th><th>Match time</th><th>Minutes</th><th>Players</th><th>Passes</th><th>Individual figure</th></tr>'
                    +"".join(rows)+'</table></div></section>')
    omitted=[s for s in result["segments"] if not s["display_eligible"]]
    audit_rows="".join(f'<tr><td>{e(s["team"])}–{e(s["opponent"])}</td><td>{s["start_label"]} → {s["end_label"]}</td>'
                       f'<td>{s["duration_seconds"]/60:.2f}</td><td>{s["network_passes"]}</td>'
                       f'<td>{e(s["display_exclusion_reason"])}</td></tr>' for s in omitted)
    policy=result["presentation_policy"]
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stable lineups — Football Passing Networks</title><style>
body{{margin:0;background:#f1f5f8;color:#20364c;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1350px;margin:auto;padding:32px 22px}}
h1{{font-size:38px;line-height:1.15}}h2{{font-size:24px}}section,header{{background:white;border-radius:14px;padding:25px;margin:24px 0}}
img{{width:100%;height:auto}}a{{color:#27628d}}.brand{{width:190px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #dce4eb}}details{{max-width:420px}}
</style><main>

<header><img class="brand" src="../../assets/statsbomb-logo.png" alt="StatsBomb"><h1>Passing networks with stable nominal lineups</h1>
<p><strong>{len(result['segments'])} nominal-lineup intervals</strong>, changing at substitutions and dismissals. <strong>{len(result['segments'])-len(omitted)} have main figures</strong>: at least {policy['min_duration_seconds']/60:g} minutes and {policy['min_network_passes']} completed open-play passes. The other {len(omitted)} remain in CSV and GraphML. Different rosters are never merged to meet a threshold.</p>
<p>Every link from <strong>one completed pass</strong> is shown. Line width is <strong>0.30 pt × pass count</strong> on the same scale throughout; arrows show direction. Raw counts from different durations are not directly comparable passing rates.</p>
<p>Substitutions in one stoppage are applied together if no football event intervenes. Opponent changes do not split a team's lineup. Temporary departures do not split the nominal roster but their durations are disclosed: all nodes need not be present at every instant. The <strong>{len(result['active_audit']['segments'])} exact active intervals</strong> are retained separately. Extra time remains separate.</p>
<p>Every roster member is plotted. Filled nodes have completed-pass involvement. Hollow nodes have none: their location comes from other events in the interval or a schematic tactical-role reference. These are not tracking measurements. Small visual offsets separate overlapping labels; grey ticks mark original positions, which remain unchanged in CSV/GraphML.</p>
<p>Duration includes stoppage time and excludes half-time breaks. Half-open event-index boundaries assign all 2,401 passes exactly once in each representation. A substitution batch uses its first change event as the common boundary. Presentation filtering does not remove passes from analysis.</p>
<p><a href="../index.html">All results</a> · <a href="../../notebooks/fixed_lineups.ipynb">Executed notebook</a> · <a href="../../docs/LINEUP_POLICY.md">Policy</a> · <a href="tables/segments.csv">Nominal intervals</a> · <a href="tables/active_segments.csv">Exact active intervals</a> · <a href="tables/temporary_absences.csv">Temporary absences</a> · <a href="validation.json">Checks</a></p><nav>{nav}</nav></header>
{''.join(body)}<section><h2>Audit-only short or sparse intervals</h2><p>The five-minute/20-pass display thresholds are practical presentation choices, not statistical tests.</p>
<div class="scroll"><table><tr><th>Team–opponent</th><th>Time</th><th>Minutes</th><th>Passes</th><th>Reason</th></tr>{audit_rows}</table></div></section>
<p>Data: StatsBomb Open Data / hudl. The same open-play filters as the aggregate analysis are used.</p>
</main></html>'''
    (out/"report.html").write_text(text,encoding="utf-8")


def run_fixed_lineups(root=ROOT):
    from .analysis import load_dataset
    dataset=load_dataset(root)
    result=compute_fixed_lineups(dataset)
    validate_fixed_lineups(dataset,result)
    figures,validation=export_fixed_lineups(dataset,result,root)
    return dataset,result,figures,validation
