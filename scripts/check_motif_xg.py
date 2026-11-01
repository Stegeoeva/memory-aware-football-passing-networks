"""Verify new source xG metrics, common cohorts, overlap and ratio uncertainty."""
from pathlib import Path
import json
import re
import sys
import unittest
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.evaluation import load_study_inputs,hierarchical_weights,assert_split
from src.motifs import clean_records
from src.sequences import sha256
from src.study import write_json


def main():
    out=ROOT/"results/motif_xg";run=json.loads((out/"run_manifest.json").read_text())
    config=json.loads((ROOT/"configs/motif_xg.json").read_text())
    for p,h in {**run["analysis"]["code"],**run["code"]}.items():assert sha256(ROOT/p)==h,("code",p)
    for p,h in run["analysis"]["inputs"].items():
        file=ROOT/p if (ROOT/p).exists() else ROOT/"results/euro_sequences"/p
        assert sha256(file)==h,("input",p)
    for p,spec in run["outputs"].items():
        file=ROOT/p
        assert file.stat().st_size==spec["bytes"] and sha256(file)==spec["sha256"],("output",p)
    tables,_=load_study_inputs(ROOT)
    events={r["event_id"]:r for r in clean_records(tables["events"])}
    source={r["sequence_id"]:r for r in clean_records(tables["receiver_sequences"])}
    periods={};positions={}
    for key,g in tables["events"].sort_values(["match_id","period","index"]).groupby(["match_id","period"]):
        ids=g.event_id.to_list();periods[key]=[events[k] for k in ids];positions.update({k:i for i,k in enumerate(ids)})
    windows=pd.read_csv(out/"tables/sequence_windows.csv.gz",low_memory=False)
    assert len(windows)==2*41927 and not windows.duplicated(["sequence_id","horizon"]).any()
    checked_shots=0
    for r in windows.itertuples():
        s=source[r.sequence_id];rows=periods[(s["match_id"],s["period"])]
        start=positions[s["a_event_id"]]
        stop=positions[r.window_end_event_id] if pd.notna(r.window_end_event_id) else len(rows)
        endpoint=rows[stop] if stop<len(rows) else None
        if r.window_end_reason=="provider_possession_change":
            assert (endpoint["possession_id"],endpoint["possession_team_id"]) != (rows[start]["possession_id"],rows[start]["possession_team_id"])
        elif r.window_end_reason=="confirmed_opponent_control":
            assert endpoint["controlled_touch"] and endpoint["team_id"]!=s["team_id"]
        elif r.window_end_reason=="goal":assert endpoint["stoppage_reason"]=="goal"
        elif r.window_end_reason=="observed_period_end":assert endpoint["event_type"]=="Half End"
        begin=positions[s["receipt_event_id"]]+1 if s["receipt_event_id"] in positions else start+1
        deadline=np.inf if r.horizon=="possession" else s["receipt_clock_seconds"]+10
        inclusive=r.window_end_reason=="goal"
        for e in rows[start+1:stop]:
            assert (e["possession_id"],e["possession_team_id"]) == (rows[start]["possession_id"],rows[start]["possession_team_id"])
            assert not(e["controlled_touch"] and e["team_id"]!=s["team_id"])
            assert e["stoppage_reason"]!="goal" and e["event_type"]!="Half End"
        selected=[e for e in rows[begin:stop+int(inclusive)] if e["team_id"]==s["team_id"] and e["event_type"]=="Shot"
                  and s["receipt_clock_seconds"]-.001<=e["clock_seconds"]<=deadline+.001]
        ids=[e["event_id"] for e in selected]
        assert (";".join(ids) if ids else None)==(r.shot_event_ids if pd.notna(r.shot_event_ids) else None)
        assert r.observed_shot_count==len(selected)
        if r.xg_observed:
            assert r.observation_complete and r.receipt_quality and r.missing_shot_xg==0
            np.testing.assert_allclose(r.xg_sum,sum(e["xg"] for e in selected),atol=1e-12)
            assert r.shot_count==len(selected) and r.shot_any==float(bool(selected))
        else:
            assert pd.isna(r.xg_sum) and pd.isna(r.shot_count) and pd.isna(r.shot_any)
        checked_shots+=len(selected)
    source_links=pd.read_csv(out/"tables/sequence_shot_links.csv.gz")
    assert not source_links.duplicated(["sequence_id","horizon","shot_event_id"]).any()
    assert len(source_links)==checked_shots
    for r in source_links.itertuples():
        e=events[r.shot_event_id]
        assert e["event_type"]=="Shot" and e["team_id"]==r.team_id and e["match_id"]==r.match_id
        np.testing.assert_allclose(e["xg"],r.shot_xg,atol=1e-12)
    print("All 83,854 source windows: boundaries, receipt landmarks, selected Shots, zeros and missingness verified.",flush=True)
    frame=pd.read_csv(out/"tables/oof_motif_xg.csv.gz",low_memory=False)
    assert len(frame)==121066 and not frame.duplicated(["motif_id","horizon"]).any()
    links=pd.read_csv(out/"tables/motif_shot_links.csv.gz")
    assert not links.duplicated(["motif_id","horizon","shot_event_id"]).any()
    labels=windows.set_index(["sequence_id","horizon"])
    expected=labels.reindex(pd.MultiIndex.from_arrays([frame.end_sequence_id,frame.horizon]))
    for col in ["shot_any","shot_count","xg_sum","observation_complete","xg_observed"]:
        np.testing.assert_allclose(frame[col].to_numpy(dtype=float),expected[col].to_numpy(dtype=float),equal_nan=True,atol=1e-12)
    splits=json.loads((out/"split_manifest.json").read_text())["splits"]
    for record in splits:
        assert_split(record)
        assert set(frame.loc[frame.fold.eq(record["fold"])].match_id)==set(record["test_match_ids"])
    for spec in config["context_specs"]:
        for outcome in config["q_outcomes"]:
            p=frame[f"pred_{spec}_{outcome}"]
            assert p.notna().all() and p.ge(0).all()
            if outcome=="shot_any":assert p.le(1).all()
            np.testing.assert_allclose(frame[f"residual_{spec}_{outcome}"],frame[outcome]-p,equal_nan=True,atol=1e-12)
    audit=pd.read_csv(out/"tables/fit_audit.csv")
    assert len(audit)==920 and audit.converged.all() and audit.stage.eq("outer").sum()==80
    for r in audit.itertuples():
        record=splits[r.fold] if r.stage=="outer" else splits[r.fold]["inner"][r.inner_fold]
        mask=frame.order.eq(r.order)&frame.horizon.eq(r.horizon)&frame.match_id.isin(record["train_match_ids"])
        assert r.training_observed==int(frame.loc[mask,r.outcome].notna().sum())
    print("Common motif cohorts, 80 outer q models, all 920 converged fits and residuals verified.",flush=True)
    metrics=pd.read_csv(out/"tables/pattern_metrics.csv")
    contrasts=pd.read_csv(out/"tables/pattern_contrasts.csv")
    for key,g in metrics.groupby(["order","horizon","policy","scope_filter"]):
        order,horizon,policy,scope=key
        mask=frame.order.eq(order)&frame.horizon.eq(horizon)&frame[policy]
        if scope=="regulation":mask&=frame.period.le(2)
        cohort=frame.loc[mask]
        for r in g.itertuples():
            raw=cohort.loc[cohort.pattern.eq(r.pattern)]
            known=raw.loc[raw.xg_observed]
            linked=links.loc[links.motif_id.isin(known.motif_id)&links.horizon.eq(horizon)]
            unique=linked.drop_duplicates("shot_event_id")
            assert len(raw)==r.motifs and len(known)==r.known_motifs and int(known.shot_count.sum())==r.linked_shots
            assert len(unique)==r.unique_shots
            if r.context=="raw":
                if r.metric=="shot_rate":value=100*known.shot_any.mean()
                elif r.metric=="xg_per_motif":value=known.xg_sum.mean()
                elif r.metric=="shots_per_motif":value=known.shot_count.mean()
                elif r.metric=="xg_per_shot":value=known.xg_sum.sum()/known.shot_count.sum() if known.shot_count.sum()>0 else np.nan
                else:value=unique.shot_xg.mean()
            else:
                outcome="shot_any" if r.metric=="shot_rate" else "xg_sum"
                value=known[f"residual_{r.context}_{outcome}"].mean()*(100 if outcome=="shot_any" else 1)
            np.testing.assert_allclose(value,r.estimate,equal_nan=True,atol=1e-12)
            if not r.support_eligible:assert pd.isna(r.ci95_low) and pd.isna(r.ci95_high)
            else:assert r.ci95_low<=r.ci95_high and r.bootstrap_valid_fraction>=.8
    for r in contrasts.itertuples():
        g=metrics.loc[metrics.order.eq(r.order)&metrics.horizon.eq(r.horizon)&metrics.policy.eq(r.policy)
            &metrics.scope_filter.eq(r.scope_filter)&metrics.context.eq(r.context)&metrics.metric.eq(r.metric)].set_index("pattern")
        np.testing.assert_allclose(r.difference,g.loc[r.pattern,"estimate"]-g.loc[r.reference,"estimate"],atol=1e-12,equal_nan=True)
    # Independent reproduction of primary ABCD-ABAB-related paired intervals.
    blocks,weights=hierarchical_weights(frame,config["bootstrap_repetitions"],config["seed"])
    main=contrasts.loc[contrasts.order.eq(3)&contrasts.horizon.eq("possession")&contrasts.policy.eq("all_linked")
        &contrasts.scope_filter.eq("all_periods")&contrasts.pattern.eq("ABAB")&contrasts.context.eq("raw")
        &contrasts.metric.isin(["shot_rate","xg_per_motif","xg_per_shot"])]
    for r in main.itertuples():
        draws=[]
        for pattern in [r.pattern,r.reference]:
            known=frame.loc[frame.order.eq(3)&frame.horizon.eq("possession")&frame.pattern.eq(pattern)&frame.xg_observed].copy()
            known["numerator"]=known.shot_any*100 if r.metric=="shot_rate" else known.xg_sum
            known["denominator"]=known.shot_count if r.metric=="xg_per_shot" else 1
            by=known.groupby("possession_key")[["numerator","denominator"]].sum().reindex(blocks.possession_key).fillna(0)
            denom=weights@by.denominator.to_numpy()
            draws.append(np.divide(weights@by.numerator.to_numpy(),denom,out=np.full(len(weights),np.nan),where=denom>0))
        np.testing.assert_allclose(np.nanquantile(draws[0]-draws[1],[.025,.975]),[r.ci95_low,r.ci95_high],atol=1e-12)
    # Identity holds even when a shooting motif contains multiple attempts.
    for (order,pattern),g in frame.loc[frame.horizon.eq("possession")&frame.xg_observed].groupby(["order","pattern"]):
        np.testing.assert_allclose(g.xg_sum.mean(),g.shot_count.mean()*(g.xg_sum.sum()/g.shot_count.sum()),atol=1e-12)
    print("Every metric/contrast, source-xG decomposition and three primary paired ratio intervals reproduced.",flush=True)
    index=json.loads((out/"figure_index.json").read_text())
    for item in index:
        p=out/"figures"/item["file"]
        with Image.open(p) as im:im.verify()
        ET.parse(p.with_suffix(".svg"))
    dictionary=pd.read_csv(out/"data_dictionary.csv")
    assert dictionary.description.notna().all() and not dictionary.duplicated(["file","column"]).any()
    for p,g in dictionary.groupby("file"):assert set(pd.read_csv(ROOT/p,nrows=1))==set(g.column)
    for ref in re.findall(r'(?:href|src)="([^"]+)"',(out/"report.html").read_text(encoding="utf-8")):
        if "://" not in ref and not ref.startswith("#"):assert(out/ref).exists(),ref
    for doc in [ROOT/"README.md",*sorted((ROOT/"docs").glob("*.md"))]:
        for ref in re.findall(r"\]\(([^)]+)\)",doc.read_text(encoding="utf-8")):
            if "://" not in ref and not ref.startswith("#"):
                target=(doc.parent/ref.split("#")[0]).resolve()
                if target!=(out/"delivery_check.json").resolve():assert target.exists(),(doc,ref)
    result={"status":"passed","source_windows_checked":len(windows),"source_shot_links_checked":len(source_links),
        "motif_horizon_rows":len(frame),"motif_shot_links":len(links),"metrics_checked":len(metrics),
        "contrasts_checked":len(contrasts),"independent_ratio_CIs_recomputed":len(main),
        "converged_fits":len(audit),"outer_q_models":80,"figures_png_svg":len(index),"documented_columns":len(dictionary),
        "unit_suite":f"{unittest.TestLoader().discover(str(ROOT/'tests')).countTestCases()} tests; run separately",
        "zero_policy":"Zero only for observed no-shot outcomes; missing/censored remains missing.",
        "prior_experiments":"Preserved unchanged."}
    write_json(out/"delivery_check.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    with threadpool_limits(1):main()
