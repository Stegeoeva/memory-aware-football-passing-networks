"""Validate source-linked motifs, completion landmarks, fits and joint intervals."""
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
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.evaluation import load_study_inputs,assert_split,hierarchical_weights
from src.continuation import base_outcomes,values_from_material
from src.sequences import sha256
from src.study import write_json


def main():
    out=ROOT/"results/motifs"
    run=json.loads((out/"run_manifest.json").read_text())
    config=json.loads((ROOT/"configs/motifs.json").read_text())
    assert run["analysis"]["config"] == {**config,"base":json.loads((ROOT/config["base_config"]).read_text())}
    for p,h in run["code"].items(): assert sha256(ROOT/p)==h,("code",p)
    for p,spec in run["outputs"].items():
        file=ROOT/p
        assert file.stat().st_size==spec["bytes"] and sha256(file)==spec["sha256"],("output",p)
    for p,h in run["analysis"]["inputs"].items():
        file=ROOT/p if (ROOT/p).is_file() else ROOT/"results/euro_sequences"/p
        assert sha256(file)==h,("input",p)
    tables,_=load_study_inputs(ROOT)
    frame=pd.read_csv(out/"tables/oof_motifs.csv.gz")
    assert frame.motif_id.is_unique and len(frame)==60533
    counts=frame.groupby("order").size().to_dict()
    assert counts=={2:33358,3:27175}
    events=tables["events"].set_index("event_id")
    sequences=tables["receiver_sequences"].set_index("sequence_id")
    paths=tables["two_pass_sequences"].loc[lambda f:f.memory_target_eligible].set_index("a_event_id")
    assert set(frame.loc[frame.order.eq(2),"initial_sequence_id"])==set(paths.sequence_id)
    previous={}
    for _,g in tables["events"].sort_values(["match_id","period","index"]).groupby(["match_id","period"]):
        ids=g.event_id.to_list()
        previous.update(dict(zip(ids[1:],ids[:-1])))
    event_rows=events.astype(object).where(events.notna(),None).to_dict("index")
    real_next_pass=paths.b_event_id.to_dict()
    first_pass=sequences.a_event_id.to_dict()
    pass_links_checked=0
    for r in frame.itertuples():
        ids=r.pass_event_ids.split(";")
        p=[event_rows[key] for key in ids]
        assert len(p)==r.order and all(e["event_type"]=="Pass" and e["completed"] and e["open_play"] for e in p)
        assert all(e["possession_key"]==r.possession_key for e in p)
        assert all(a["index"]<b["index"] for a,b in zip(p[:-1],p[1:]))
        players=[int(p[0]["player_id"])]+[int(e["recipient_id"]) for e in p]
        assert [e["recipient_id"] for e in p[:-1]]==[e["player_id"] for e in p[1:]]
        assert players==[int(v) for v in r.player_ids.split(";")]
        unique=list(dict.fromkeys(players))
        label="".join(chr(65+unique.index(player)) for player in players)
        assert label==r.pattern and label in config["patterns"][str(r.order)]
        # Every adjacent pair must be an existing audited real transition.
        for a,b in zip(ids[:-1],ids[1:]):
            assert real_next_pass[a]==b
            pass_links_checked+=1
        assert first_pass[r.initial_sequence_id]==ids[0]
        assert first_pass[r.end_sequence_id]==ids[-1]
        actual_previous=[previous.get(key) for key in ids]
        assert actual_previous==r.previous_event_ids.split(";")
        times=np.array([e["clock_seconds"] for e in p])
        event_gaps=times-np.array([event_rows[key]["clock_seconds"] for key in actual_previous])
        np.testing.assert_allclose(event_gaps,[float(v) for v in r.gaps_to_previous_events.split(";")],atol=1e-6)
        np.testing.assert_allclose(np.diff(times),[float(v) for v in r.gap_between_pass_starts.split(";")],atol=1e-6)
        assert r.previous_event_5s==bool(((event_gaps>=-.001)&(event_gaps<=5.001)).all())
        gap=np.diff(times)
        assert r.between_passes_5s==bool(((gap>=-.001)&(gap<=5.001)).all())
    print("Actual motifs: canonical IDs, all ordered pass links and both five-second definitions verified.",flush=True)
    base=base_outcomes(tables).set_index("sequence_id")
    for outcome in ["shot10","box3"]:
        np.testing.assert_allclose(frame[outcome],base.loc[frame.end_sequence_id,outcome],equal_nan=True)
    xt=json.loads((ROOT/"results/continuation/xt_models.json").read_text())
    splits=json.loads((out/"split_manifest.json").read_text())["splits"]
    assert splits==xt["split_manifest"]
    material=pd.read_csv(ROOT/"results/continuation/tables/outcome_material.csv.gz")
    for record in splits:
        assert_split(record)
        test=frame.loc[frame.fold.eq(record["fold"])]
        assert set(test.match_id)==set(record["test_match_ids"])
        assert set(test.match_id).isdisjoint(record["train_match_ids"])
        model=next(m for m in xt["models"] if m["level"]=="outer" and m["fold"]==record["fold"])
        assert model["training_match_ids"]==record["train_match_ids"]
        expected=values_from_material(material,model,test.end_sequence_id)["R3"]
        np.testing.assert_allclose(test.R3,expected,equal_nan=True,atol=1e-12)
    status=material.loc[material.horizon.eq(3)].set_index("sequence_id").status.map({"retained":1.,"shot":1.,"loss":0.})
    np.testing.assert_allclose(frame.survival3,frame.end_sequence_id.map(status),equal_nan=True)
    shot_links=0
    for r in frame.loc[frame.shot15_event_ids.notna()].itertuples():
        shots=events.loc[r.shot15_event_ids.split(";")]
        assert shots.event_type.eq("Shot").all() and shots.team_id.eq(r.team_id).all()
        assert shots.period.eq(r.period).all()
        assert shots.clock_seconds.between(r.completion_clock_seconds-.001,r.last_pass_clock_seconds+15.001).all()
        assert r.shot15_pass_start==1 and r.shot15_source_only==1
        shot_links+=len(shots)
    for spec in config["context_specs"]:
        for outcome in config["outcomes"]:
            pred=frame[f"pred_{spec}_{outcome}"]
            assert pred.notna().all() and pred.between(0,1).all()
            np.testing.assert_allclose(frame[f"residual_{spec}_{outcome}"],frame[outcome]-pred,equal_nan=True,atol=1e-12)
    audit=pd.read_csv(out/"tables/fit_audit.csv")
    assert len(audit)==1240 and audit.converged.all()
    assert len(audit.loc[audit.stage.eq("outer")])==100
    for r in audit.itertuples():
        record=splits[r.fold] if r.stage=="outer" else splits[r.fold]["inner"][r.inner_fold]
        expected=frame.loc[frame.match_id.isin(record["train_match_ids"])&frame.order.eq(r.order),r.outcome].notna().sum()
        assert r.training_observed==expected
    print("Post-completion targets: source shots, source windows, fold-specific R3, all residuals and 1,240 fits verified.",flush=True)
    effects=pd.read_csv(out/"tables/pattern_effects.csv")
    contrasts=pd.read_csv(out/"tables/pattern_contrasts.csv")
    for key,g in effects.groupby(["order","policy","scope_filter","context","outcome"]):
        order,policy,scope,spec,outcome=key
        selection=frame.order.eq(order)&frame[policy]
        if scope=="regulation": selection &= frame.period.le(2)
        selected=frame.loc[selection]
        for r in g.itertuples():
            rows=selected.loc[selected.pattern.eq(r.pattern)]
            known=rows.loc[rows[outcome].notna()]
            assert r.motifs==len(rows) and r.known==len(known)
            assert r.matches==known.match_id.nunique()
            np.testing.assert_allclose(r.frequency,len(rows)/len(selected),atol=1e-12)
            np.testing.assert_allclose(r.raw_per100,100*known[outcome].mean(),equal_nan=True,atol=1e-10)
            np.testing.assert_allclose(r.adjusted_excess_per100,100*known[f"residual_{spec}_{outcome}"].mean(),equal_nan=True,atol=1e-10)
            supported=r.known>=50 and r.matches>=5
            assert r.support_eligible==supported
            if not supported: assert pd.isna(r.ci95_low) and pd.isna(r.ci95_high)
    for r in contrasts.itertuples():
        group=effects.loc[effects.order.eq(r.order)&effects.policy.eq(r.policy)&effects.scope_filter.eq(r.scope_filter)
                          &effects.context.eq(r.context)&effects.outcome.eq(r.outcome)].set_index("pattern")
        a,b=group.loc[r.pattern],group.loc[r.reference]
        np.testing.assert_allclose(r.adjusted_difference_per100,a.adjusted_excess_per100-b.adjusted_excess_per100,atol=1e-10)
        np.testing.assert_allclose(r.raw_difference_per100,a.raw_per100-b.raw_per100,atol=1e-10)
    # Independent paired bootstrap calculation: shared draws, both groups together.
    blocks,weights=hierarchical_weights(frame,config["bootstrap_repetitions"],config["seed"])
    sample=contrasts.loc[contrasts.policy.eq("all_linked")&contrasts.scope_filter.eq("all_periods")
                        &contrasts.context.eq("completion_geometry")&contrasts.outcome.eq("shot10")]
    for r in sample.itertuples():
        draws=[]
        for pattern in [r.pattern,r.reference]:
            known=frame.loc[frame.order.eq(r.order)&frame.pattern.eq(pattern)&frame.shot10.notna()]
            by=known.groupby("possession_key").residual_completion_geometry_shot10.agg(["sum","count"]).reindex(blocks.possession_key).fillna(0)
            denom=weights@by["count"].to_numpy()
            draws.append(np.divide(100*(weights@by["sum"].to_numpy()),denom,
                out=np.full(len(weights),np.nan),where=denom>0))
        np.testing.assert_allclose([r.ci95_low,r.ci95_high],np.nanquantile(draws[0]-draws[1],[.025,.975]),atol=1e-10)
    print("All pattern summaries/contrasts verified; five principal joint-bootstrap contrasts independently reproduced.",flush=True)
    figures=json.loads((out/"figure_index.json").read_text())
    for item in figures:
        p=out/"figures"/item["file"]
        with Image.open(p) as image: image.verify()
        ET.parse(p.with_suffix(".svg"))
    dictionary=pd.read_csv(out/"data_dictionary.csv")
    assert dictionary.description.notna().all()
    assert not dictionary.duplicated(["file","column"]).any()
    for p,g in dictionary.groupby("file"):
        assert set(pd.read_csv(ROOT/p,nrows=1))==set(g.column)
    for ref in re.findall(r'(?:href|src)="([^"]+)"',(out/"report.html").read_text(encoding="utf-8")):
        if "://" not in ref and not ref.startswith("#"): assert (out/ref).exists(),ref
    for doc in [ROOT/"README.md",*sorted((ROOT/"docs").glob("*.md"))]:
        for ref in re.findall(r"\]\(([^)]+)\)",doc.read_text(encoding="utf-8")):
            if "://" not in ref and not ref.startswith("#"):
                target=(doc.parent/ref.split("#")[0]).resolve()
                if target!=(out/"delivery_check.json").resolve(): assert target.exists(),(doc,ref)
    result={"status":"passed","completed_roadmap_stage":4,"matches":51,"motifs_by_order":counts,
        "actual_adjacent_pass_links_checked":pass_links_checked,"source_shot_links_checked":shot_links,
        "train_only_xt_fits_reused":20,"outer_conditional_q_models":100,"audited_converged_fits":len(audit),
        "all_summary_rows_checked":len(effects),"all_contrast_rows_checked":len(contrasts),
        "independently_reproduced_joint_CIs":len(sample),"figures_png_svg":len(figures),
        "documented_columns":len(dictionary),"unit_test_suite":f"{unittest.TestLoader().discover(str(ROOT/'tests')).countTestCases()} tests, run separately",
        "previous_steps":"Numerical experiment outputs preserved; English presentation changes are recorded in release provenance."}
    write_json(out/"delivery_check.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    with threadpool_limits(1): main()
