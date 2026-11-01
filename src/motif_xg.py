"""Source-shot xG after completed motifs, without changing prior experiments."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .motifs import clean_records, motif_features, NUMERICAL
from .evaluation import load_study_inputs, assert_split, hierarchical_weights
from .sequences import sha256
from .study import write_json, study_key


def sequence_windows(tables, horizons=("possession","10s")):
    """Every label starts at the last receipt; preserve every source Shot UUID."""
    records, links = [], []
    source = tables["receiver_sequences"]
    for _, group in tables["events"].sort_values(["match_id","period","index"]).groupby(["match_id","period"],sort=False):
        rows=clean_records(group)
        pos={r["event_id"]:i for i,r in enumerate(rows)}
        n=len(rows)
        # Administrative annotations do not reset a possession. Real controls do.
        run_end=[n]*n
        for i in range(n-2,-1,-1):
            a,b=rows[i],rows[i+1]
            run_end[i]=i+1 if (a["possession_id"],a["possession_team_id"])!=(b["possession_id"],b["possession_team_id"]) else run_end[i+1]
        own_source=source.loc[source.a_event_id.isin(pos)]
        for s in clean_records(own_source):
            i=pos[s["a_event_id"]]
            receipt=pos.get(s["receipt_event_id"])
            begin=(receipt+1) if receipt is not None else i+1
            quality=(s["provider_possession_team_matches"] and not s["receipt_after_boundary"]
                     and s["receipt_completed"] is not False)
            stop=run_end[i]
            reason="provider_possession_change" if stop<n else "unresolved_file_end"
            inclusive=False
            for j in range(i+1,stop):
                r=rows[j]
                if r["event_type"]=="Half End":
                    stop,reason,inclusive=j,"observed_period_end",False
                    break
                if r["controlled_touch"] and r["team_id"]!=s["team_id"]:
                    stop,reason,inclusive=j,"confirmed_opponent_control",False
                    break
                if r["stoppage_reason"]=="goal":
                    stop,reason,inclusive=j,"goal",True
                    break
            stop_time=rows[stop]["clock_seconds"] if stop<n else max(r["clock_seconds"] for r in rows)
            terminal=rows[stop]["event_id"] if stop<n else None
            if stop<begin-1 or s["receipt_clock_seconds"]>stop_time+.001:
                quality=False
            after=[r for r in rows[begin:stop+int(inclusive)]
                   if r["clock_seconds"]>=s["receipt_clock_seconds"]-.001]
            for horizon in horizons:
                deadline=np.inf if horizon=="possession" else s["receipt_clock_seconds"]+10
                selected=[r for r in after if r["clock_seconds"]<=deadline+.001]
                closed=reason!="unresolved_file_end"
                time_reached=horizon!="possession" and stop_time+.001>=deadline
                complete=quality and (closed or time_reached)
                shots=[r for r in selected if r["team_id"]==s["team_id"] and r["event_type"]=="Shot"]
                missing_xg=sum(r["xg"] is None or not np.isfinite(r["xg"]) for r in shots)
                common=complete and missing_xg==0
                xg=sum(r["xg"] for r in shots) if common else np.nan
                any_shot=float(bool(shots)) if common else np.nan
                own_goals=[r for r in selected if r["event_type"] in {"Own Goal Against","Own Goal For"}
                           and r["goal_beneficiary_team_id"]==s["team_id"]]
                records.append({"sequence_id":s["sequence_id"],"match_id":s["match_id"],"period":s["period"],
                    "team_id":s["team_id"],"possession_key":s["possession_key"],"horizon":horizon,
                    "receipt_event_id":s["receipt_event_id"],"receipt_index":rows[receipt]["index"] if receipt is not None else None,
                    "receipt_clock_seconds":s["receipt_clock_seconds"],"last_pass_index":s["a_index"],
                    "window_end_event_id":terminal,"window_end_index":rows[stop]["index"] if stop<n else None,
                    "window_end_clock_seconds":min(stop_time,deadline),"window_end_reason":reason,
                    "receipt_quality":quality,"observation_complete":complete,"xg_observed":common,
                    "missing_shot_xg":missing_xg,"shot_any":any_shot,
                    "shot_count":len(shots) if common else np.nan,"xg_sum":xg,
                    "observed_shot_count":len(shots),"observed_source_xg_sum":sum(r["xg"] or 0. for r in shots),
                    "shot_event_ids":";".join(r["event_id"] for r in shots),
                    "own_goal_for_count":len(own_goals),
                    "contains_restart":any(r["stoppage_reason"]=="restart" for r in selected),
                    "contains_stoppage":any(r["stoppage_reason"] in {"foul_stoppage","injury_stoppage","substitution","ball_out","shot_out"} for r in selected)})
                for r in shots:
                    links.append({"sequence_id":s["sequence_id"],"horizon":horizon,"match_id":s["match_id"],
                        "period":s["period"],"team_id":s["team_id"],"possession_key":s["possession_key"],
                        "shot_event_id":r["event_id"],"shot_index":r["index"],"shot_clock_seconds":r["clock_seconds"],
                        "shot_xg":r["xg"],"shot_type":r["shot_type"],"common_observed":common})
    columns=["sequence_id","horizon","match_id","period","team_id","possession_key",
             "shot_event_id","shot_index","shot_clock_seconds","shot_xg","shot_type","common_observed"]
    return pd.DataFrame(records),pd.DataFrame(links,columns=columns)


def fit_q(train,test,y_train,y_test,source,spec,parameters,binary,max_iter):
    known=np.isfinite(y_train)
    a=motif_features(train.loc[known],source,spec)
    b=motif_features(test,source,spec)
    numeric=[c for c in a if c in NUMERICAL]
    categorical=[c for c in a if c not in numeric]
    enc=ColumnTransformer([("numeric",StandardScaler(),numeric),
        ("category",OneHotEncoder(handle_unknown="ignore",sparse_output=True),categorical)],sparse_threshold=1)
    X,Z=enc.fit_transform(a),enc.transform(b)
    target=np.asarray(y_train)[known]
    results,scores,audit={},{},[]
    for parameter in parameters:
        if binary and len(np.unique(target))<2:
            prediction=np.full(len(test),(target.sum()+.5)/(len(target)+1));iterations=0;converged=True
        elif binary:
            model=LogisticRegression(C=parameter,max_iter=max_iter,tol=1e-4)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always",ConvergenceWarning);model.fit(X,target.astype(int))
            prediction=model.predict_proba(Z)[:,1];iterations=int(model.n_iter_.max())
            converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
        else:
            model=Ridge(alpha=parameter,solver="lsqr",tol=1e-6,max_iter=1000)
            model.fit(X,target)
            # Multiple rebound shots can sum above one. Do not cap at one.
            prediction=np.maximum(0,model.predict(Z));iterations=int(np.max(model.n_iter_));converged=iterations<1000
        results[parameter]=prediction
        observed=np.isfinite(y_test)
        y,p=np.asarray(y_test)[observed],prediction[observed]
        loss=(-y*np.log(np.clip(p,1e-12,1))-(1-y)*np.log(np.clip(1-p,1e-12,1))) if binary else (y-p)**2
        scores[parameter]=float(loss.sum())
        audit.append({"parameter":parameter,"training_observed":len(target),"test_observed":int(observed.sum()),
            "encoded_features":X.shape[1],"iterations":iterations,"converged":converged})
    coverage=[{"feature":c,"unknown_rows":int((~b[c].isin(a[c].unique())).sum()),"test_rows":len(test),
        "training_categories":a[c].nunique()} for c in categorical]
    return results,scores,audit,coverage


def fit_fold(frame,source,record,config,base,progress):
    assert_split(record)
    result=frame.loc[frame.match_id.isin(record["test_match_ids"])].copy()
    result["fold"]=record["fold"]
    parameters,audits,coverages=[],[],[]
    for order in [2,3]:
        for horizon in config["horizons"]:
            cohort=frame.order.eq(order)&frame.horizon.eq(horizon)
            for spec in config["context_specs"]:
                for outcome in config["q_outcomes"]:
                    binary=outcome=="shot_any"
                    grid=base["logistic_C_grid"] if binary else base["ridge_alpha_grid"]
                    totals={p:0. for p in grid}
                    meta={"fold":record["fold"],"order":order,"horizon":horizon,"context":spec,"outcome":outcome}
                    for j,inner in enumerate(record["inner"]):
                        a=cohort&frame.match_id.isin(inner["train_match_ids"])
                        b=cohort&frame.match_id.isin(inner["test_match_ids"])
                        _,score,log,_=fit_q(frame.loc[a],frame.loc[b],frame.loc[a,outcome].to_numpy(),
                            frame.loc[b,outcome].to_numpy(),source,spec,grid,binary,base["max_iter"])
                        for p in grid:totals[p]+=score[p]
                        audits.extend({**meta,**r,"stage":"inner","inner_fold":j} for r in log)
                    chosen=min(totals,key=totals.get)
                    a=cohort&frame.match_id.isin(record["train_match_ids"])
                    b=cohort&frame.match_id.isin(record["test_match_ids"])
                    prediction,_,log,cov=fit_q(frame.loc[a],frame.loc[b],frame.loc[a,outcome].to_numpy(),
                        frame.loc[b,outcome].to_numpy(),source,spec,[chosen],binary,base["max_iter"])
                    result.loc[frame.loc[b].index,f"pred_{spec}_{outcome}"]=prediction[chosen]
                    result.loc[frame.loc[b].index,f"null_{outcome}"]=frame.loc[a,outcome].mean()
                    audits.extend({**meta,**r,"stage":"outer","inner_fold":-1} for r in log)
                    coverages.extend({**meta,**r} for r in cov)
                    parameters.append({**meta,"selected":chosen,"inner_loss_sum":totals[chosen]})
            progress(f"xG f{record['fold']} {order} passes {horizon}: both context baselines fitted")
    for spec in config["context_specs"]:
        for outcome in config["q_outcomes"]:
            result[f"residual_{spec}_{outcome}"]=result[outcome]-result[f"pred_{spec}_{outcome}"]
    return result,parameters,audits,coverages


def bootstrap_ratio(frame,selected,patterns,blocks,weights,numerator,denominator):
    known=selected&frame.xg_observed
    ids=np.flatnonzero(known.to_numpy())
    group=pd.Categorical(frame.pattern,categories=patterns).codes
    block=frame.possession_key.map(pd.Series(np.arange(len(blocks)),index=blocks.possession_key)).to_numpy(dtype=int)
    coordinates=(group[ids],block[ids])
    shape=(len(patterns),len(blocks))
    num=np.ones(len(ids)) if numerator is None else frame.iloc[ids][numerator].to_numpy(dtype=float)
    den=np.ones(len(ids)) if denominator is None else frame.iloc[ids][denominator].to_numpy(dtype=float)
    a=sparse.coo_matrix((num,coordinates),shape=shape).tocsr()
    b=sparse.coo_matrix((den,coordinates),shape=shape).tocsr()
    denominator_draws=b@weights.T
    draws=np.divide(a@weights.T,denominator_draws,out=np.full_like(denominator_draws,np.nan,dtype=float),where=denominator_draws>0)
    totals=np.asarray(a.sum(axis=1)).ravel();counts=np.asarray(b.sum(axis=1)).ravel()
    point=np.divide(totals,counts,out=np.full(len(patterns),np.nan),where=counts>0)
    return point,draws


def summarize(frame,links,config):
    blocks,weights=hierarchical_weights(frame,config["bootstrap_repetitions"],config["seed"])
    summary,contrasts=[],[]
    for order in [2,3]:
        patterns=config["patterns"][str(order)];reference=config["reference"][str(order)];ri=patterns.index(reference)
        for horizon in config["horizons"]:
            for policy in config["policies"]:
                for scope in config["scope_filters"]:
                    select=frame.order.eq(order)&frame.horizon.eq(horizon)&frame[policy]
                    if scope=="regulation":select&=frame.period.le(2)
                    selected=frame.loc[select]
                    link=links.loc[links.order.eq(order)&links.horizon.eq(horizon)&links.motif_id.isin(selected.loc[selected.xg_observed,"motif_id"])]
                    n=selected.groupby("pattern").size().reindex(patterns,fill_value=0)
                    known=selected.loc[selected.xg_observed]
                    nk=known.groupby("pattern").size().reindex(patterns,fill_value=0)
                    nm=known.groupby("pattern").match_id.nunique().reindex(patterns,fill_value=0)
                    unique=link.drop_duplicates(["pattern","shot_event_id"])
                    ns=unique.groupby("pattern").size().reindex(patterns,fill_value=0)
                    shot_matches=unique.groupby("pattern").match_id.nunique().reindex(patterns,fill_value=0)
                    raw_metrics=[("shot_rate","shot_any",None,100),("xg_per_motif","xg_sum",None,1),
                                 ("xg_per_shot","xg_sum","shot_count",1),("shots_per_motif","shot_count",None,1)]
                    effects={}
                    metadata={"order":order,"horizon":horizon,"policy":policy,"scope_filter":scope}
                    for metric,numerator,denominator,scale in raw_metrics:
                        point,draws=bootstrap_ratio(frame,select,patterns,blocks,weights,numerator,denominator)
                        effects[metric]=(point*scale,draws*scale)
                    dedup=unique.copy()
                    dedup["xg_observed"]=True
                    point,draws=bootstrap_ratio(dedup,pd.Series(True,index=dedup.index),patterns,blocks,weights,"shot_xg",None)
                    effects["unique_xg_per_shot"]=(point,draws)
                    for spec in config["context_specs"]:
                        for outcome,metric,scale in [("shot_any","shot_rate",100),("xg_sum","xg_per_motif",1)]:
                            point,draws=bootstrap_ratio(frame,select,patterns,blocks,weights,f"residual_{spec}_{outcome}",None)
                            effects[f"{spec}:{metric}"]=(point*scale,draws*scale)
                    for metric,(point,draws) in effects.items():
                        context,measure=metric.split(":") if ":" in metric else ("raw",metric)
                        supported=(nk>=config["minimum_observed_motifs"])&(nm>=config["minimum_matches"])
                        if measure in {"xg_per_shot","unique_xg_per_shot"}:
                            supported&=(ns>=config["minimum_unique_shots"])&(shot_matches>=config["minimum_matches"])
                        supported&=np.isfinite(draws).mean(axis=1)>=.8
                        for j,pattern in enumerate(patterns):
                            bounds=np.nanquantile(draws[j],[.025,.975])
                            dedup=unique.loc[unique.pattern.eq(pattern)]
                            group=known.loc[known.pattern.eq(pattern)]
                            summary.append({**metadata,"context":context,"metric":measure,"pattern":pattern,
                                "motifs":int(n[pattern]),"known_motifs":int(nk[pattern]),"known_matches":int(nm[pattern]),
                                "coverage":nk[pattern]/n[pattern] if n[pattern] else np.nan,
                                "linked_shots":int(group.shot_count.sum()),"unique_shots":int(ns[pattern]),
                                "unique_shot_matches":int(shot_matches[pattern]),
                                "unique_shot_mean_xg":dedup.shot_xg.mean(),"estimate":point[j],
                                "ci95_low":bounds[0] if supported.iloc[j] else np.nan,
                                "ci95_high":bounds[1] if supported.iloc[j] else np.nan,
                                "support_eligible":bool(supported.iloc[j]),
                                "bootstrap_valid_fraction":float(np.isfinite(draws[j]).mean())})
                        for j,pattern in enumerate(patterns):
                            if pattern==reference:continue
                            diff=draws[j]-draws[ri];bounds=np.nanquantile(diff,[.025,.975])
                            ok=bool(supported.iloc[j] and supported.iloc[ri])
                            contrasts.append({**metadata,"context":context,"metric":measure,"pattern":pattern,
                                "reference":reference,"difference":point[j]-point[ri],
                                "ci95_low":bounds[0] if ok else np.nan,"ci95_high":bounds[1] if ok else np.nan,
                                "support_eligible":ok,"bootstrap_valid_fraction":float(np.isfinite(diff).mean())})
    audit=[]
    for (order,horizon),rows in frame.groupby(["order","horizon"]):
        for spec in config["context_specs"]:
            for outcome in config["q_outcomes"]:
                known=rows.loc[rows.xg_observed]
                y=known[outcome];p=known[f"pred_{spec}_{outcome}"]
                audit.append({"order":order,"horizon":horizon,"context":spec,"outcome":outcome,
                    "known":len(known),"mean_outcome":y.mean(),"mean_prediction":p.mean(),
                    "mse":np.mean((y-p)**2),"training_mean_baseline_mse":np.mean((y-known[f"null_{outcome}"])**2)})
    return {"pattern_metrics":pd.DataFrame(summary),"pattern_contrasts":pd.DataFrame(contrasts),
            "outcome_model_metrics":pd.DataFrame(audit)}


def run_analysis(root,config,base,progress=print):
    root=Path(root);out=root/"results/motif_xg";(out/"tables").mkdir(parents=True,exist_ok=True)
    tables,hashes=load_study_inputs(root)
    parent=json.loads((root/"results/motifs/run_manifest.json").read_text())
    motif_path="results/motifs/tables/oof_motifs.csv.gz"
    assert sha256(root/motif_path)==parent["outputs"][motif_path]["sha256"]
    hashes[motif_path]=sha256(root/motif_path)
    for p in ["src/motifs.py","src/evaluation.py"]:
        expected=parent["code"].get(p) or parent["analysis"]["code"][p]
        assert sha256(root/p)==expected
    splits=json.loads((root/"results/motifs/split_manifest.json").read_text())["splits"]
    hashes["results/motifs/split_manifest.json"]=sha256(root/"results/motifs/split_manifest.json")
    key,provenance=study_key(root,{**config,"base":base},hashes,["src/motif_xg.py","src/motifs.py","src/evaluation.py","src/sequences.py"])
    cache=root/".runtime/motif_xg"/key;cache.mkdir(parents=True,exist_ok=True)
    progress("Building post-final-receipt, source-shot windows for the remaining possession and 10 seconds.")
    windows,source_links=sequence_windows(tables,config["horizons"])
    windows.to_csv(out/"tables/sequence_windows.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    source_links.to_csv(out/"tables/sequence_shot_links.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    motifs=pd.read_csv(root/motif_path,low_memory=False)
    keep=["motif_id","order","pattern","match_id","period","scope","team_id","team","opponent_id","possession_key",
          "initial_sequence_id","end_sequence_id","duration_seconds","net_progression_x","total_pass_length",
          "unique_zones_6x3","span_x","span_y","intervening_carries","intervening_dribbles","geometry_complete",
          "all_linked","previous_event_5s","between_passes_5s","no_carry_dribble"]
    label_columns=[c for c in windows if c not in {"match_id","period","team_id","possession_key"}]
    frame=motifs[keep].merge(windows[label_columns],left_on="end_sequence_id",right_on="sequence_id",validate="many_to_many").drop(columns="sequence_id")
    assert not frame.duplicated(["motif_id","horizon"]).any() and len(frame)==len(motifs)*len(config["horizons"])
    motif_links=motifs[["motif_id","order","pattern","end_sequence_id"]].merge(source_links,left_on="end_sequence_id",right_on="sequence_id",validate="many_to_many").drop(columns="sequence_id")
    motif_links.to_csv(out/"tables/motif_shot_links.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    predictions,parameters,audits,coverage=[],[],[],[]
    for record in splits:
        stem=f"f{record['fold']}"
        if (cache/f"{stem}.json").exists():
            meta=json.loads((cache/f"{stem}.json").read_text())
            result=pd.read_csv(cache/f"{stem}.csv.gz",low_memory=False)
            p,a,c=meta["parameters"],meta["audit"],meta["coverage"]
            progress(f"xG {stem}: restored hash-keyed checkpoint")
        else:
            with threadpool_limits(base["blas_threads"]):
                result,p,a,c=fit_fold(frame,tables["receiver_sequences"],record,config,base,progress)
            result.to_csv(cache/f"{stem}.csv.gz",index=False,compression={"method":"gzip","mtime":0})
            write_json(cache/f"{stem}.json",{"parameters":p,"audit":a,"coverage":c})
        predictions.append(result);parameters.extend(p);audits.extend(a);coverage.extend(c)
    oof=pd.concat(predictions,ignore_index=True)
    oof.to_csv(out/"tables/oof_motif_xg.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    with threadpool_limits(base["blas_threads"]):
        outputs=summarize(oof,motif_links,config)
    for name,data in outputs.items():data.to_csv(out/f"tables/{name}.csv",index=False)
    pd.DataFrame(parameters).to_csv(out/"tables/selected_parameters.csv",index=False)
    pd.DataFrame(audits).to_csv(out/"tables/fit_audit.csv",index=False)
    pd.DataFrame(coverage).to_csv(out/"tables/q_feature_coverage.csv",index=False)
    write_json(out/"split_manifest.json",{"splits":splits})
    write_json(out/"analysis_manifest.json",{**provenance,"run_key":key,
        "shot_quality":"Source StatsBomb xG; no xG model fitted here, no xT mixed into these targets.",
        "denominators":"Common complete xG-observed motif cohort for all three requested metrics; xG/shot = sum source xG / linked shot count.",
        "overlap":"Same physical shot may be linked to several earlier motifs. Unique-shot means also exported. Credits cannot be added as independent opportunities.",
        "inference":"2000 joint match/whole-possession bootstrap draws; fixed fitted q; pointwise CIs, no individual hypothesis-test family.",
        "natural_period_end":"A recorded Half End closes the remaining possession; unresolved file tail is censored."})
    write_json(out/"DONE.json",{"status":"complete","run_key":key})
    progress("xG: source metrics, joint contrasts and cross-fitted context adjustments exported.")
