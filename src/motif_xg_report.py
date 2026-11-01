"""Reviewable shot-frequency and source-xG results for completed passing motifs."""
from pathlib import Path
import html
import importlib.metadata
import json
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analysis import configure_style,SOURCE_COMMIT
from .sequences import sha256
from .study import write_json

BLUE,RED,TEAL,GOLD="#3479ad","#ce6853","#329384","#b38b35"
PATTERNS={2:["ABA","ABC"],3:["ABAB","ABAC","ABCA","ABCB","ABCD"]}
LABELS={"shot_rate":"Motifs with a later shot (%)","xg_per_motif":"Source xG per motif",
        "xg_per_shot":"Source xG per linked shot","shots_per_motif":"Shots per motif",
        "unique_xg_per_shot":"Source xG per unique shot"}
DIFF_LABELS={"shot_rate":"Shot-rate difference (percentage points)",
             "xg_per_motif":"Difference in source xG per motif","xg_per_shot":"Difference in source xG per linked shot"}


def select(frame,**kwargs):
    keep=pd.Series(True,index=frame.index)
    for key,value in kwargs.items():keep&=frame[key].eq(value)
    return frame.loc[keep]


def save(fig,out,index,stem,title,caption):
    fig.suptitle(title,x=.025,ha="left",fontsize=16,weight="bold",y=.985)
    fig.text(.025,.015,"EURO 2024 · StatsBomb Open Data · source shot xG · pointwise 95% match/possession bootstrap intervals",
             fontsize=8,color="#607587")
    fig.tight_layout(rect=(.01,.07,.99,.86 if fig.legends else .935),h_pad=2,w_pad=2)
    path=out/"figures"/(stem+".png")
    fig.savefig(path,dpi=155,bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"),bbox_inches="tight",metadata={"Date":None})
    plt.close(fig)
    index.append({"file":path.name,"title":title,"caption":caption})


def forest(ax,rows,point="estimate",positions=None,color=BLUE,label=None):
    positions=np.arange(len(rows)) if positions is None else np.asarray(positions)
    ax.hlines(positions,rows.ci95_low,rows.ci95_high,color=color,lw=1.7)
    ax.scatter(rows[point],positions,color=color,s=32,label=label,zorder=3)
    ax.grid(axis="x",alpha=.5)


def make_figures(out,metrics,contrasts,config):
    configure_style();plt.rcParams["svg.hashsalt"]="euro2024-motif-xg"
    index=[];(out/"figures").mkdir(parents=True,exist_ok=True)
    primary=select(metrics,horizon="possession",policy="all_linked",scope_filter="all_periods",context="raw")
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for row,order in enumerate([2,3]):
        for ax,metric in zip(axes[row],["shot_rate","xg_per_motif","xg_per_shot"]):
            g=select(primary,order=order,metric=metric).set_index("pattern").loc[PATTERNS[order]]
            forest(ax,g)
            ax.set_yticks(range(len(g)),[f"{p} · n={r.known_motifs:,}" for p,r in g.iterrows()],fontsize=8)
            ax.invert_yaxis();ax.set_xlabel(LABELS[metric]);ax.set_title(f"{order} passes",fontsize=10)
    save(fig,out,index,"01_frequency_value_quality","Frequency, total threat and shot quality are separate measures",
         "From the final receipt to the end of the same provider possession, stopping earlier at opponent control, goal or observed period end. All three panels use the same complete-observation cohort. No shot means xG=0; unknown observations stay missing. xG per shot is a ratio over shot links, not an average over non-shooting motifs.")
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    main=select(contrasts,horizon="possession",policy="all_linked",scope_filter="all_periods",context="raw",order=3)
    for ax,metric in zip(axes,["shot_rate","xg_per_motif","xg_per_shot"]):
        g=select(main,metric=metric).set_index("pattern").loc[PATTERNS[3][:-1]]
        forest(ax,g,"difference")
        ax.set_yticks(range(len(g)),[f"{p} − ABCD" for p in g.index]);ax.invert_yaxis()
        ax.axvline(0,color="#8999a7",ls="--");ax.set_xlabel(DIFF_LABELS[metric])
    save(fig,out,index,"02_three_pass_differences","What differs from four distinct players?",
         "Raw differences versus ABCD, with both sides jointly resampled. A difference in xG per motif may arise from shot frequency, number of shots and/or shot quality. The ratio xG/shot is recomputed inside every bootstrap replicate; intervals are pointwise, not simultaneous tests.")
    fig,axes=plt.subplots(1,2,figsize=(14,5.8))
    main=select(contrasts,horizon="possession",policy="all_linked",scope_filter="all_periods",order=3)
    for ax,metric in zip(axes,["shot_rate","xg_per_motif"]):
        for j,(spec,color,label) in enumerate([("raw",TEAL,"Raw"),
                ("initial_receipt",BLUE,"Initial receipt X"),("completion_geometry",RED,"Completion X + geometry")]):
            g=select(main,metric=metric,context=spec).set_index("pattern").loc[PATTERNS[3][:-1]]
            forest(ax,g,"difference",np.arange(4)+(j-1)*.15,color,label)
        ax.set_yticks(range(4),[f"{p} − ABCD" for p in PATTERNS[3][:-1]]);ax.invert_yaxis()
        ax.axvline(0,color="#8999a7",ls="--");ax.set_xlabel(DIFF_LABELS[metric])
    fig.legend(*axes[-1].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.53,.94),ncol=3,frameon=False,fontsize=8)
    save(fig,out,index,"03_context_adjusted_xg","Does the source-xG difference remain after context adjustment?",
         "New conditional q models fit source xG sum and shot occurrence separately for each pass length and horizon. Full-possession xG models do not consistently beat a training-mean MSE baseline; treat adjusted contrasts as exploratory sensitivity. Initial and completion adjustments answer different descriptive questions. Completion/geometry can remove part of the motif's spatial pathway. These are residual associations, not causal effects.")
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,metric in zip(axes,["shot_rate","xg_per_motif","xg_per_shot"]):
        for j,(horizon,color,label) in enumerate([("possession",BLUE,"Remaining possession"),("10s",RED,"Next 10 seconds")]):
            g=select(metrics,order=3,horizon=horizon,policy="all_linked",scope_filter="all_periods",context="raw",metric=metric)
            g=g.set_index("pattern").loc[PATTERNS[3]]
            forest(ax,g,positions=np.arange(5)+(j-.5)*.14,color=color,label=label)
        ax.set_yticks(range(5),PATTERNS[3]);ax.invert_yaxis();ax.set_xlabel(LABELS[metric])
    axes[-1].legend(fontsize=8)
    save(fig,out,index,"04_horizon_sensitivity","How the time horizon changes the observed opportunities",
         "Both horizons start at the final receipt and use the same new provider-possession boundary rule. Thus the 10-second values need not equal the earlier step-4 bounded-continuous-play results. Natural period end closes a remaining-possession window; an unresolved file tail is censored.")
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,order in zip(axes,[2,3]):
        for j,(metric,color,label) in enumerate([("xg_per_shot",BLUE,"Motif-associated shot links"),
                ("unique_xg_per_shot",RED,"Each physical shot once per pattern")]):
            g=select(primary,order=order,metric=metric).set_index("pattern").loc[PATTERNS[order]]
            forest(ax,g,positions=np.arange(len(g))+(j-.5)*.15,color=color,label=label)
        ax.set_yticks(range(len(PATTERNS[order])),PATTERNS[order]);ax.invert_yaxis()
        ax.set_xlabel("Mean source xG per shot");ax.set_title(f"{order} passes")
    fig.legend(*axes[-1].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.53,.94),ncol=2,frameon=False,fontsize=8)
    save(fig,out,index,"05_overlap_weighting","Does crediting overlapping motifs change apparent shot quality?",
         "The primary xG/shot denominator is the number of motif–shot links, which preserves the xG/motif decomposition. The sensitivity deduplicates UUIDs within each order/pattern/policy/horizon. The same physical shot may still support different patterns. Both use shared match/possession resampling; credits must not be added as independent chances.")
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    for ax,metric in zip(axes,["xg_per_motif","xg_per_shot"]):
        for j,(policy,color,label) in enumerate([("all_linked",BLUE,"All linked"),
                ("previous_event_5s",RED,"Previous event ≤5s"),("between_passes_5s",TEAL,"Between passes ≤5s")]):
            g=select(metrics,order=3,horizon="possession",policy=policy,scope_filter="all_periods",context="raw",metric=metric)
            g=g.set_index("pattern").loc[PATTERNS[3]]
            forest(ax,g,positions=np.arange(5)+(j-1)*.14,color=color,label=label)
        ax.set_yticks(range(5),PATTERNS[3]);ax.invert_yaxis();ax.set_xlabel(LABELS[metric])
    axes[-1].legend(fontsize=8)
    save(fig,out,index,"06_timing_sensitivity","Source-xG results under both five-second definitions",
         "All timing, no-carry and regulation-only results are retained in CSV. Small groups remain without intervals. Quality intervals require at least 20 unique shots across five matches; other metrics require 50 complete motifs across five matches. Fixed full-cohort q models are used for adjusted sensitivity subsets.")
    write_json(out/"figure_index.json",index)
    return index


NEW_DEFINITIONS={
 "shot_rate":"100 × fraction of fully observed motifs followed by at least one recorded source Shot.",
 "xg_per_motif":"Mean source xG sum per fully observed motif, with zero for observed no-shot cases.",
 "xg_per_shot":"Sum of source xG divided by linked Shot count; overlapping motif associations count separately.",
 "shots_per_motif":"Mean count of later recorded Shots per fully observed motif, including rebounds.",
 "unique_xg_per_shot":"Mean source xG after deduplicating physical Shot UUID within pattern/order/horizon.",
 "horizon":"possession: remainder of current provider possession; 10s: until receipt+10s or same possession end.",
 "sequence_id":"Original source sequence ID for the last completed pass; target landmark is its receipt.",
 "window_end_event_id":"Underlying possession-boundary UUID, even if a 10s deadline precedes that event.",
 "window_end_index":"Source index of audited boundary; inclusive only for a source goal Shot.",
 "window_end_clock_seconds":"Minimum of possession boundary time and horizon deadline, within source period.",
 "window_end_reason":"Underlying possession-end reason; a 10s deadline may occur earlier than this recorded boundary.",
 "receipt_quality":"Valid source final receipt/possession context; invalid context censors all primary labels.",
 "observation_complete":"Entire requested possession/time horizon is observed, including natural confirmed period end.",
 "xg_observed":"Complete horizon and no missing xG on any selected Shot; common denominator for all requested metrics.",
 "missing_shot_xg":"Count of selected Shot events lacking finite source xG; never imputed as zero.",
 "shot_any":"0/1 indicator of at least one later recorded Shot in common observed window; own goals excluded.",
 "shot_count":"Number of later source Shot events, including rebounds, in common observed window.",
 "xg_sum":"Sum of all later source-shot xG in common window; zero for observed no-shot motifs, can exceed one.",
 "observed_shot_count":"Number of source Shots actually seen, even in a censored window; not a complete target.",
 "observed_source_xg_sum":"Sum of finite source xG seen in window, even if incomplete; audit only, not an imputed full outcome.",
 "shot_event_ids":"All selected later Shot UUIDs, in source event order.",
 "own_goal_for_count":"Beneficiary own-goal annotations within window, separate from source Shot/xG metrics.",
 "contains_restart":"An own restart remains inside the same provider possession and requested horizon; audit flag.",
 "contains_stoppage":"A recorded foul/injury/out/substitution remains in same provider possession before horizon closes.",
 "common_observed":"Parent sequence/horizon has complete source-xG observation; incomplete links are audit-only.",
 "shot_event_id":"Physical source Shot UUID; may be associated with several motifs.",
 "shot_index":"Original source Shot event index.",
 "shot_clock_seconds":"Source Shot time, seconds within period.",
 "shot_xg":"Unmodified source StatsBomb xG for the physical shot.",
 "shot_type":"Source shot type; open-play and same-possession restart shots remain labelled.",
 "context":"raw, initial_receipt or completion_geometry. Adjusted values are OOF residual means/differences.",
 "metric":"shot_rate(percent), xg_per_motif(mean source sum), xg_per_shot(link ratio), shots_per_motif, unique_xg_per_shot.",
 "scope_filter":"all_periods includes extra time; regulation restricts periods1/2.",
 "known_motifs":"Complete xG-observed motifs; identical denominator for primary shot-rate and xG-per-motif metrics.",
 "known_matches":"Distinct matches among known motifs.",
 "linked_shots":"Number of motif–shot associations among known motifs; repeated physical shots count per association.",
 "unique_shots":"Number of distinct physical Shot UUIDs supporting this pattern in the stated cohort.",
 "unique_shot_matches":"Distinct matches containing those unique physical shots.",
 "unique_shot_mean_xg":"Shot mean after deduplicating source UUID within this pattern/horizon/policy/order.",
 "estimate":"Raw metric or adjusted residual mean. Shot rate uses percent/percentage points; xG metrics use source xG units.",
 "difference":"Pattern minus ABC (two passes) or ABCD (three passes), on the same metric scale.",
 "support_eligible":"At least50 observed motifs/5 matches; shot-quality metrics additionally20 unique shots/5 shot matches; >=80% valid draws.",
 "receipt_index":"Index of observed final source receipt, missing for duration fallback.",
 "last_pass_index":"Index of the final completed pass whose recipient defines the landmark.",
}


def dictionary(root):
    inherited={}
    for file in ["results/euro_sequences/data_dictionary.csv","results/memory/data_dictionary.csv","results/motifs/data_dictionary.csv"]:
        table=pd.read_csv(root/file)
        inherited.update(table.drop_duplicates("column").set_index("column").description.to_dict())
    records=[]
    for p in sorted((root/"results/motif_xg/tables").glob("*.csv*")):
        data=pd.read_csv(p,nrows=500)
        for col in data:
            definition=NEW_DEFINITIONS.get(col) or inherited.get(col)
            if col.startswith("pred_"):definition="Cross-fitted q expectation for "+col[5:]+", trained only on other matches of same horizon/pass-length cohort."
            if col.startswith("residual_"):definition="Observed source outcome minus cross-fitted q for "+col[9:]+"."
            if col.startswith("null_"):definition="Outer-training mean of "+col[5:]+" for baseline MSE, no held-out outcomes."
            if definition is None:raise ValueError(f"Undocumented xG column {p.name}:{col}")
            records.append({"file":p.relative_to(root).as_posix(),"column":col,"sample_inferred_dtype":str(data[col].dtype),
                "description":definition,"missing_policy":"Unknown/incomplete/inapplicable stays missing. Zero means a fully observed zero outcome."})
    pd.DataFrame(records).to_csv(root/"results/motif_xg/data_dictionary.csv",index=False)
    return records


def build_report(root,config):
    root=Path(root);out=root/"results/motif_xg"
    metrics=pd.read_csv(out/"tables/pattern_metrics.csv")
    contrasts=pd.read_csv(out/"tables/pattern_contrasts.csv")
    index=make_figures(out,metrics,contrasts,config)
    columns=dictionary(root)
    raw=select(metrics,horizon="possession",policy="all_linked",scope_filter="all_periods",context="raw")
    primary=raw.pivot(index=["order","pattern"],columns="metric",values="estimate").reset_index()
    primary=primary[["order","pattern","shot_rate","xg_per_motif","xg_per_shot","shots_per_motif","unique_xg_per_shot"]]
    primary.to_csv(out/"tables/primary_comparison.csv",index=False)
    # Rebuild after adding the compact reviewer-facing table.
    columns=dictionary(root)
    core=select(contrasts,order=3,horizon="possession",policy="all_linked",scope_filter="all_periods",
                context="completion_geometry",metric="xg_per_motif")
    model_metrics=pd.read_csv(out/"tables/outcome_model_metrics.csv")
    model_review=model_metrics.loc[model_metrics.outcome.eq("xg_sum"),
        ["order","horizon","context","mse","training_mean_baseline_mse"]]
    cards="".join(f'<section><h2>{html.escape(i["title"])}</h2><p>{html.escape(i["caption"])}</p>'
        f'<a href="figures/{i["file"]}"><img src="figures/{i["file"]}" alt="{html.escape(i["title"])}"></a>'
        f'<p><a href="figures/{Path(i["file"]).with_suffix(".svg").name}">SVG</a></p></section>' for i in index)
    links="".join(f'<li><a href="tables/{p.name}">{p.name}</a></li>' for p in sorted((out/"tables").glob("*.csv*")))
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Motifs: shot frequency and source-xG quality</title><style>body{{margin:0;background:#edf2f6;color:#20364c;font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1280px;margin:auto;padding:30px}}section{{background:white;padding:25px;border-radius:12px;margin:22px 0}}
h1{{font-size:34px;line-height:1.25}}h2{{font-size:23px}}img{{max-width:100%}}a{{color:#216b9a}}.logo{{width:175px}}
table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{padding:7px;border-bottom:1px solid #dce4eb;text-align:left}}.scroll{{overflow-x:auto}}
.note{{border-left:4px solid #ce6853;padding-left:18px}}</style><main>

<img class="logo" src="../../assets/statsbomb-logo.png" alt="StatsBomb"><h1>More shots or better chances?</h1>
<p>Source-shot xG · 51 EURO 2024 matches · 33,358 two-pass and 27,175 three-pass motifs</p>
<section><h2>Three complementary measurements</h2><p>Shot frequency records whether a later shot occurs. <strong>xG/motif</strong> averages the sum of source xG from all later shots, with zero for fully observed no-shot windows. <strong>xG/shot</strong> divides that xG sum by the number of linked shots, describing the quality of shots actually taken.</p>
<p>The primary horizon runs <strong>after the final receipt to the end of the same provider possession</strong>, stopping earlier at confirmed opponent control, goal or observed period end. Own restarts within the same provider possession would be retained and flagged. A 10-second sensitivity uses the same boundaries. This differs from the earlier bounded-continuous-play shot10 definition.</p>
<div class="scroll">{primary.to_html(index=False,border=0,float_format=lambda v:f"{v:.4f}")}</div>
<p>shot_rate is a percentage; xg_per_motif and xg_per_shot are means. The same complete-observation cohort is used. Exact n, coverage and intervals appear in pattern_metrics.csv. unique_xg_per_shot counts each physical shot once per pattern.</p>
<p><strong>xG/motif = shots/motif × xG/linked shot.</strong> With rebounds, shot frequency and shots/motif differ. Summed xG may exceed one and is not the probability of scoring at least one goal.</p>
<h2>Context-adjusted three-pass xG differences versus ABCD</h2><div class="scroll">{core[["pattern","reference","difference","ci95_low","ci95_high"]].to_html(index=False,border=0,float_format=lambda v:f"{v:.5f}")}</div>
<p>New q models fit source-xG sums and shot occurrence separately by motif length/horizon, with five outer/three inner whole-match folds. Source xG is supplied by StatsBomb; no new xG model is trained and xT is not mixed into this target. Initial-receipt and completion/geometry residual comparisons are separate. Features exclude pattern and player IDs.</p>
<p><strong>Model limitation:</strong> full-possession xG models do not consistently beat a training-mean baseline. Three-pass initial/completion MSE is 0.003995/0.003852 versus baseline 0.003816. Treat adjustment as exploratory; raw source metrics and cluster intervals are primary.</p>
<div class="scroll">{model_review.to_html(index=False,border=0,float_format=lambda v:f"{v:.6f}")}</div>
<p class="note">One physical shot can follow several motifs. Credits overlap and cannot be summed as independent chances. Shared 2,000-draw match/possession bootstrap weights preserve overlapping paths. Ratios and paired differences are recomputed in every draw. Deduplicated shot-quality estimates have their own intervals. These are pointwise associations, not causal effects.</p>
<p>Unknown outcomes or missing source xG are not recoded as zero. Observed Half End naturally closes the remaining-possession horizon; an unresolved file tail is censored. Own goals remain distinct, without invented shots/xG. Completed-motif comparisons exclude the cost of unsuccessful formation attempts.</p>
<p><a href="../../docs/MOTIF_XG.md">Exact definitions and results</a> · <a href="data_dictionary.csv">Columns</a> · <a href="run_manifest.json">Hashes and provenance</a> · <a href="../index.html">All results</a></p></section>
{cards}<section><h2>Complete exports</h2><ul>{links}</ul><p>Read CSV.gz directly with pandas. All timing, no-carry and regulation policies remain, with unsupported groups flagged. No family of individual tests or FDR claims was made.</p></section>
</main></html>'''
    (out/"report.html").write_text(report,encoding="utf-8")
    code=["src/motif_xg.py","src/motif_xg_report.py","scripts/run_motif_xg.py","scripts/check_motif_xg.py",
          "configs/motif_xg.json","requirements-memory-lock.txt"]
    write_json(out/"run_manifest.json",{"source_commit":SOURCE_COMMIT,
        "analysis":json.loads((out/"analysis_manifest.json").read_text()),
        "code":{p:sha256(root/p) for p in code},"figures":len(index),"documented_columns":len(columns),
        "outputs":{p.relative_to(root).as_posix():{"sha256":sha256(p),"bytes":p.stat().st_size}
            for p in out.rglob("*") if p.is_file() and p.name not in {"run_manifest.json","delivery_check.json"}},
        "environment":{"python":platform.python_version(),**{p:importlib.metadata.version(p)
            for p in ["pandas","numpy","scipy","scikit-learn","matplotlib"]}},
        "extension":"Source-shot xG evaluation of completed step-4 motifs; stages5–7 remain pending."})
    print(f"Motif xG report: {len(index)} PNG/SVG figures; {len(columns)} documented columns.",flush=True)
