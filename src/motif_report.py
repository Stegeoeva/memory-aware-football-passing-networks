"""Scientific figures and readable report for actual passing motifs (step 4)."""
from pathlib import Path
import html
import importlib.metadata
import json
import platform
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image, ImageOps, ImageDraw

from .analysis import configure_style, pitch, SOURCE_COMMIT
from .sequences import sha256
from .study import write_json

BLUE, RED, TEAL, GOLD = "#3479ad", "#ce6853", "#329384", "#b38b35"
COLORS = [BLUE, RED, TEAL, GOLD, "#8265a9"]
SPECS = {"initial_receipt":"Initial receipt X", "completion_geometry":"Completion X + geometry"}
OUTCOMES = {"R3":"R3 value / 100", "shot10":"Shot in 10 s (percentage points)",
            "box3":"New box entry in 3 actions (percentage points)",
            "survival3":"Retained or shot in 3 actions (percentage points)",
            "shot15_pass_start":"Shot in 15 s from last pass (percentage points)"}
POLICIES = {"all_linked":"All linked", "previous_event_5s":"Previous event ≤5 s",
            "between_passes_5s":"Between passes ≤5 s", "no_carry_dribble":"No carry/dribble"}
SHORT_NAMES={"Robin Aime Robert Le Normand":"Robin Le Normand","Fabián Ruiz Peña":"Fabián Ruiz",
    "Nicholas Williams Arthuer":"Nico Williams","Álvaro Borja Morata Martín":"Álvaro Morata",
    "Pedro González López":"Pedri","Daniel Olmo Carvajal":"Dani Olmo","Marc Cucurella Saseta":"Marc Cucurella",
    "Rodrigo Hernández Casccante":"Rodri","Rodrigo Hernández Cascante":"Rodri",
    "Lamine Yamal Nasraoui Ebana":"Lamine Yamal"}


def save(fig, out, index, stem, title, caption):
    fig.suptitle(title,x=.025,ha="left",fontsize=17,weight="bold",y=.985)
    fig.text(.025,.018,"EURO 2024 · StatsBomb Open Data · exploratory · pointwise 95% match/possession intervals",
             fontsize=8,color="#607587")
    fig.tight_layout(rect=(.01,.075,.99,.935),h_pad=2,w_pad=2)
    p=out/"figures"/(stem+".png")
    fig.savefig(p,dpi=150,bbox_inches="tight")
    fig.savefig(p.with_suffix(".svg"),bbox_inches="tight",metadata={"Date":None})
    plt.close(fig)
    index.append({"file":p.name,"title":title,"caption":caption})


def forest(ax, frame, column, positions, color, label=None):
    ax.hlines(positions,frame.ci95_low,frame.ci95_high,color=color,lw=1.7)
    ax.scatter(frame[column],positions,color=color,s=30,label=label,zorder=3)
    ax.axvline(0,color="#8797a4",ls="--",lw=.9)
    ax.grid(axis="x",alpha=.45)


def make_figures(root, data, config, source):
    out=root/"results/motifs"
    (out/"figures").mkdir(parents=True,exist_ok=True)
    configure_style()
    plt.rcParams["svg.hashsalt"]="euro2024-step4"
    index=[]
    frame=data["oof_motifs"]
    effects=data["pattern_effects"]
    contrasts=data["pattern_contrasts"]
    main=effects.loc[effects.policy.eq("all_linked") & effects.scope_filter.eq("all_periods")]
    primary=contrasts.loc[contrasts.policy.eq("all_linked") & contrasts.scope_filter.eq("all_periods")]
    fig,axes=plt.subplots(2,4,figsize=(13,6))
    patterns=["ABA","ABC","ABAB","ABAC","ABCA","ABCB","ABCD"]
    for ax,pattern in zip(axes.flat,patterns):
        nodes=list(dict.fromkeys(pattern))
        positions={c:p for c,p in zip(nodes,[(.1,.5),(.4,.8),(.75,.65),(.8,.2)])}
        for k,(a,b) in enumerate(zip(pattern[:-1],pattern[1:])):
            start,end=positions[a],positions[b]
            rad=.15 if k%2==0 else -.2
            patch=FancyArrowPatch(start,end,connectionstyle=f"arc3,rad={rad}",arrowstyle="-|>",
                mutation_scale=15,color=COLORS[k],linewidth=2,shrinkA=12,shrinkB=12)
            ax.add_patch(patch)
            x,y=(start[0]+end[0])/2,(start[1]+end[1])/2
            ax.text(x,y+(k-1)*.09,str(k+1),color=COLORS[k],fontsize=11,weight="bold")
        for c,(x,y) in positions.items():
            ax.scatter(x,y,s=260,color=BLUE,zorder=3);ax.text(x,y,c,color="white",ha="center",va="center",zorder=4)
        ax.set_title(pattern);ax.set_xlim(-.05,1);ax.set_ylim(0,1);ax.axis("off")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0,.7,"A/B/C/D label distinct players.\nRepeated letters mean the same ID.\nArrows 1–3 are real ordered passes.\n\nFuture outcomes start after\ncompletion of the final pass.",fontsize=10,va="top")
    save(fig,out,index,"01_motif_definitions","Two passes versus three passes",
         "Canonical labels discard names while preserving repeated-player identities. Two passes give ABA/ABC; three give five possible patterns. Carry/dribble links are recorded, not converted into extra passes.")

    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for ax,order in zip(axes,[2,3]):
        g=main.loc[main.order.eq(order) & main.context.eq("initial_receipt") & main.outcome.eq("R3")]
        g=g.set_index("pattern").loc[config["patterns"][str(order)]]
        ax.bar(g.index,g.frequency*100,color=COLORS[:len(g)])
        for j,r in enumerate(g.itertuples()):
            ax.text(j,r.frequency*100+1,f"{r.motifs:,}\n{r.frequency:.1%}",ha="center",fontsize=9)
        ax.set_ylim(0,110*g.frequency.max()+8);ax.set_ylabel("% of this pass-length cohort")
        ax.set_title(f"{order} passes · n={g.motifs.sum():,}");ax.grid(axis="y",alpha=.4)
    save(fig,out,index,"02_pattern_frequencies","How often does each pattern occur?",
         "All 51 matches, regulation and extra time; one row per actual rolling motif. Overlapping motifs are included and are not independent chances. Three-pass motifs are a selected subset of chains that survived through a third completed pass.")

    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for ax,outcome in zip(axes,["R3","shot10","box3"]):
        g=primary.loc[primary.order.eq(2)&primary.outcome.eq(outcome)]
        for j,(spec,color) in enumerate(zip(SPECS,[BLUE,RED])):
            r=g.loc[g.context.eq(spec)]
            forest(ax,r,"adjusted_difference_per100",[j],color,SPECS[spec])
        r=g.iloc[0]
        ax.hlines(2,r.raw_ci95_low,r.raw_ci95_high,color=TEAL,lw=1.7)
        ax.scatter(r.raw_difference_per100,2,color=TEAL,s=30)
        ax.set_yticks([0,1,2],["Initial X","Completion + geometry","Unadjusted"])
        ax.set_title(OUTCOMES[outcome],fontsize=10);ax.set_xlabel("ABA − ABC");ax.invert_yaxis()
    save(fig,out,index,"03_return_vs_third","Return to A versus a third player: subsequent outcomes",
         "Joint bootstrap contrasts on common OOF data; positive means higher outcome after ABA than ABC. Initial-X and completion-X residual contrasts answer different descriptive questions. Future outcomes start after B→C completes, never before the motif is identified.")

    fig,axes=plt.subplots(1,3,figsize=(15,6))
    for ax,outcome in zip(axes,["R3","shot10","box3"]):
        for j,(spec,color) in enumerate(zip(SPECS,[BLUE,RED])):
            g=primary.loc[primary.order.eq(3)&primary.outcome.eq(outcome)&primary.context.eq(spec)]
            g=g.set_index("pattern").loc[["ABAB","ABAC","ABCA","ABCB"]]
            forest(ax,g,"adjusted_difference_per100",np.arange(4)+(j-.5)*.17,color,SPECS[spec])
        ax.set_yticks(range(4),["ABAB − ABCD","ABAC − ABCD","ABCA − ABCD","ABCB − ABCD"])
        ax.set_title(OUTCOMES[outcome],fontsize=10);ax.set_xlabel("Residual group-mean difference");ax.invert_yaxis()
    axes[-1].legend(fontsize=8,loc="best")
    save(fig,out,index,"04_three_pass_contrasts","Three-pass patterns compared with four distinct players",
         "Same post-completion R3/shot/box outcomes and common fold splits. Effects are differences of group-mean OOF residuals, not causal effects, common-population standardized risks or GAM coefficients. Pointwise intervals are not simultaneous multiple-testing guarantees.")

    fig,axes=plt.subplots(1,2,figsize=(14,6.5))
    for ax,outcome in zip(axes,["R3","shot15_pass_start"]):
        for j,(policy,color) in enumerate(zip(config["policies"],COLORS)):
            g=contrasts.loc[contrasts.order.eq(3)&contrasts.policy.eq(policy)&contrasts.scope_filter.eq("all_periods")
                           &contrasts.context.eq("completion_geometry")&contrasts.outcome.eq(outcome)]
            g=g.set_index("pattern").loc[["ABAB","ABAC","ABCA","ABCB"]]
            forest(ax,g,"adjusted_difference_per100",np.arange(4)+(j-1.5)*.12,color,POLICIES[policy])
        ax.set_yticks(range(4),["ABAB − ABCD","ABAC − ABCD","ABCA − ABCD","ABCB − ABCD"])
        ax.set_xlabel(OUTCOMES[outcome]);ax.invert_yaxis()
    axes[-1].legend(fontsize=8,loc="best")
    save(fig,out,index,"05_timing_sensitivity","Timing and carry-policy sensitivity",
         "Five seconds after the previous recorded event differs from five seconds between pass starts. Both filters are audited. All policies reuse the full-cohort OOF q models, without subset refits; different subsets can have different residual expectations. Markers without intervals have insufficient outcome/match support. The 15-second target clocks from the final pass start and counts beneficiary own goals separately.")

    fig,ax=plt.subplots(figsize=(11,10))
    team=data["team_frequencies"].loc[lambda f:f.order.eq(3)]
    pivot=team.pivot(index="team",columns="pattern",values="frequency").reindex(columns=config["patterns"]["3"]).sort_index()
    sizes=team.groupby("team").motifs.sum()
    im=ax.imshow(pivot.to_numpy()*100,cmap="Blues",vmin=0,vmax=70,aspect="auto")
    ax.set_yticks(range(len(pivot)),[f"{t} · n={sizes[t]:,}" for t in pivot.index],fontsize=9)
    ax.set_xticks(range(5),pivot.columns)
    for y in range(len(pivot)):
        for x in range(5):
            value=pivot.iloc[y,x]*100
            ax.text(x,y,f"{value:.1f}%",ha="center",va="center",color="white" if value>40 else "#20364c",fontsize=8)
    fig.colorbar(im,ax=ax,label="% of team's observed three-pass motifs")
    save(fig,out,index,"06_team_styles","Team motif profiles",
         "Team-specific shares sum to 100%. Alphabetical order; raw style frequencies do not rank team strength, control for opponents or adjust for unequal match counts.")

    geometry=[]
    for (order,pattern),rows in frame.groupby(["order","pattern"]):
        for field in ["duration_seconds","net_progression_x","unique_zones_6x3","total_pass_length"]:
            known=rows[field].dropna()
            geometry.append({"order":order,"pattern":pattern,"field":field,"known":len(known),
                "median":known.median(),"q25":known.quantile(.25),"q75":known.quantile(.75)})
    geo=pd.DataFrame(geometry);geo.to_csv(out/"tables/pattern_geometry.csv",index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,(field,label) in zip(axes.flat,[("duration_seconds","Start-to-final-receipt duration (s)"),
            ("net_progression_x","Net forward displacement (StatsBomb x units)"),
            ("unique_zones_6x3","Distinct zones of pass endpoints (6 × 3)"),
            ("total_pass_length","Sum of recorded pass lengths")]):
        g=geo.loc[geo.order.eq(3)&geo.field.eq(field)].set_index("pattern").loc[config["patterns"]["3"]]
        ax.vlines(range(5),g.q25,g.q75,color=BLUE,lw=4)
        ax.scatter(range(5),g["median"],color=RED,s=35,zorder=3)
        ax.set_xticks(range(5),g.index);ax.set_ylabel(label);ax.grid(axis="y",alpha=.4)
    save(fig,out,index,"07_pattern_geometry","Patterns differ in where and how they move the ball",
         "Medians and interquartile ranges, not confidence intervals. Geometry is observed ball-event information, not tracking or off-ball player locations. Completion adjustment includes these quantities and may remove part of the pathway through which a motif is useful.")

    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,order in zip(axes,[2,3]):
        g=main.loc[main.order.eq(order)&main.context.eq("initial_receipt")]
        for j,(outcome,color) in enumerate(zip(["R3","shot10","box3","shot15_pass_start"],COLORS)):
            k=g.loc[g.outcome.eq(outcome)].set_index("pattern").loc[config["patterns"][str(order)]]
            ax.bar(np.arange(len(k))+(j-1.5)*.18,100*k.coverage,width=.18,color=color,label=outcome)
        ax.set_xticks(range(len(config["patterns"][str(order)])),config["patterns"][str(order)])
        ax.set_ylim(0,104);ax.set_ylabel("% known after completed motif");ax.set_title(f"{order} passes")
    axes[-1].legend(loc="lower right",fontsize=8)
    save(fig,out,index,"08_outcome_coverage","Observation coverage by pattern and outcome",
         "Unknown or unresolved horizons remain missing. New box entry additionally requires the final receipt to be outside the box. Different coverage can change comparisons; raw counts, denominators, model-category coverage and censoring audits remain in the CSV exports.")

    # Exact observed examples from the three original Spain matches.
    events=source["events"].set_index("event_id")
    fig,axes=plt.subplots(3,2,figsize=(16,15))
    examples=[]
    for row,mid in enumerate([3942226,3942752,3943043]):
        for col,order in enumerate([2,3]):
            ax=axes[row,col]
            candidates=frame.loc[frame.match_id.eq(mid)&frame.team.eq("Spain")&frame.order.eq(order)&frame.shot15_event_ids.notna()]
            if order==2: candidates=candidates.loc[candidates.pattern.eq("ABA")]
            if candidates.empty:
                candidates=frame.loc[frame.match_id.eq(mid)&frame.team.eq("Spain")&frame.order.eq(order)]
            r=candidates.sort_values(["period","first_pass_clock_seconds","motif_id"]).iloc[0]
            pitch(ax)
            ids=r.pass_event_ids.split(";")
            for j,key in enumerate(ids):
                e=events.loc[key]
                arrow=FancyArrowPatch((e.x,e.y),(e.end_x,e.end_y),connectionstyle=f"arc3,rad={.08 if j%2==0 else -.1}",
                    color=COLORS[j],arrowstyle="-|>",mutation_scale=15,linewidth=2)
                ax.add_patch(arrow)
                ax.scatter([e.x,e.end_x],[e.y,e.end_y],color=COLORS[j],s=22,zorder=4)
                ax.text((e.x+e.end_x)/2,(e.y+e.end_y)/2-2,str(j+1),fontsize=10,weight="bold",color=COLORS[j],
                        bbox={"facecolor":"white","alpha":.85,"edgecolor":"none","pad":1})
            shot_id=str(r.shot15_event_ids).split(";")[0] if pd.notna(r.shot15_event_ids) else None
            if shot_id and shot_id in events.index:
                shot=events.loc[shot_id]
                ax.scatter(shot.x,shot.y,marker="*",s=180,color=GOLD,zorder=6)
                ax.text(shot.x-2,shot.y+7,f"later shot xG={shot.xg:.2f}",fontsize=9,ha="right")
            t=r.first_pass_clock_seconds
            ax.set_title(f"{r.pattern} · period {r.period}, {int(t//60):02d}:{int(t%60):02d}",loc="left",weight="bold")
            opponent=source["receiver_sequences"].loc[lambda f:f.sequence_id.eq(r.initial_sequence_id)].opponent.iloc[0]
            chain=" → ".join(SHORT_NAMES.get(n,n) for n in r.player_names.split(" → "))
            ax.text(0,90,f"Spain vs {opponent}\n"+textwrap.fill(chain,66),fontsize=8,va="top")
            examples.append({"motif_id":r.motif_id,"match_id":mid,"order":order,"pattern":r.pattern,
                "pass_event_ids":r.pass_event_ids,"later_shot_event_id":shot_id,"selection":"earliest Spain example with shot within final-pass 15s; ABA restriction for 2 passes; first available fallback"})
    pd.DataFrame(examples).to_csv(out/"tables/illustrative_examples.csv",index=False)
    save(fig,out,index,"09_observed_examples","Observed examples from the three Spain case studies",
         "Pass arrows use source origins/endpoints; numbers give chronological order. A gold star marks an observed later shot; the unplotted movement between the motif and shot is not inferred. Examples are deterministic illustrations, selected as the earliest shot-associated ABA/two-pass or three-pass motif, not a representative sample.")

    # A transparent descriptive view within initial context strata.
    strata=data["context_strata"].loc[lambda f:f.order.eq(2)&f.outcome.eq("shot10")&f.stratum.isin(["initial_receipt_third","initial_pressure"])]
    records=[]
    for (stratum,value),rows in strata.groupby(["stratum","stratum_value"]):
        if set(rows.pattern)=={"ABA","ABC"}:
            r=rows.set_index("pattern")
            records.append({"label":f"{stratum.replace('initial_','')}: {value}",
                "raw":r.loc["ABA","raw_per100"]-r.loc["ABC","raw_per100"],
                "adjusted":r.loc["ABA","completion_adjusted_per100"]-r.loc["ABC","completion_adjusted_per100"],
                "n_aba":r.loc["ABA","known"],"n_abc":r.loc["ABC","known"]})
    view=pd.DataFrame(records)
    fig,ax=plt.subplots(figsize=(12,5.5))
    ax.scatter(view.raw,np.arange(len(view))-.08,color=TEAL,s=40,label="Raw difference")
    ax.scatter(view.adjusted,np.arange(len(view))+.08,color=RED,s=40,label="Completion-adjusted residual difference")
    ax.set_yticks(range(len(view)),[f"{r.label} · n={int(r.n_aba)}/{int(r.n_abc)}" for r in view.itertuples()],fontsize=9)
    ax.axvline(0,color="#8797a4",ls="--");ax.set_xlabel("ABA − ABC: shot within 10 s (percentage points)")
    ax.invert_yaxis();ax.legend(fontsize=8)
    save(fig,out,index,"10_context_strata","Return patterns in different receipt contexts",
         "Descriptive subgroup means with known-outcome counts (ABA/ABC), without subgroup significance tests or intervals. Do not infer significance from these dots. The primary joint bootstrap contrasts, not this selected stratification, are the uncertainty analysis.")
    write_json(out/"figure_index.json",index)
    sheet=Image.new("RGB",(1500,4*340),"#e8eef3");draw=ImageDraw.Draw(sheet)
    for j,item in enumerate(index):
        im=ImageOps.contain(Image.open(out/"figures"/item["file"]).convert("RGB"),(490,300))
        x,y=(j%3)*500,(j//3)*340
        sheet.paste(im,(x+(500-im.width)//2,y+25));draw.text((x+10,y+5),item["file"],fill="#20364c")
    sheet.save(out/"contact_sheet.jpg",quality=90)
    return index


DEFINITIONS={
 "motif_id":"Unique initial sequence plus m2/m3; one real rolling motif, not an independent chance.",
 "order":"Number of completed actual passes (2 or 3).",
 "pattern":"Canonical player-ID sequence: ABA/ABC or ABAB/ABAC/ABCA/ABCB/ABCD.",
 "initial_sequence_id":"Step-1 sequence ID of first pass; joins initial-receipt context.",
 "end_sequence_id":"Step-1 sequence ID of last pass; joins post-final-receipt outcomes.",
 "pass_event_ids":"Ordered source pass UUIDs, semicolon-separated.",
 "player_ids":"Ordered actual player IDs, semicolon-separated; final recipient is not an inferred failed-pass target.",
 "player_names":"Source names in motif order, including repeated players.",
 "first_pass_event_id":"Source UUID of first motif pass.",
 "last_pass_event_id":"Source UUID of final motif pass.",
 "first_pass_clock_seconds":"First pass start, seconds within source period.",
 "last_pass_clock_seconds":"Final pass start, seconds within source period; origin for 15-second paper-inspired target.",
 "completion_clock_seconds":"Final receipt time; source receipt or documented duration fallback.",
 "completion_receipt_event_id":"Final observed receipt UUID, if available.",
 "completion_receipt_observed":"Final pass has a linked source receipt event.",
 "completion_receipt_quality":"Final sequence has matching provider possession team, no after-boundary receipt and no incomplete receipt.",
 "gap_between_pass_starts":"Adjacent pass-start differences in seconds, semicolon-separated.",
 "max_pass_start_gap":"Largest adjacent motif pass-start gap, seconds.",
 "previous_event_ids":"Immediately previous recorded source UUID for each pass; may include receipt/pressure/admin events.",
 "previous_event_types":"Types of immediately preceding source records, preserving event-taxonomy ambiguity.",
 "gaps_to_previous_events":"Seconds since previous recorded event for every pass, including first.",
 "previous_event_5s":"Every motif pass starts within 5 seconds of immediately preceding recorded same-period event, with 1ms tolerance.",
 "between_passes_5s":"All adjacent motif pass starts separated by at most 5 seconds; distinct sensitivity.",
 "intervening_carries":"Own carries between first and final pass starts.",
 "intervening_dribbles":"Own successful dribbles between first and final pass starts.",
 "no_carry_dribble":"No carry/dribble between motif passes; linked receipt/pressure annotations may remain.",
 "all_linked":"Primary policy: all valid actual links with source-allowed carries/dribbles, no additional timing cutoff.",
 "initial_receipt_zone":"6×3 zone at first pass reception.",
 "initial_receipt_third":"First receipt x<40 defensive, 40–80 middle, x>=80 attacking.",
 "initial_pressure":"First receipt pressure state, including explicitly unobserved.",
 "completion_zone":"6×3 zone of final receipt.",
 "completion_third":"Final receipt x<40 defensive, 40–80 middle, x>=80 attacking.",
 "completion_pressure":"Final receipt pressure state.",
 "initial_receipt_x":"Source first receipt x, StatsBomb 120×80 units.",
 "initial_receipt_y":"Source first receipt y, StatsBomb 120×80 units.",
 "completion_x":"Source final receipt x.",
 "completion_y":"Source final receipt y.",
 "duration_seconds":"Final receipt minus first pass start; known at completed-motif landmark.",
 "net_progression_x":"Final receipt x minus first pass origin x; signed, not a reward for direction alone.",
 "total_pass_length":"Sum of recorded lengths of all motif passes; same StatsBomb spatial units.",
 "unique_zones_6x3":"Distinct zones among all pass origins/endpoints; missing if any coordinate unknown.",
 "span_x":"Maximum minus minimum x among all pass origins/endpoints; missing if incomplete geometry.",
 "span_y":"Maximum minus minimum y among all pass origins/endpoints; missing if incomplete geometry.",
 "geometry_complete":"All 2×order origins/endpoints are observed valid points.",
 "shot15_pass_start":"Known source shot within 15s of final pass start in bounded same-team suffix; beneficiary own goal also counts.",
 "shot15_source_only":"Same clock/observation rules but only recorded Shot events, excluding own-goal convention.",
 "shot15_own_goal_counted":"Beneficiary own goal inside 15s counted separately; no xG invented.",
 "shot15_event_ids":"Recorded Shot UUIDs inside 15s from final pass start.",
 "shot15_boundary_event_id":"Source boundary UUID for auditing 15-second label.",
 "context":"initial_receipt uses first-receipt X; completion_geometry uses final-receipt X plus observed path geometry; neither includes motif pattern or IDs.",
 "policy":"Timing/carry filter applied to fixed full-cohort OOF predictions; no sensitivity subset q refit.",
 "scope_filter":"all_periods includes regulation/extra time; regulation restricts periods1/2.",
 "motifs":"Number of actual completed motifs before outcome-specific missingness.",
 "frequency":"Motifs of this pattern divided by all motifs of same order/policy/scope, or team.",
 "known":"Number of non-censored targets for this pattern/outcome.",
 "possessions":"Number of distinct possession keys among known outcomes.",
 "coverage":"Known target count divided by motifs; outside-box restriction also reduces box3 coverage.",
 "raw_per100":"100 times known outcome mean; R3 proxy units or binary percentages.",
 "adjusted_excess_per100":"100 times group mean(outcome minus cross-fitted q), in group-specific context distribution.",
 "support_eligible":"At least 50 known outcomes and 5 observed matches; practical support threshold, not significance.",
 "reference":"ABC for two passes, ABCD for three; compared on same completed-length cohort.",
 "known_pattern":"Known outcomes of compared motif pattern.",
 "known_reference":"Known outcomes of reference pattern.",
 "raw_difference_per100":"Raw mean outcome of pattern minus reference, times100.",
 "adjusted_difference_per100":"Difference of pattern/reference OOF residual means, times100; descriptive association, not ATE.",
 "bootstrap_valid_fraction":"Fraction of 2000 joint hierarchical draws with nonempty denominator(s).",
 "ci95_low":"2.5th percentile of joint match/whole-possession bootstrap adjusted estimate/difference; fixed fitted models, pointwise.",
 "ci95_high":"97.5th percentile of same bootstrap; missing for unsupported groups.",
 "raw_ci95_low":"2.5th percentile of raw group mean or raw difference joint bootstrap.",
 "raw_ci95_high":"97.5th percentile of raw group mean or difference joint bootstrap.",
 "stratum":"Source context field used for descriptive subgroup means.",
 "stratum_value":"Observed level of descriptive subgroup.",
 "initial_adjusted_per100":"Subgroup mean initial-receipt OOF residual times100; no subgroup CI.",
 "completion_adjusted_per100":"Subgroup mean completion/geometry OOF residual times100; no subgroup CI.",
 "note":"Explicit scope/interpretation note.",
 "encoded_features":"Number of train-fitted encoded numeric/categorical features in this q fit.",
 "test_observed":"Known targets scored in this validation/test split.",
 "selected":"Nested-selected q hyperparameter: Ridge alpha for R3, logistic C for binary targets.",
 "inner_loss_sum":"Sum of inner-validation squared errors (R3) or log loss in natural units (binary).",
 "training_matches":"Distinct whole matches in this outer training split.",
 "mean_outcome":"Known-outcome OOF target mean.",
 "mean_prediction":"OOF q mean on exactly the known target observations.",
 "mse":"OOF mean squared prediction error; binary Brier for binary targets.",
 "field":"Observed geometry field summarized by quantiles.",
 "median":"Observed geometry median, not a confidence bound.",
 "q25":"25th percentile of observed geometry distribution, not a confidence bound.",
 "q75":"75th percentile of observed geometry distribution.",
 "later_shot_event_id":"Source shot UUID in the selected illustration, if present.",
 "selection":"Deterministic illustrative-example selection rule; not an inferential subsample.",
}


def make_dictionary(root):
    prior=pd.read_csv(root/"results/memory/data_dictionary.csv")
    inherited=prior.drop_duplicates("column").set_index("column").description.to_dict()
    source=pd.read_csv(root/"results/euro_sequences/data_dictionary.csv")
    inherited.update(source.drop_duplicates("column").set_index("column").description.to_dict())
    records=[]
    for p in sorted((root/"results/motifs/tables").glob("*.csv*")):
        sample=pd.read_csv(p,nrows=1000)
        for c in sample:
            definition=DEFINITIONS.get(c) or inherited.get(c)
            if c.startswith("pred_"): definition="OOF q expectation with "+c[5:]+", trained only on matching completed-length cohort of other matches."
            if c.startswith("residual_"): definition="Observed outcome minus "+c[9:]+" OOF q; missing when outcome censored."
            if c=="R3": definition="Post-final-receipt three-action state/terminal proxy using held-out fold train-only xT."
            if c is not None and definition is None: raise ValueError(f"Undocumented {p.name}:{c}")
            records.append({"file":p.relative_to(root).as_posix(),"column":c,"sample_inferred_dtype":str(sample[c].dtype),
                "description":definition,"missing_policy":"Empty means censored, unobserved or inapplicable; never an automatic zero."})
    pd.DataFrame(records).to_csv(root/"results/motifs/data_dictionary.csv",index=False)
    return records


def build_report(root,config):
    root=Path(root);out=root/"results/motifs"
    data={p.name.removesuffix(".csv.gz").removesuffix(".csv"):pd.read_csv(p,low_memory=False)
          for p in (out/"tables").glob("*.csv*")}
    from .evaluation import load_study_inputs
    source,_=load_study_inputs(root)
    index=make_figures(root,data,config,source)
    dictionary=make_dictionary(root)
    effects=data["pattern_effects"]
    contrasts=data["pattern_contrasts"]
    primary=contrasts.loc[contrasts.policy.eq("all_linked")&contrasts.scope_filter.eq("all_periods")
                         &contrasts.context.eq("completion_geometry")&contrasts.outcome.isin(["R3","shot10"])]
    count=data["oof_motifs"].groupby("order").size()
    cards="".join(f'<section><h2>{html.escape(i["title"])}</h2><p>{html.escape(i["caption"])}</p>'
        f'<a href="figures/{i["file"]}"><img src="figures/{i["file"]}" alt="{html.escape(i["title"])}"></a>'
        f'<p><a href="figures/{Path(i["file"]).with_suffix(".svg").name}">SVG</a></p></section>' for i in index)
    links="".join(f'<li><a href="tables/{p.name}">{p.name}</a></li>' for p in sorted((out/"tables").glob("*.csv*")))
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Real passing motifs</title><style>body{{margin:0;background:#edf2f6;color:#20364c;font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1280px;margin:auto;padding:30px}}section{{background:white;padding:25px;border-radius:12px;margin:22px 0}}
h1{{font-size:36px;line-height:1.2}}h2{{font-size:24px}}img{{max-width:100%}}a{{color:#216b9a}}.logo{{width:175px}}
table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{padding:7px;border-bottom:1px solid #dce4eb;text-align:left}}
.scroll{{overflow-x:auto}}.note{{border-left:4px solid #ce6853;padding-left:18px}}</style><main>

<img class="logo" src="../../assets/statsbomb-logo.png" alt="StatsBomb"><h1>Which real passing motifs have better continuation?</h1>
<p>EURO 2024 · 51 matches · {count[2]:,} two-pass motifs · {count[3]:,} three-pass motifs</p>
<section><h2>Actual sequences, not hypothetical graph paths</h2><p>Compare ABA/ABC and ABAB/ABAC/ABCA/ABCB/ABCD. Each pass has a source UUID. Failures, other handlers, opponent control, restarts and boundaries are not skipped to construct a chain.</p>
<p><strong>Future outcomes start after the final receipt.</strong> Passes needed to identify a completed motif are not counted as subsequent success. Failed attempts to form the motif are outside this conditional sample.</p>
<p>R3, shot10, retention/shot and new box entry reuse the continuation definitions from final receipt. A literature-inspired shot15 target starts at the last pass start, with own goals flagged separately. Pure source-shot xG is added in the <a href="../motif_xg/report.html">xG extension</a>.</p>
<h2>Two adjustment questions</h2><p><strong>Initial receipt:</strong> how does continuation compare with the usual outcome for its initial context? <strong>Completion + geometry:</strong> does a difference remain after final location/pressure/role and chain duration/progression/spread? The second can remove part of the spatial pathway through which a motif helps.</p>
<p>Separate conditional q models for two-/three-pass motifs exclude pattern and player IDs. They use five outer/three inner whole-match folds and the 20 verified training-only xT fits. Evaluated matches do not train their q or xT.</p>
<h2>Completion/geometry-adjusted differences</h2><div class="scroll">{primary[["order","pattern","reference","outcome","known_pattern","known_reference","adjusted_difference_per100","ci95_low","ci95_high"]].to_html(index=False,border=0,float_format=lambda v:f"{v:.4f}")}</div>
<p>Positive means higher than ABC/ABCD. Binary differences use percentage points; R3 uses proxy units per 100. Raw and initial-adjusted results, including negative and unsupported estimates, remain in the complete exports.</p>
<p class="note">Differences in mean OOF residuals are descriptive associations, not causal effects or player-quality grades. The 2,000 shared match/possession bootstrap draws keep overlapping paths together. Intervals are pointwise and conditional on fitted models. No individual significance-test family or FDR claims were made.</p>
<h2>Sensitivities</h2><p>Separate policies are all linked paths, ≤5 seconds after the previous recorded event, ≤5 seconds between pass starts, and no intermediate carries/dribbles. A previous event can be a pressure or receipt, so the two timing definitions differ. Subsets reuse full-cohort OOF models. Regulation-only results are included.</p>
<p><a href="{html.escape(config['paper_reference'])}">Håland et al. (2020)</a> uses a different provider, competition, GAM and feature set. This is a documented adaptation, not a reproduction.</p>
<p><a href="../../docs/MOTIFS.md">Methods and interpretation</a> · <a href="data_dictionary.csv">Every column</a> · <a href="run_manifest.json">Provenance</a> · <a href="../index.html">All results</a></p></section>
{cards}<section><h2>Full tables and audit</h2><ul>{links}</ul><p>CSV.gz loads directly in pandas. Canonical A/B/C/D labels retain actual identity order. Groups below 50 known outcomes or five matches remain without intervals. Infomap is future work.</p></section>
</main></html>'''
    (out/"report.html").write_text(report,encoding="utf-8")
    inputs=json.loads((out/"analysis_manifest.json").read_text())
    code=["src/motifs.py","src/motif_report.py","scripts/run_motifs.py","scripts/check_motifs.py",
          "configs/motifs.json","requirements-memory-lock.txt"]
    write_json(out/"run_manifest.json",{"source_commit":SOURCE_COMMIT,"analysis":inputs,
        "code":{p:sha256(root/p) for p in code},"figures":len(index),"documented_columns":len(dictionary),
        "outputs":{p.relative_to(root).as_posix():{"sha256":sha256(p),"bytes":p.stat().st_size}
                   for p in out.rglob("*") if p.is_file() and p.name not in {"run_manifest.json","delivery_check.json"}},
        "environment":{"python":platform.python_version(),**{p:importlib.metadata.version(p)
                      for p in ["pandas","numpy","scipy","scikit-learn","matplotlib"]}},
        "completed_roadmap_stage":4,"pending_roadmap_stages":[5,6,7]})
    print(f"Step 4 report: {len(index)} figures, {len(dictionary)} documented columns.",flush=True)
