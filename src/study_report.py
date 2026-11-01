"""Reviewable figures, tables and provenance for authorized roadmap steps 2/3."""
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
from PIL import Image

from .analysis import configure_style, SOURCE_COMMIT
from .evaluation import load_study_inputs, context_features
from .sequences import sha256
from .study import assemble_memory, write_json

BLUE, RED, TEAL, GOLD = "#3479ad", "#ce6853", "#329384", "#b38b35"
REPS = {"zones_6x3": "Zones 6 × 3", "zones_4x3": "Zones 4 × 3", "players": "Player identities"}
ROLES = {"defender": "Defenders", "defensive_midfielder": "Defensive midfielders",
         "attacking_midfielder_winger": "Attacking midfielders / wingers",
         "midfielder": "Midfielders", "goalkeeper": "Goalkeepers", "forward": "Forwards"}
ALIASES = {"Lamine Yamal Nasraoui Ebana": "Lamine Yamal",
           "Nicholas Williams Arthuer": "Nico Williams", "Daniel Olmo Carvajal": "Dani Olmo",
           "Kylian Mbappé Lottin": "Kylian Mbappé", "Daniel Carvajal Ramos": "Dani Carvajal",
           "Theo Bernard François Hernández": "Theo Hernández",
           "Kléper Laveran Lima Ferreira": "Pepe", "Fabian Lukas Schär": "Fabian Schär",
           "Manuel Obafemi Akanji": "Manuel Akanji", "Robin Aime Robert Le Normand": "Robin Le Normand",
           "Ricardo Iván Rodríguez Araya": "Ricardo Rodríguez",
           "Rafael Alexandre Conceição Leão": "Rafael Leão",
           "Aurélien Djani Tchouaméni": "Aurélien Tchouaméni",
           "Christian Dannemann Eriksen": "Christian Eriksen", "Marc Cucurella Saseta": "Marc Cucurella"}


def name(value):
    return ALIASES.get(value, value)


def forest(ax, frame, labels, *, color=BLUE):
    y = np.arange(len(frame))
    center = frame.adjusted_excess_per100.to_numpy()
    # A percentile interval need not contain its point estimate.
    ax.hlines(y, frame.ci95_low, frame.ci95_high, color=color, lw=1.5, alpha=.85)
    ax.scatter(center, y, color=color, s=28, zorder=3)
    ax.set_yticks(y, labels, fontsize=8)
    ax.axvline(0, color="#80909c", ls="--", lw=1)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.55)


def save_figure(fig, out, index, stem, title, caption):
    fig.suptitle(title, x=.035, ha="left", fontsize=16, fontweight="bold", y=.985)
    fig.text(.035, .015, "EURO 2024 · StatsBomb Open Data · exploratory · 95% cluster intervals where shown",
             fontsize=8, color="#607587")
    if fig.get_figheight() >= 10:
        fig.subplots_adjust(top=.945, bottom=.06)
    else:
        fig.subplots_adjust(top=.90, bottom=.12)
    png = out / "figures" / (stem + ".png")
    fig.savefig(png, dpi=155, bbox_inches="tight")
    fig.savefig(png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    index.append({"file": png.name, "title": title, "caption": caption})


def figures(root, memory, cont, config):
    out = root / "results/memory"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    configure_style()
    index = []
    comparisons = memory["memory_comparisons"]
    fig, ax = plt.subplots(figsize=(11, 5))
    labels, y = [], []
    for j, rep in enumerate(config["representations"]):
        for k, (comparison, color, label) in enumerate([
                ("M2_vs_M1", BLUE, "Counts"), ("M2_context_vs_M1_context", RED, "Shared context")]):
            r = comparisons.loc[comparisons.representation.eq(rep) & comparisons.comparison.eq(comparison)].iloc[0]
            pos = j*3+k
            ax.hlines(pos, r.ci95_low, r.ci95_high, color=color, lw=3)
            ax.scatter(r.gain_bits, pos, color=color, s=45)
            ax.annotate(f"{r.gain_bits:.4f} [{r.ci95_low:.4f}, {r.ci95_high:.4f}]",
                        (r.ci95_high, pos), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9)
            labels.append(f"{REPS[rep]} · {label}"); y.append(pos)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.axvline(0, color="#8999a7", ls="--")
    ax.set_xlim(-.008, .115); ax.set_xlabel("First-order log loss − second-order log loss (bits / pass); positive = improvement")
    ax.grid(axis="x", alpha=.5)
    save_figure(fig, out, index, "01_memory_gain", "Does the previous state help?",
                "Primary 5-fold OOF predictions; 33,358 completed A→B→C paths. Paired bootstrap of 51 whole matches, 2,000 replicates. Spatial memory and player identity memory are different questions.")

    metrics = memory["model_metrics"].loc[lambda f: f["repeat"].eq(0)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    models = ["M0", "M1", "M2", "M1_context", "M2_context"]
    for ax, rep in zip(axes, config["representations"]):
        g = metrics.loc[metrics.representation.eq(rep)].set_index("model").loc[models]
        bars = ax.bar(np.arange(5), g.log_loss_bits, color=["#b4c2cc", BLUE, TEAL, "#e0a094", RED])
        ax.set_xticks(range(5), ["M0", "M1", "M2", "M1+X", "M2+X"], rotation=20)
        for bar, value in zip(bars, g.log_loss_bits):
            ax.text(bar.get_x()+bar.get_width()/2, value+.025, f"{value:.3f}", ha="center", fontsize=8)
        ax.set_ylim(0, 4.45); ax.set_title(REPS[rep]); ax.grid(axis="y", alpha=.4)
    axes[0].set_ylabel("OOF log loss (bits / pass); lower is better")
    save_figure(fig, out, index, "02_model_losses", "Absolute prediction quality matters too",
                "M0/M1/M2 are smoothed counts; +X denotes common receipt context. Identity context models have worse log loss than the count models despite a small M2+X versus M1+X gain. Losses across different target alphabets are not direct comparisons.")

    team_names = cont["player_effects"].drop_duplicates("team_id").set_index("team_id").team.to_dict()
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    team = memory["memory_by_team"]
    for ax, rep in zip(axes, ["zones_6x3", "players"]):
        g = team.loc[team.representation.eq(rep) & team.comparison.eq("M2_context_vs_M1_context")].sort_values("gain_bits")
        pos = np.arange(len(g))
        ax.hlines(pos, g.ci95_low, g.ci95_high, color=BLUE)
        ax.scatter(g.gain_bits, pos, color=RED, s=22)
        ax.set_yticks(pos, [f"{team_names[r.team_id]}  n={r.observations:,}" for r in g.itertuples()], fontsize=8)
        ax.set_title(REPS[rep]); ax.axvline(0, color="#8999a7", ls="--")
        ax.set_xlabel("M2+X gain (bits / pass)"); ax.grid(axis="x", alpha=.45)
    fig.tight_layout(rect=(0, .06, 1, .93))
    save_figure(fig, out, index, "03_memory_by_team", "Memory gain by team",
                "Each team has only 3–7 matches. These are exploratory, pointwise intervals, not multiplicity-adjusted hypothesis tests or team rankings.")

    calibration = memory["calibration"].loc[lambda f: f["repeat"].eq(0)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, rep in zip(axes, config["representations"]):
        for model, color in [("M1_context", BLUE), ("M2_context", RED)]:
            g = calibration.loc[calibration.representation.eq(rep) & calibration.model.eq(model)]
            ax.plot(g.confidence, g.accuracy, ".-", color=color, label=model.replace("_context", "+X"))
            ax.scatter(g.confidence, g.accuracy, s=np.maximum(8, np.sqrt(g.n)), color=color, alpha=.6)
        ax.plot([0, 1], [0, 1], color="#a3afb8", ls="--"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(REPS[rep]); ax.set_xlabel("Top-choice confidence"); ax.legend(fontsize=8)
    axes[0].set_ylabel("Observed top-choice accuracy")
    save_figure(fig, out, index, "04_calibration", "Are predicted confidences reliable?",
                "Ten fixed confidence bins; marker area scales with √n. Sparse high-confidence bins can be unstable. Detailed counts and Brier/top-1/top-3 scores are exported.")

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, rep in zip(axes, config["representations"]):
        subset = memory["model_metrics"].loc[lambda f: f.representation.eq(rep)]
        for first, second, color, label in [("M1", "M2", BLUE, "Counts"),
                ("M1_context", "M2_context", RED, "Shared context")]:
            p = subset.pivot(index="repeat", columns="model", values="log_loss_bits")
            ax.plot(np.arange(3), p[first]-p[second], "o-", color=color, label=label)
        rolling = memory["rolling_metrics"].loc[lambda f: f.representation.eq(rep)].set_index("model")
        rolling_gain = rolling.loc["M1", "log_loss_bits"]-rolling.loc["M2", "log_loss_bits"]
        ax.scatter(3, rolling_gain, marker="D", s=60, color=TEAL, label="Rolling counts")
        ax.set_xticks(range(4), ["Split 1", "Split 2", "Split 3", "Rolling"], rotation=25)
        ax.axhline(0, color="#8999a7", ls="--"); ax.set_title(REPS[rep]); ax.grid(axis="y", alpha=.45)
    axes[0].set_ylabel("First-order loss − second-order loss (bits / pass)")
    axes[-1].legend(loc="best", fontsize=8)
    save_figure(fig, out, index, "05_split_sensitivity", "Split and chronological sensitivity",
                "Repeats are robustness checks, not independent samples. Rolling counts use only prior calendar dates and fixed alpha1=alpha2=10; they have different history and tuning from nested retrospective CV.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cover = memory["prediction_coverage"].loc[lambda f: f.model.eq("M2")]
    fields = ["unseen_B", "unseen_AB", "unseen_target", "no_group_history"]
    for j, rep in enumerate(config["representations"]):
        r = cover.loc[cover.representation.eq(rep)].iloc[0]
        axes[0].bar(np.arange(4)+(j-1)*.24, 100*r[fields].astype(float)/r.rows, width=.24,
                    label=REPS[rep], color=[BLUE, RED, TEAL][j])
    axes[0].set_xticks(range(4), ["Unseen B", "Unseen A,B", "Unseen target", "No team history"], rotation=22)
    axes[0].set_ylabel("% primary prediction rows"); axes[0].legend(fontsize=8)
    cover = cont["outcome_model_metrics"].loc[lambda f: ~f.outcome.isin(["R3_with_B", "R3_triple"])]
    axes[1].barh(cover.outcome, cover.coverage*100, color=BLUE)
    for j, r in enumerate(cover.itertuples()):
        axes[1].text(r.coverage*100+1, j, f"{r.observed:,}", va="center", fontsize=8)
    axes[1].set_xlim(0, 113); axes[1].set_xlabel("% of 41,927 initial passes with known outcome")
    axes[1].invert_yaxis()
    fig.tight_layout(rect=(0, .06, 1, .92))
    save_figure(fig, out, index, "06_coverage", "What is observed, and what needs backoff?",
                "Censored outcomes remain missing, never automatic failures. Box-entry coverage includes the restriction to receipts outside the box. Identity candidates are the nominal roster known at receipt, excluding B.")

    player = cont["player_effects"]
    primary = player.loc[player.metric.eq("R3") & player.ranking_eligible]
    roles = [r for r in ROLES if (primary.role.eq(r)).sum() >= 2]
    fig, axes = plt.subplots(len(roles), 1, figsize=(13, 3*len(roles)), squeeze=False)
    for ax, role in zip(axes[:, 0], roles):
        g = primary.loc[primary.role.eq(role)].sort_values("adjusted_excess_per100", ascending=False)
        # Show both ends of each role, always retaining all rows in the CSV.
        chosen = pd.concat([g.head(3), g.tail(2)]).drop_duplicates("a_id")
        forest(ax, chosen, [f"{name(r.a)} · {r.team} · n={r.observed_passes}, m={r.observed_matches}" for r in chosen.itertuples()])
        ax.set_title(f"{ROLES[role]} · {len(g)} supported players", fontsize=11, loc="left")
        ax.set_xlabel("Adjusted R3 excess / 100 completed initial passes")
    fig.tight_layout(rect=(0, .04, 1, .965))
    # Override the generic top spacing for the multi-panel forest.
    save_figure(fig, out, index, "07_player_effects", "Continuation after A's passes: within-role examples",
                "Three highest and two lowest point estimates per role, not proven winners/losers. Eligibility: ≥100 observed outcomes, ≥3 matches, ≥80% valid bootstrap draws. Whole-match/possession percentile intervals are pointwise and conditional on fitted q/xT. The lone supported forward has no supported forward peer.")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, table, title in zip(axes, ["pair_effects", "triple_effects"], ["A → B pairs", "Actual A → B → C triples"]):
        g = cont[table].loc[lambda f: f.ranking_eligible & f.metric.isin(["R3", "R3_triple"])]
        g = g.sort_values("adjusted_excess_per100", ascending=False).head(8)
        labels = []
        for r in g.itertuples():
            chain = name(r.a)+" → "+name(r.b)
            if table == "triple_effects": chain += " → "+name(r.c)
            labels.append("\n".join(textwrap.wrap(chain, 34))+f"\nn={r.observed_passes}, matches={r.observed_matches}")
        forest(ax, g, labels)
        ax.set_title(title); ax.set_xlabel("Adjusted R3 excess / 100 eligible paths")
    fig.tight_layout(rect=(0, .06, 1, .93))
    save_figure(fig, out, index, "08_pair_triple_effects", "Cooperation estimates need substantial uncertainty",
                "Top eight supported point estimates, with all estimates retained in CSV. Pairs require 30 outcomes/2 matches; triples 20/2. Triples condition on successful B→C and use a separately trained conditional q; their estimand differs from all initial A→B passes.")

    fig, ax = plt.subplots(figsize=(8, 7))
    right = player.loc[player.metric.eq("R3_with_B"), ["a_id", "adjusted_excess_per100"]]
    joined = primary.merge(right, on="a_id", suffixes=("_main", "_B"), validate="one_to_one")
    ax.scatter(joined.adjusted_excess_per100_main, joined.adjusted_excess_per100_B, s=26, color=BLUE, alpha=.7)
    limit = [min(joined.adjusted_excess_per100_main.min(), joined.adjusted_excess_per100_B.min())-.05,
             max(joined.adjusted_excess_per100_main.max(), joined.adjusted_excess_per100_B.max())+.05]
    ax.plot(limit, limit, ls="--", color="#8999a7")
    for r in joined.sort_values("adjusted_excess_per100_main", ascending=False).head(4).itertuples():
        ax.annotate(name(r.a), (r.adjusted_excess_per100_main, r.adjusted_excess_per100_B),
                    xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set_xlim(limit); ax.set_ylim(limit); ax.set_xlabel("Main q(X), without B identity")
    ax.set_ylabel("Sensitivity q(X, B), with B identity")
    save_figure(fig, out, index, "09_receiver_identity", "How much changes when receiver identity is controlled?",
                "Supported players only; both axes are adjusted R3 excess per 100. Including B removes part of the association with choosing that receiver; it is a different descriptive adjustment, not an automatic causal improvement.")

    # Keep the same chosen players across all secondary outcomes and horizons.
    chosen_ids = []
    for role in ["attacking_midfielder_winger", "defensive_midfielder", "defender", "midfielder", "goalkeeper"]:
        rows = primary.loc[primary.role.eq(role)].sort_values("adjusted_excess_per100", ascending=False)
        chosen_ids.extend(rows.head(1).a_id.tolist())
    chosen = primary.set_index("a_id").loc[chosen_ids].reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric, title in zip(axes, ["R3", "receiver_loss", "shot10"],
            ["R3 continuation / 100", "Receiver loss: percentage points", "Shot within 10 s: percentage points"]):
        g = chosen[["a_id"]].merge(player.loc[player.metric.eq(metric)], on="a_id", how="left", validate="one_to_one")
        forest(ax, g, [name(r.a)+f" (n={r.observed_passes})" for r in g.itertuples()],
               color=RED if metric == "receiver_loss" else BLUE)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Observed − receipt-context expectation")
    fig.tight_layout(rect=(0, .06, 1, .92))
    save_figure(fig, out, index, "10_secondary_outcomes", "A profile of outcomes, rather than one total grade",
                "The same five role-leading R3 point estimates are shown across outcomes. Lower receiver-loss excess is preferable; higher shot/R3 excess is preferable. Counts differ with outcome observability. These selected examples do not establish between-player significance.")

    fig, axes = plt.subplots(len(roles), 1, figsize=(12, 2.3*len(roles)), squeeze=False)
    for ax, role in zip(axes[:, 0], roles):
        g = primary.loc[primary.role.eq(role)].sort_values("adjusted_excess_per100", ascending=False).head(3)
        pos = np.arange(len(g))
        ax.hlines(pos, g.rank_ci95_low, g.rank_ci95_high, color=BLUE, lw=2)
        ax.scatter(g.role_rank, pos, color=RED, s=35)
        ax.set_yticks(pos, [f"{name(r.a)} · top 3 in {r.top3_frequency:.0%} of valid draws" for r in g.itertuples()], fontsize=8)
        ax.invert_yaxis(); ax.set_xlim(.5, g.role_peer_count.max()+.5)
        ax.set_title(ROLES[role]+f" · {int(g.role_peer_count.max())} peers", fontsize=10, loc="left")
        ax.set_xlabel("Point rank and 95% bootstrap rank range (1 = highest)")
    fig.tight_layout(rect=(0, .04, 1, .96))
    save_figure(fig, out, index, "11_rank_stability", "How stable are the apparent role leaders?",
                "Ranks are recomputed within each role for each hierarchical bootstrap draw, retaining fixed supported peers. Percentile rank endpoints can be fractional. Missing group draws are excluded. A small peer group makes top-three frequency easier to achieve.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for j, r in enumerate(chosen.itertuples()):
        g = player.loc[player.a_id.eq(r.a_id) & player.metric.isin(["R1", "R3", "R5"])].set_index("metric").loc[["R1", "R3", "R5"]]
        color = plt.cm.tab10(j)
        axes[0].plot([1, 3, 5], g.adjusted_excess_per100, "o-", color=color, label=name(r.a))
        axes[0].vlines([1, 3, 5], g.ci95_low, g.ci95_high, color=color, alpha=.5)
    axes[0].axhline(0, color="#8999a7", ls="--"); axes[0].set_xticks([1, 3, 5])
    axes[0].set_xlabel("Later ball actions"); axes[0].set_ylabel("Adjusted continuation excess / 100")
    axes[0].legend(fontsize=8)
    material = cont["outcome_material"]
    counts = pd.crosstab(material.horizon, material.status).reindex(columns=["retained", "shot", "loss", "censored"], fill_value=0)
    left = np.zeros(len(counts))
    for status, color in zip(counts, [BLUE, TEAL, RED, "#b4c2cc"]):
        values = counts[status].to_numpy()/counts.sum(axis=1).to_numpy()*100
        axes[1].barh(counts.index.astype(str), values, left=left, color=color, label=status)
        left += values
    axes[1].set_xlabel("% of all initial passes"); axes[1].set_ylabel("Action horizon")
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=4)
    save_figure(fig, out, index, "12_horizon_sensitivity", "The evaluation horizon changes the question",
                "Same selected players, separately cross-fitted R1/R3/R5 models. At the first shot the outcome becomes source xG; confirmed loss becomes zero; retained endpoints use train-only xT. Censoring grows with horizon; no missing value is recoded as zero.")
    write_json(out / "figure_index.json", index)
    return index


FIELD_DEFINITIONS = {
    "parameter": "Candidate Ridge alpha or logistic C used in this audited q fit.",
    "feature": "Receipt-time categorical q feature, fit only on training categories.",
    "training_categories": "Distinct known-outcome training categories for this feature/fold/outcome.",
    "unknown_rows": "Held-out rows with a category unseen by this q encoder.",
    "unknown_policy": "Unseen one-hot category maps to zeros; no test-time vocabulary fitting.",
    "representation": "State/target alphabet: zones_6x3, zones_4x3, or within-team player IDs.",
    "model": "M0 frequency; M1 current B; M2 previous A plus B; _context adds common receipt context X.",
    "repeat": "Grouped-split repeat 0/1/2; 0 is the primary result. Not an independent sample.",
    "fold": "Outer held-out whole-match fold (0–4). Both teams stay together.",
    "comparison": "Second model versus baseline on exactly the same OOF rows.",
    "gain_bits": "Baseline log loss minus second-model log loss; positive means better, bits per path.",
    "match_mean_gain_bits": "Equal-weight mean of per-match gain, rather than pass-weighted mean.",
    "observations": "Number of evaluated paths in the stated comparison.",
    "rows": "Number of evaluated rows in this group/model/repeat.",
    "matches": "Number of distinct observed matches.",
    "p_true": "Out-of-fold probability assigned to the actual target, after common candidate masking.",
    "log_loss_bits": "Minus log2 of actual-target probability, averaged in summary tables.",
    "brier": "Multiclass sum of squared errors over the common candidate alphabet; range 0–2.",
    "correct": "Whether the highest-probability candidate equals the target.",
    "top3_correct": "Whether target is among the three highest-probability candidates.",
    "predicted_target": "ID of highest-probability candidate; spatial zone or player ID.",
    "confidence": "Largest predicted probability; calibration tables report its bin mean.",
    "accuracy": "Mean top-1 correctness, fraction 0–1.",
    "top3_accuracy": "Mean top-3 correctness, fraction 0–1.",
    "n_b_training": "Number of training transitions with this B state, within team for identities.",
    "n_ab_training": "Number of training transitions with this A,B state, within team for identities.",
    "training_group_matches": "Number of training matches (same team for identity models).",
    "target_seen_in_training": "Whether actual target appeared in training (same team for identities).",
    "unseen_B": "Count of evaluated rows with no B-state training transition.",
    "unseen_AB": "Count of evaluated rows with no A,B-state training transition.",
    "unseen_target": "Count of evaluated rows whose target was not seen in training.",
    "no_group_history": "Count of evaluated rows with no team training history.",
    "has_history": "Rolling model has at least one strictly earlier match for its training group.",
    "target": "Actual successful B-pass destination zone or actual receiver player ID.",
    "alpha1": "Nested-selected Dirichlet prior mass for M1 toward M0.",
    "alpha2": "Nested-selected Dirichlet prior mass for M2 toward M1.",
    "C1": "Nested-selected inverse L2 regularization strength for M1-context.",
    "C2": "Nested-selected inverse L2 regularization strength for M2-context.",
    "C": "Inverse L2 regularization strength for this fitted logistic model.",
    "train_rows": "Transitions in the outer training split.",
    "test_rows": "Transitions in the outer held-out split.",
    "training_rows": "Transitions used in this individual fitted model.",
    "features": "Number of encoded numeric/categorical features fitted only on training rows.",
    "iterations": "Solver iterations. Logistic limit is 600; Ridge LSQR limit is 1000.",
    "converged": "Logistic solver reported convergence; Ridge fits use tolerance 1e-6.",
    "inner_fold": "Nested validation fold (0–2); -1 for final outer-training fit.",
    "order": "Memory order 1 or 2 of the context classifier.",
    "stage": "Inner tuning fit or final outer-training fit.",
    "level": "Fit level inner/outer, or grouping player/pair/triple as appropriate.",
    "single_class_fallback": "Training target had one class: use smoothed prior-backed probabilities.",
    "no_team_history_fallback": "No training match of this team; use common candidate prior.",
    "outcome": "Outcome whose q model is evaluated; R3_with_B controls B ID; R3_triple conditions on observed B→C.",
    "observed": "Number of non-censored outcomes in this OOF model evaluation.",
    "coverage": "Observed outcomes / initial passes in this entity or overall sample; fraction 0–1.",
    "mean_outcome": "Observed OOF target mean on known outcomes.",
    "mean_prediction": "OOF q prediction mean on the same known outcomes.",
    "mse": "OOF mean squared prediction error; for binary targets this is binary Brier.",
    "training_mean_baseline_mse": "MSE of each fold's training-outcome mean on the same held-out rows.",
    "metric": "Specific downstream endpoint used in this entity-effect row; see methodology.",
    "initial_passes": "All completed open-play initial passes of this entity; triples condition on observed B→C.",
    "initial_matches": "Distinct matches represented by initial_passes.",
    "role": "Modal source/tactical A role within this entity; can differ from overall player's modal role.",
    "observed_passes": "Initial passes with known target and finite OOF residual for this metric.",
    "observed_matches": "Distinct matches with known target and finite residual.",
    "observed_possessions": "Distinct whole possession keys with known target and finite residual.",
    "raw_per100": "100 × observed target mean; binary outcomes are percentages, R targets are proxy-value units.",
    "adjusted_excess_per100": "100 × mean(observed outcome − OOF q(X)); binary outcomes are percentage-point excess.",
    "support_eligible": "Known-outcome count/matches meet minimum: player100/3, pair30/2, triple20/2.",
    "ranking_eligible": "Support-eligible and at least 80% of bootstrap draws have a nonempty denominator.",
    "bootstrap_valid_fraction": "Share of 2,000 hierarchical match/possession replicates with a known-outcome row for entity.",
    "raw_ci95_low": "2.5th percentile of hierarchical bootstrap raw_per100; missing for unsupported groups.",
    "raw_ci95_high": "97.5th percentile of hierarchical bootstrap raw_per100; missing for unsupported groups.",
    "ci95_low": "2.5th percentile of cluster bootstrap effect; memory uses paired whole matches; continuation matches then possessions.",
    "ci95_high": "97.5th percentile of cluster bootstrap effect; conditional on fitted models; not simultaneous.",
    "match_mean_ci95_low": "2.5th bootstrap percentile for equal-match-weight gain.",
    "match_mean_ci95_high": "97.5th bootstrap percentile for equal-match-weight gain.",
    "rank_ci95_low": "2.5th percentile of bootstrap within-role rank among fixed supported player peers.",
    "rank_ci95_high": "97.5th percentile of bootstrap within-role rank; fractional endpoints possible.",
    "role_rank": "Point-estimate within-role rank among supported entities; lower receiver loss ranks higher. No significance claim.",
    "top3_frequency": "Fraction of valid bootstrap draws ranking this player in top 3 of supported role peers.",
    "role_peer_count": "Number of supported players with same modal role; singleton has no peer comparison.",
    "leave_match_out_min": "Minimum adjusted excess after deleting each of this entity's observed matches, fixed OOF fits.",
    "leave_match_out_max": "Maximum adjusted excess after deleting each observed match, fixed OOF fits.",
    "continuation_gain_per100": "100 × mean(R3 − fold xT at B receipt), shared-chain descriptive proxy.",
    "chain_gain_per100": "100 × mean(R3 − fold xT at A origin), shared-chain descriptive proxy; overlapping credits.",
    "empirical_bayes_rate_per100": "Pooled binary count shrunk toward averaged cross-fitted role Beta prior; descriptive, not causal.",
    "role_prior_alpha": "Mean role Beta alpha from fits excluding each row's match; empirical-Bayes descriptive shrinkage.",
    "role_prior_beta": "Mean role Beta beta from fits excluding each row's match; empirical-Bayes descriptive shrinkage.",
    "open_play_attempts": "All initial open-play attempts by this player, including failures and lacking-receiver attempts.",
    "completed_attempts": "Provider-completed open-play attempts, including those not eligible as a named A→B edge.",
    "attempt_completion_rate": "completed_attempts / all open_play_attempts; fraction, not percentage.",
    "observed_next_controls": "Attempts with a subsequently recorded control before period censoring.",
    "opponent_next_controls": "Attempts whose next recorded control is by opponent; separate from downstream B loss.",
    "selected": "Hyperparameter selected using only nested training/validation matches: Ridge alpha or logistic C.",
    "include_receiver_B": "q includes receiver B identity as a sensitivity adjustment.",
    "conditional_completed_two_pass": "q training and target restricted to actual completed B→C paths.",
    "training_observed": "Known target observations used to fit this q model.",
    "validation_observed": "Known target observations scored in this validation/test fold.",
    "minimum_passes": "Known-outcome threshold for descriptive player support sensitivity (50/100/150).",
    "rank": "Point-estimate role rank under this support threshold; not bootstrap uncertainty.",
    "R3_observed": "Number of known R3 targets in this match/team.",
    "R3_mean": "Raw R3 mean in this match/team, using held-out fold xT.",
    "receiver_loss_observed": "Number of observed receiver-resolution-loss labels in this match/team.",
    "box3_observed": "Number of known new-box-entry labels from eligible outside-box receipts.",
    "shot10_observed": "Number of known shot-within-10-second labels in this match/team.",
    "triple_observed": "Initial pass has an actual completed and memory-eligible next B→C.",
    "xt_at_receipt": "Outer-training-only xT state value at initial B receipt.",
    "xt_at_A_origin": "Outer-training-only xT state value at initial A pass origin.",
    "continuation_gain_3": "R3 minus training-only xT at initial receipt.",
    "chain_gain_3": "R3 minus training-only xT at initial A origin.",
    "R1": "First shot source xG / confirmed loss zero / retained xT endpoint within 1 future action; else missing.",
    "R3": "Same state/terminal-value proxy within 3 future actions, with explicit censoring.",
    "R5": "Same state/terminal-value proxy within 5 future actions, with explicit censoring.",
    "R3_with_B": "Same R3 target; q additionally includes B identity.",
    "R3_triple": "R3 target conditional on actual completed B→C; q fitted on same conditional training population.",
    "survival3": "1 if R3 resolves as retained or first shot, 0 if confirmed loss; not literal retention after shot.",
    "box3": "New box entry in next 3 actions from a receipt outside the box; known labels only.",
    "shot5": "Shot within 5 seconds in bounded possession suffix; unknown horizon stays missing.",
    "shot10": "Shot within 10 seconds in bounded possession suffix; unknown horizon stays missing.",
    "shot20": "Shot within 20 seconds in bounded possession suffix; unknown horizon stays missing.",
    "xg10": "Sum of source shot xG within 10 seconds, only fully observed time windows; overlapping credits.",
    "receiver_loss": "Confirmed loss during B resolution from source sequence audit, unknown stays missing.",
    "horizon": "Number of future ball actions (1/3/5), counted after B receipt.",
    "status": "Future endpoint: retained / first shot / confirmed loss / censored.",
    "reason": "Auditable reason for endpoint classification; no source xG invented for own goals.",
    "shot_xg": "Source event xG of first observed shot, not newly estimated.",
    "terminal_event_id": "Source UUID of first shot or last selected action; follow source next control for loss confirmation.",
    "bin": "Fixed confidence decile 0–9; predictions exactly 1 fall in bin9.",
    "n": "Number of rows in confidence bin.",
    "excluded_team_id": "Team deleted from both focal-team and opponent evaluation rows; fitted models unchanged.",
    "method": "Explicit sensitivity method; fixed-OOF deletions do not refit q or classifiers.",
}


def dictionary(root):
    base = pd.read_csv(root / "results/euro_sequences/data_dictionary.csv")
    description_column = "description" if "description" in base else "definition"
    inherited = base.drop_duplicates("column").set_index("column")[description_column].to_dict()
    records = []
    paths = sorted((root / "results/memory").glob("tables/*.csv"))
    for rep in REPS:
        paths.extend(sorted((root / f"results/memory/{rep}").glob("*.csv*")))
    paths.extend(sorted((root / "results/continuation/tables").glob("*.csv*")))
    for path in paths:
        if path.stat().st_size < 3:
            continue
        frame = pd.read_csv(path, nrows=400)
        for col in frame:
            definition = FIELD_DEFINITIONS.get(col)
            if definition is None and col.startswith("residual_"):
                definition = "Observed "+col[9:]+" minus prediction from training excluding this entire match; missing when target censored."
            if definition is None and col.startswith("pred_"):
                definition = "Cross-fitted q expectation of "+col[5:]+" using receipt-time X, optionally B ID; no A identity."
            if definition is None and col.startswith("null_"):
                definition = "Outer-training observed mean for "+col[5:]+"; baseline without receipt context."
            if definition is None and col.startswith(("prior_a_", "prior_b_")):
                definition = "Training-only empirical-Bayes Beta role prior parameter for "+col[8:]+"."
            if definition is None:
                definition = inherited.get(col)
            if definition is None:
                raise ValueError(f"Undocumented result column {path.name}:{col}")
            records.append({"file": path.relative_to(root).as_posix(), "column": col,
                            "sample_inferred_dtype": str(frame[col].dtype), "description": definition,
                            "missing_policy": "Empty means unknown/censored/not applicable; zero is an observed value.",
                            "source": "step-1 source-derived field" if col in inherited and col not in FIELD_DEFINITIONS else "steps-2/3 derived result"})
    result = pd.DataFrame(records)
    result.to_csv(root / "results/memory/data_dictionary.csv", index=False)
    return result


def feature_coverage(root, cont):
    """Audit the categories available to each final outer-training q encoder."""
    frame = cont["oof_continuation"]
    records = []
    splits = json.loads((root / "results/continuation/xt_models.json").read_text())["split_manifest"]
    for split in splits:
        for outcome in cont["outcome_model_metrics"].outcome:
            train = frame.loc[frame.match_id.isin(split["train_match_ids"]) & frame[outcome].notna()]
            test = frame.loc[frame.match_id.isin(split["test_match_ids"])]
            train_features = context_features(train, include_b=outcome == "R3_with_B")
            test_features = context_features(test, include_b=outcome == "R3_with_B")
            for feature in train_features:
                if feature in ["rx", "ry", "rx2", "ry2", "rxy"]:
                    continue
                a = train_features[feature]
                b = test_features[feature]
                records.append({"fold": split["fold"], "outcome": outcome, "feature": feature,
                                "training_categories": a.nunique(), "test_rows": len(test),
                                "unknown_rows": int((~b.isin(a.unique())).sum()),
                                "unknown_policy": "one-hot all zero for unseen category; other context retained"})
    result = pd.DataFrame(records)
    result.to_csv(root / "results/continuation/tables/q_feature_coverage.csv", index=False)
    cont["q_feature_coverage"] = result


def build_study_report(root, config):
    root = Path(root)
    _, memory = assemble_memory(root, config)
    cont = {p.name.removesuffix(".csv.gz").removesuffix(".csv"): pd.read_csv(p, low_memory=False)
            for p in (root / "results/continuation/tables").glob("*.csv*")}
    feature_coverage(root, cont)
    index = figures(root, memory, cont, config)
    data_dictionary = dictionary(root)
    comp = memory["memory_comparisons"]
    core = comp.loc[comp.comparison.eq("M2_context_vs_M1_context")].copy()
    core["representation"] = core.representation.map(REPS)
    primary = cont["player_effects"].loc[lambda f: f.metric.eq("R3") & f.ranking_eligible]
    table = primary.sort_values("adjusted_excess_per100", ascending=False).head(12)
    columns = ["a", "team", "role", "observed_passes", "observed_matches", "raw_per100",
               "adjusted_excess_per100", "ci95_low", "ci95_high", "role_rank", "role_peer_count"]
    cards = "".join(f'<section><h2>{html.escape(item["title"])}</h2><p>{html.escape(item["caption"])}</p>'
                    f'<a href="figures/{item["file"]}"><img src="figures/{item["file"]}" alt="{html.escape(item["title"])}"></a>'
                    f'<p><a href="figures/{Path(item["file"]).with_suffix(".svg").name}">SVG</a></p></section>' for item in index)
    links = []
    for folder, prefix in [(root / "results/memory/tables", "tables"), (root / "results/continuation/tables", "../continuation/tables")]:
        links.extend(f'<li><a href="{prefix}/{p.name}">{p.name}</a></li>' for p in sorted(folder.glob("*.csv*")))
    output = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Passing memory and downstream continuation</title><style>
body{{margin:0;background:#edf2f6;color:#20364c;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1250px;margin:auto;padding:32px}}
h1{{font-size:36px;line-height:1.2}}h2{{font-size:24px}}section,.intro{{background:white;border-radius:12px;padding:24px;margin:24px 0}}
img{{max-width:100%;height:auto}}.logo{{width:175px}}a{{color:#216b9a}}table{{border-collapse:collapse;font-size:13px;width:100%}}
td,th{{border-bottom:1px solid #dce4eb;padding:7px;text-align:left}}.scroll{{overflow-x:auto}}.note{{border-left:4px solid #ce6853;padding-left:18px}}
</style><main>

<img class="logo" src="../../assets/statsbomb-logo.png" alt="StatsBomb"><h1>Memory-Aware Football Passing Networks</h1>
<p>EURO 2024 · 51 matches · real passing sequences · exploratory held-out evaluation</p>
<section><h2>What does the previous pass add?</h2><p>First- and second-order models use the same 33,358 successful A→B→C paths. Spatial memory improves prediction beyond common receipt context. Player-identity memory is smaller and baseline-sensitive: its count-model interval includes zero, and identity context models have worse absolute log loss than the count baselines.</p>
<div class="scroll">{core.to_html(index=False,border=0,float_format=lambda v:f"{v:.5f}")}</div>
<p>Positive gain is an improvement in bits/pass. X contains information available at initial receipt. Five outer whole-match folds, three inner tuning folds and three fixed split repeats are used. The 2,000 paired whole-match bootstrap draws keep both teams together; repeats are sensitivities, not independent samples.</p>
<h2>Downstream continuation</h2><p>R3 uses the first shot's source xG within three later ball actions, zero after confirmed loss, or train-only xT at a retained endpoint. It is a mixed exploratory proxy, not the pure source-xG measure in the later motif extension. There are 39,528 known outcomes among 41,927 initial passes (94.28%); unresolved cases remain censored.</p>
<p><strong>Continuation excess per 100 = 100 × mean[R3 − q(X)].</strong> q and xT exclude the evaluated match, including during inner tuning. q uses receipt location, pressure, B role, teams, time/period and score; A identity is excluded.</p>
<p>{len(primary)} players meet the primary support rules. The following are high point estimates, not a definitive ranking. Read coverage, role, intervals and rank stability together.</p>
<div class="scroll">{table[columns].to_html(index=False,border=0,float_format=lambda v:f"{v:.3f}")}</div>
<p class="note">Overlapping credits describe associations, not A's causal responsibility. They cannot be added as independent chances. Backward passes are not automatically penalised. No supported positive primary player effect has a pointwise interval wholly above zero. Intervals condition on fitted models, omit full training uncertainty and do not resolve repeated-team dependence. No family of player significance tests was run.</p>
<p><a href="../../docs/MEMORY_EVALUATION.md">Exact methods and findings</a> · <a href="data_dictionary.csv">Column dictionary</a> · <a href="run_manifest.json">Provenance and hashes</a> · <a href="../motifs/report.html">Completed-motif analysis</a> · <a href="../index.html">All results</a></p></section>
{cards}<section><h2>Full exports</h2><ul>{''.join(links)}</ul><p>CSV.gz loads directly in pandas. Unsupported rows remain included. Probability NPZ files contain common candidates, targets and primary-fold probabilities; load with allow_pickle=False. Infomap remains future work. The three earlier case studies are not an untouched test set.</p></section>
</main></html>'''
    out = root / "results/memory"
    (out / "report.html").write_text(output, encoding="utf-8")
    code = ["src/evaluation.py", "src/memory.py", "src/continuation.py", "src/study.py", "src/study_report.py",
            "scripts/run_memory_study.py", "scripts/check_memory_study.py", "configs/memory_study.json",
            "requirements-memory.txt", "requirements-memory-lock.txt"]
    paths = list((root / "results/memory").rglob("*")) + list((root / "results/continuation").rglob("*"))
    outputs = {p.relative_to(root).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
               for p in paths if p.is_file() and p.name not in {"run_manifest.json", "delivery_check.json"}}
    _, hashes = load_study_inputs(root)
    write_json(out / "run_manifest.json", {"source_commit": SOURCE_COMMIT, "config": config,
        "inputs": hashes, "code": {p: sha256(root / p) for p in code},
        "environment": {"python": platform.python_version(), **{p: importlib.metadata.version(p)
                        for p in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "threadpoolctl"]}},
        "figures": len(index), "data_dictionary_rows": len(data_dictionary), "outputs": outputs,
        "completed_roadmap_stages": [1, 2, 3], "pending_roadmap_stages": [4, 5, 6, 7]})
    print(f"Steps 2/3 report: {len(index)} PNG/SVG figures, {len(data_dictionary)} column definitions", flush=True)
