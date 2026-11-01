"""Data-only runner, provenance, validation and coverage report for step 1."""
from __future__ import annotations

from collections import Counter
from bisect import bisect_left, bisect_right
import html
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .analysis import ROOT, MATCH_IDS, SOURCE_COMMIT
from .sequences import (SCHEMA_VERSION, ACTION_HORIZONS, TIME_HORIZONS, EPSILON,
                        build_match_sequences, dataframe, source_inventory, zone_definitions, sha256)
from .sequence_schema import RECEIPT_FEATURES, column_info, data_dictionary

TABLE_UNITS = {
    "matches": "One match, metadata, source checksum and independently reconstructed goal totals.",
    "events": "One original event in periods 1-4; raw indices and provider IDs retained.",
    "players": "One known player/team/match from events or tactical snapshots; not a full unused-bench roster.",
    "pass_attempts": "Every pass attempt, including failures and restarts, with explicit initial-sequence eligibility.",
    "receiver_sequences": "One completed open-play A->B with a named receiver, context as of receipt and separately marked outcomes.",
    "two_pass_sequences": "One observed B next pass attempt, successful or failed; actual C is populated only on a completed pass.",
    "sequence_actions": "One of the first five own on-ball actions after A->B; event IDs join to the complete event table.",
    "future_windows": "One initial pass and horizon: 1/3/5 actions, 5/10/20 seconds or the remaining bounded possession segment.",
    "possession_runs": "One contiguous run of provider possession ID/team in a period, with own actions and intervening controls audited.",
    "coverage_by_match_team": "One match/team data coverage and missingness summary; not an effectiveness ranking.",
    "zone_definitions": "One zero-based row-major zone in the 6x3 primary grid or 4x3 sensitivity grid.",
    "role_definitions": "One provider position ID/name and its declared coarse role group.",
    "legacy_comparison": "One of the 2,401 original case-study passes, compared with the previously exported three-match follow-ups.",
}

RECEIPT_FEATURES = [
    "team_id", "opponent_id", "a_id", "b_id", "period", "scope",
    "a_clock_seconds", "a_duration_seconds", "a_x", "a_y", "a_end_x", "a_end_y",
    "a_position_id", "a_role", "a_under_pressure", "a_pass_length", "a_pass_height",
    "a_start_zone_6x3", "a_start_zone_4x3", "a_end_zone_6x3", "a_end_zone_4x3",
    "receipt_clock_seconds", "receipt_elapsed_seconds", "receipt_x", "receipt_y",
    "receipt_zone_6x3", "receipt_zone_4x3", "b_position_id", "b_role",
    "b_under_pressure_at_receipt", "b_pressure_state", "home_score_at_receipt", "away_score_at_receipt",
    "score_difference_at_receipt",
]


def possession_runs(events):
    keys = events[["match_id", "period", "possession_id", "possession_team_id"]]
    run_numbers = keys.ne(keys.shift()).any(axis=1).cumsum()
    names = events.drop_duplicates("team_id").set_index("team_id").team.to_dict()
    result = []
    for _, group in events.groupby(run_numbers, sort=False):
        first, last = group.iloc[0], group.iloc[-1]
        own = group.loc[group.team_id.eq(first.possession_team_id)]
        passes = own.loc[own.event_type.eq("Pass")]
        shots = own.loc[own.event_type.eq("Shot")]
        result.append({"match_id": first.match_id, "period": first.period, "scope": first.scope,
            "possession_id": first.possession_id, "possession_key": first.possession_key,
            "possession_run_key": f"{first.possession_key}:{first['index']}",
            "team_id": first.possession_team_id, "team": names[first.possession_team_id],
            "start_index": first["index"], "end_index": last["index"],
            "start_clock_seconds": first.clock_seconds, "end_clock_seconds": last.clock_seconds,
            "events": len(group), "on_ball_actions": int(own.on_ball_action.sum()),
            "pass_attempts": len(passes), "completed_passes": int(passes.completed.sum()),
            "initial_passes": int(passes.initial_pass_eligible.sum()), "shots": len(shots),
            "source_xg_sum": float(shots.xg.sum()),
            "intervening_opponent_controls": int((group.team_id.ne(first.possession_team_id) & group.controlled_touch).sum())})
    return dataframe(result)


def coverage_table(tables):
    rows = []
    for (mid, tid), attempts in tables["pass_attempts"].groupby(["match_id", "team_id"], sort=False):
        s = tables["receiver_sequences"].loc[lambda f: f.match_id.eq(mid) & f.team_id.eq(tid)]
        b = tables["two_pass_sequences"].loc[lambda f: f.match_id.eq(mid) & f.team_id.eq(tid)]
        rows.append({"match_id": mid, "team_id": tid, "team": attempts.team.iloc[0], "opponent": attempts.opponent.iloc[0],
            "pass_attempts": len(attempts), "unsuccessful_passes": int((~attempts.completed).sum()),
            "open_play_attempts": int(attempts.open_play.sum()),
            "initial_passes": len(s), "regulation_initial_passes": int(s.scope.eq("regulation").sum()),
            "extra_time_initial_passes": int(s.scope.eq("extra_time").sum()),
            "observed_receipts": int(s.receipt_observed.sum()), "receipt_fallbacks": int((~s.receipt_observed).sum()),
            "receiver_role_missing": int(s.b_role.isna().sum()),
            "receiver_pressure_unobserved": int(s.b_under_pressure_at_receipt.isna().sum()),
            "observed_receiver_resolutions": int(s.receiver_resolution_event_id.notna().sum()),
            "other_teammate_handlers": int(s.receiver_status.eq("other_teammate_handler").sum()),
            "next_pass_attempts": len(b), "unsuccessful_next_passes": int((~b.b_completed).sum()),
            "successful_memory_targets": int(b.memory_target_eligible.sum()),
            "spatial_memory_targets": int(b.spatial_target_eligible.sum()),
            "receipt_after_boundary": int(s.receipt_after_boundary.sum())})
    return dataframe(rows)


def legacy_comparison(root, tables):
    old = pd.read_csv(root / "results/extended/tables/followup_passes.csv")
    old_motifs = pd.read_csv(root / "results/extended/tables/two_pass_motifs.csv")
    current = tables["receiver_sequences"].loc[lambda f: f.match_id.isin(MATCH_IDS)]
    assert set(old.pass_event_id) == set(current.a_event_id)
    previous_b = old_motifs.set_index("a_event_id").b_event_id.to_dict()
    new_b = tables["two_pass_sequences"].set_index("a_event_id").b_event_id.to_dict()
    rows = []
    for r in current.itertuples():
        was, now = previous_b.get(r.a_event_id), new_b.get(r.a_event_id)
        rows.append({"match_id": r.match_id, "a_event_id": r.a_event_id, "sequence_id": r.sequence_id,
                     "legacy_b_event_id": was, "current_b_event_id": now,
                     "next_pass_agrees": was == now,
                     "difference": "unchanged" if was == now else "new_observation" if was is None else "removed_or_changed",
                     "current_stop_reason": r.stop_reason, "current_receiver_status": r.receiver_status})
    return pd.DataFrame(rows)


def build_euro_tables(root=ROOT, progress=print):
    root = Path(root)
    matches, manifest = source_inventory(root)
    by_id = {f["match_id"]: f for f in manifest["files"]}
    parts, match_rows = {}, []
    for number, match in enumerate(matches, 1):
        mid = match["match_id"]
        source = by_id[mid]
        raw = json.loads((root / source["path"]).read_bytes())
        built = build_match_sequences(raw, match)
        for name, rows in built.items():
            parts.setdefault(name, []).append(dataframe(rows))
        home, away = match["home_team"], match["away_team"]
        goals = Counter(r["scoring_team_id"] for r in built["events"] if r["scoring_team_id"] is not None)
        match_rows.append({"match_id": mid, "date": match["match_date"], "kick_off": match["kick_off"],
            "stage": match["competition_stage"]["name"], "home_id": home["home_team_id"], "home": home["home_team_name"],
            "away_id": away["away_team_id"], "away": away["away_team_name"],
            "home_score": match["home_score"], "away_score": match["away_score"],
            "reconstructed_home_score": goals[home["home_team_id"]], "reconstructed_away_score": goals[away["away_team_id"]],
            "case_study": mid in MATCH_IDS, "source_path": source["path"], "source_sha256": source["sha256"],
            "raw_events": len(raw), "events": len(built["events"]),
            "shootout_events_excluded": sum(e["period"] == 5 for e in raw),
            "pass_attempts": len(built["pass_attempts"]), "initial_passes": len(built["receiver_sequences"]),
            "next_pass_attempts": len(built["two_pass_sequences"]),
            "regulation_events": sum(r["period"] <= 2 for r in built["events"]),
            "extra_time_events": sum(r["period"] > 2 for r in built["events"])})
        if progress and (number % 5 == 0 or number == len(matches)):
            progress(f"Built {number}/51 matches; latest {mid}")
    tables = {name: dataframe(pd.concat(frames, ignore_index=True)) for name, frames in parts.items()}
    tables["matches"] = dataframe(match_rows)
    tables["possession_runs"] = possession_runs(tables["events"])
    tables["coverage_by_match_team"] = coverage_table(tables)
    tables["zone_definitions"] = zone_definitions()
    tables["role_definitions"] = (tables["events"][["position_id", "position", "role"]].dropna(subset=["position_id"])
                                  .drop_duplicates().sort_values(["position_id", "position"]).reset_index(drop=True))
    tables["legacy_comparison"] = legacy_comparison(root, tables)
    return tables, manifest


def validate_euro_tables(tables, manifest):
    e, a, s, b = (tables[n] for n in ["events", "pass_attempts", "receiver_sequences", "two_pass_sequences"])
    w, links, matches = tables["future_windows"], tables["sequence_actions"], tables["matches"]
    assert len(matches) == 51 and set(matches.match_id) == set(manifest["match_ids"])
    assert e.event_id.is_unique and s.sequence_id.is_unique and a.event_id.is_unique
    assert e.period.isin([1, 2, 3, 4]).all()
    assert len(e) == int(matches.events.sum())
    assert int(matches.raw_events.sum()) == len(e) + int(matches.shootout_events_excluded.sum())
    assert set(a.event_id) == set(e.loc[e.event_type.eq("Pass"), "event_id"])
    assert set(s.a_event_id) == set(a.loc[a.initial_pass_eligible, "event_id"])
    assert a.loc[~a.completed].shape[0] > 0 and b.loc[~b.b_completed].shape[0] > 0
    assert set(s.loc[s.b_next_pass_observed, "sequence_id"]) == set(b.sequence_id)
    assert b.sequence_id.is_unique
    assert b.memory_target_eligible.eq(b.memory_target_exclusion_reason.eq("")).all()
    lookup = e.set_index("event_id")
    context = s.set_index("sequence_id")
    assert b.b_event_id.map(lookup.player_id).eq(b.b_id).all()
    assert b.a_event_id.map(lookup.recipient_id).eq(b.b_id).all()
    for key in ["match_id", "period", "team_id", "possession_id"]:
        assert b.a_event_id.map(lookup[key]).eq(b.b_event_id.map(lookup[key])).all(), key
    assert b.b_event_id.map(lookup["index"]).gt(b.a_event_id.map(lookup["index"])).all()
    # Independent check on actual intervening handler events, rather than only
    # trusting the sequence builder's chosen endpoint.
    handlers = e.loc[e.controlled_touch | e.on_ball_action | e.event_type.isin(["Foul Won", "Shield"])]
    handler_groups = {}
    for key, group in handlers.groupby(["match_id", "period"], sort=False):
        compact = list(group[["index", "team_id", "player_id", "event_type", "completed", "stoppage_reason", "controlled_touch"]].itertuples(index=False, name=None))
        handler_groups[key] = ([r[0] for r in compact], compact)
    for path in b.itertuples():
        indices, compact = handler_groups[(path.match_id, path.period)]
        lo = bisect_right(indices, lookup.at[path.a_event_id, "index"])
        hi = bisect_left(indices, lookup.at[path.b_event_id, "index"])
        for _, tid, pid, kind, completed, stopping, control in compact[lo:hi]:
            if tid != path.team_id:
                # An opposing failed touch (e.g. Miscontrol annotated at B's
                # successful receipt) is not itself confirmed opposing control.
                assert not control, (path.sequence_id, "skipped opposing control", kind)
                continue
            assert pid == path.b_id, (path.sequence_id, "skipped teammate", pid)
            assert (kind not in {"Pass", "Shot", "Dispossessed", "Miscontrol", "Clearance"}
                    and (kind != "Dribble" or completed)
                    and (kind != "Foul Won" or pd.isna(stopping) or not stopping)), (path.sequence_id, kind)
    assert b.loc[~b.b_completed, "c_id"].isna().all()
    successful = b.loc[b.memory_target_eligible]
    assert successful.b_completed.all() and successful.b_open_play.all() and successful.c_id.notna().all()
    assert successful.c_id.eq(successful.b_event_id.map(lookup.recipient_id)).all()
    assert successful.seconds_from_receipt_to_release.ge(-EPSILON).all()
    assert not context.loc[successful.sequence_id, "receipt_after_boundary"].any()
    for actor in ["a", "b"]:
        source = s[f"{actor}_role_source_event_id"]
        limit = s.a_index if actor == "a" else s.receipt_event_id.map(lookup["index"]).fillna(s.a_index)
        known = source.notna()
        assert source.loc[known].map(lookup["index"]).le(limit.loc[known]).all(), f"future role leakage: {actor}"
    observed = s.loc[s.receipt_observed]
    assert observed.receipt_event_id.map(lookup.player_id).eq(observed.b_id).all()
    assert observed.receipt_event_id.map(lookup.event_type).eq("Ball Receipt*").all()
    assert observed.receipt_event_id.map(lookup["index"]).gt(observed.a_index).all()
    assert s.loc[~s.receipt_observed, "b_under_pressure_at_receipt"].isna().all()
    assert s.loc[~s.receipt_observed, "b_pressure_state"].eq("unobserved_receipt").all()
    for prefix in ["start", "end"]:
        for grid, upper in [("6x3", 17), ("4x3", 11)]:
            assert e[f"{prefix}_zone_{grid}"].dropna().between(0, upper).all()
    assert matches.home_score.eq(matches.reconstructed_home_score).all()
    assert matches.away_score.eq(matches.reconstructed_away_score).all()
    assert not w.duplicated(["sequence_id", "horizon_kind", "horizon_value"]).any()
    assert w.groupby("sequence_id").size().eq(7).all()
    assert w.loc[w.horizon_kind.eq("actions"), "observed_actions"].le(w.loc[w.horizon_kind.eq("actions"), "horizon_value"]).all()
    assert w.censored.eq(~w.observation_complete).all()
    assert w.loc[w.censored & ~w.shot_observed, "shot_outcome_known"].eq(False).all()
    assert links.event_id.isin(lookup.index).all()
    assert not links.duplicated(["sequence_id", "step"]).any()
    assert links.step.between(1, 5).all()
    for key in ["match_id", "period", "possession_key"]:
        assert links.event_id.map(lookup[key]).eq(links.sequence_id.map(context[key])).all(), key
    assert links.event_id.map(lookup.team_id).eq(links.sequence_id.map(context.team_id)).all()
    assert links.event_id.map(lookup["index"]).gt(links.sequence_id.map(context.a_index)).all()
    # The terminal own action can be included; no subsequent event may leak in.
    stop_index = links.sequence_id.map(context.stop_index)
    own_terminal = links.sequence_id.map(context.stop_action_included)
    event_index = links.event_id.map(lookup["index"])
    assert ((event_index < stop_index) | (event_index.eq(stop_index) & own_terminal) | stop_index.isna()).all()
    assert links.event_id.map(lookup.on_ball_action).all()
    assert len(s.loc[s.match_id.isin(MATCH_IDS)]) == 2401
    assert len(s.loc[s.match_id.isin(MATCH_IDS) & s.scope.eq("regulation")]) == 2160
    assert not any("xt" in col.lower().split("_") for table in tables.values() for col in table.columns)
    assert tables["coverage_by_match_team"].shape[0] == 102
    assert int(tables["coverage_by_match_team"].initial_passes.sum()) == len(s)
    return {"status": "passed", "schema_version": SCHEMA_VERSION, "matches": len(matches),
        "teams": int(e.team_id.nunique()), "source_files_verified": len(manifest["files"]),
        "raw_events": int(matches.raw_events.sum()), "events_periods_1_to_4": len(e),
        "shootout_events_excluded": int(matches.shootout_events_excluded.sum()),
        "pass_attempts": len(a), "unsuccessful_pass_attempts": int((~a.completed).sum()),
        "initial_completed_open_play_passes": len(s),
        "regulation_initial_passes": int(s.scope.eq("regulation").sum()),
        "extra_time_initial_passes": int(s.scope.eq("extra_time").sum()),
        "observed_next_pass_attempts": len(b), "unsuccessful_next_passes": int((~b.b_completed).sum()),
        "successful_memory_targets": len(successful), "spatial_memory_targets": int(b.spatial_target_eligible.sum()),
        "observed_receipts": int(s.receipt_observed.sum()), "receipt_fallbacks": int((~s.receipt_observed).sum()),
        "missing_receiver_roles": int(s.b_role.isna().sum()), "receipt_after_boundary": int(s.receipt_after_boundary.sum()),
        "future_windows": len(w), "sequence_action_links": len(links), "possession_runs": len(tables["possession_runs"]),
        "legacy_2401_initial_passes_preserved": True,
        "legacy_next_pass_differences": int((~tables["legacy_comparison"].next_pass_agrees).sum()),
        "goals_reconstructed_correctly_in_all_51_matches": True,
        "models_fitted": 0, "evaluation_experiments_run": 0,
        "checks": ["pinned input SHA256", "51 unique matches / 102 match-teams", "all attempts retained",
            "exact initial-pass partition", "actual receiver-to-next-passer identity", "period/team/possession continuity",
            "no invented C after unsuccessful passes", "role provenance precedes prediction time",
            "no own handler or confirmed opposing control skipped when linking A-B-C",
            "missing receipt pressure stays unobserved", "both zone grids bounded", "goals / own goals counted once",
            "future window coverage and censoring", "future actions stay before or at explicit terminal event",
            "original 2401 case-study passes conserved", "no tournament-wide in-sample xT labels"]}


def draw_coverage_figures(tables, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .analysis import configure_style
    configure_style()
    directory = out / "figures"
    directory.mkdir(exist_ok=True)
    figures = []

    def save(fig, stem, title):
        fig.text(.01, .005, "Data: StatsBomb Open Data | EURO 2024 | Dataset coverage, no fitted model", fontsize=8)
        logo_ax = fig.add_axes([.86, .002, .12, .045])
        logo_ax.imshow(plt.imread(ROOT / "assets/statsbomb-logo.png"))
        logo_ax.axis("off")
        for suffix in ["png", "svg"]:
            fig.savefig(directory / f"{stem}.{suffix}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        figures.append({"file": f"{stem}.png", "title": title})

    m = tables["matches"]
    fig, ax = plt.subplots(figsize=(18, 6))
    x = np.arange(len(m))
    ax.bar(x, m.pass_attempts, color="#b8cbd8", label="All pass attempts")
    ax.bar(x, m.initial_passes, color="#27628d", label="Eligible completed open-play A-B")
    ax.set_xticks(x, [f"{i+1:02d}" for i in x], fontsize=8)
    ax.set(xlabel="Match number (chronological; see matches.csv)", ylabel="Pass events", title="Coverage across all 51 EURO 2024 matches")
    ax.legend(frameon=False)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save(fig, "01_match_coverage", "Match and initial-pass coverage")

    s, b = tables["receiver_sequences"], tables["two_pass_sequences"]
    values = [len(s), int(s.receipt_observed.sum()), int(s.b_role.notna().sum()), len(b), int(b.memory_target_eligible.sum())]
    labels = ["Initial completed A-B", "Linked receipt event", "Receiver role known at receipt", "Observed next B pass attempt", "Successful actual A-B-C target"]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(labels[::-1], values[::-1], color="#27628d")
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + len(s)*.008, bar.get_y()+bar.get_height()/2, f"{value:,} ({100*value/len(s):.1f}%)", va="center", fontsize=10)
    ax.set_xlim(0, len(s)*1.24)
    ax.set(xlabel="Count; percentages relative to all initial A-B passes", title="Observation coverage and next-pass sample")
    fig.tight_layout(rect=(0, .05, 1, 1))
    save(fig, "02_receipt_coverage", "Receipt, role and next-pass coverage")

    count = s.stop_reason.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(count.index.str.replace("_", " "), count.values, color="#668d83")
    for bar, value in zip(bars, count.values):
        ax.text(value + count.max()*.01, bar.get_y()+bar.get_height()/2, f"{value:,}", va="center")
    ax.set_xlim(0, count.max()*1.16)
    ax.set(xlabel="Initial A-B sequences", title="Why the bounded future suffix stops")
    fig.tight_layout(rect=(0, .04, 1, 1))
    save(fig, "03_sequence_boundaries", "Sequence continuation boundaries")
    return figures


def write_sequence_report(tables, validation, files, figures, out):
    escape = html.escape
    table_rows = "".join(
        f'<tr><td><a href="tables/{files[name]}">{escape(name)}</a></td><td>{len(frame):,}</td>'
        f'<td>{len(frame.columns)}</td><td>{escape(TABLE_UNITS[name])}</td></tr>'
        for name, frame in tables.items())
    s, b = tables["receiver_sequences"], tables["two_pass_sequences"]
    candidate = []
    for mask in [b.memory_target_eligible & b.pattern.eq("ABA"), b.memory_target_eligible & b.pattern.eq("ABC"),
                 ~b.b_completed, b.b_completed & b.b_direction.eq("backward")]:
        candidate.extend(b.loc[mask, "sequence_id"].head(2).tolist())
    preview = s.set_index("sequence_id").loc[list(dict.fromkeys(candidate))].reset_index()
    target = b.set_index("sequence_id")
    preview = preview[["sequence_id", "match_id", "team", "a", "b", "receipt_clock_seconds", "b_pressure_state", "stop_reason"]]
    preview["c"] = preview.sequence_id.map(target.c)
    preview["b_completed"] = preview.sequence_id.map(target.b_completed)
    preview["b_direction"] = preview.sequence_id.map(target.b_direction)
    preview.to_csv(out / "sequence_examples.csv", index=False)
    sample_html = preview.drop(columns="sequence_id").to_html(index=False, escape=True, na_rep="—", border=0)
    image_html = "".join(
        f'<section><h2>{escape(f["title"])}</h2><a href="figures/{f["file"]}">'
        f'<img src="figures/{f["file"]}" alt="{escape(f["title"])}"></a></section>' for f in figures)
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EURO 2024 — Real passing sequences</title><style>
body{{background:#f1f5f8;color:#20364c;font:16px/1.6 system-ui,sans-serif;margin:0}}main{{max-width:1450px;margin:auto;padding:28px}}
header,section{{background:white;padding:26px;border-radius:14px;margin:22px 0}}h1{{font-size:34px;line-height:1.2}}h2{{font-size:23px}}a{{color:#27628d}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;border-bottom:1px solid #dde5eb;padding:9px}}.scroll{{overflow:auto}}img{{max-width:100%;height:auto}}.brand{{width:180px}}
</style><main>

<header><img class="brand" src="../../assets/statsbomb-logo.png" alt="StatsBomb"><h1>Real passing sequences across all 51 EURO 2024 matches</h1>
<p>Data construction only: real paths, failures, pressure, zones, roles, score and later outcomes. No models are fitted by this stage.</p>
<p><strong>{validation['events_periods_1_to_4']:,} events</strong> · <strong>{validation['pass_attempts']:,} pass attempts</strong> · <strong>{len(s):,} initial A→B passes</strong> · <strong>{len(b):,} observed next B attempts</strong> · <strong>{validation['successful_memory_targets']:,} eligible successful A→B→C paths</strong>.</p>
<p>Regulation and extra time are labelled separately; {validation['shootout_events_excluded']} shootout events are excluded. Reconstructed final scores agree with metadata in all 51 matches; paired own-goal annotations count once.</p>
<p>Event indices define order. Continuation stops at opponent control, possession/period boundaries or explicit interruptions. There are {validation['receipt_fallbacks']} missing linked receipt events: time falls back to pass duration, with pressure unobserved. {validation['receipt_after_boundary']} fallback receipts exceed a boundary and remain flagged/censored and excluded from memory targets.</p>
<p>C is an actual recipient only after successful B→C. Intended recipients of failed passes remain separate. Receipt roles use already observed positions/tactics, not future release information. Missing roles/fallbacks are documented.</p>
<p><a href="../../docs/EURO_SEQUENCES.md">Definitions and reproduction</a> · <a href="data_dictionary.csv">Columns and provenance</a> · <a href="schema.json">Typed schema</a> · <a href="validation.json">Checks</a> · <a href="source_manifest.json">Source SHA256</a> · <a href="../index.html">All results</a></p></header>
<section><h2>Tables</h2><p>Read CSV.gz directly with pandas.read_csv. IDs are keys. Future fields and final match scores are excluded from receipt-context features.</p><div class="scroll"><table><tr><th>Table</th><th>Rows</th><th>Columns</th><th>Observation unit</th></tr>{table_rows}</table></div></section>
{image_html}<section><h2>Auditable examples</h2><p>First chronological examples of returns, third-player continuations, failures and backward passes. Missing C after failure is intentional; these are not player rankings.</p><div class="scroll">{sample_html}</div><p><a href="sequence_examples.csv">CSV with sequence IDs</a></p></section>
<section><h2>Future windows</h2><p>Per initial pass: 1/3/5 actions, 5/10/20 seconds from receipt and the remaining bounded possession suffix. An early boundary or uncertain interruption can censor a horizon; absence of a recorded shot is not automatically a known zero.</p><p>Source xG comes from recorded shots. The same chance can follow several earlier passes; future-xG credits must not be summed as independent opportunities. No xT fitted to these evaluation matches is added at this data-construction stage.</p></section>
<p>Data: StatsBomb Open Data / hudl · pinned commit {SOURCE_COMMIT} · schema {SCHEMA_VERSION}.</p>
</main></html>'''
    (out / "report.html").write_text(text, encoding="utf-8")


def export_euro_sequences(tables, manifest, root=ROOT, *, figures=True):
    root = Path(root)
    out = root / "results/euro_sequences"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    validation = validate_euro_tables(tables, manifest)
    files = {}
    compressed = {"events", "pass_attempts", "receiver_sequences", "two_pass_sequences", "sequence_actions", "future_windows", "possession_runs"}
    for name, frame in tables.items():
        filename = f"{name}.csv.gz" if name in compressed else f"{name}.csv"
        frame.to_csv(out / "tables" / filename, index=False,
                     compression={"method": "gzip", "mtime": 0} if name in compressed else None)
        files[name] = filename
    dictionary = data_dictionary(tables, TABLE_UNITS)
    dictionary.to_csv(out / "data_dictionary.csv", index=False)
    schema = {"schema_version": SCHEMA_VERSION, "row_units": TABLE_UNITS,
              "receipt_feature_candidates": RECEIPT_FEATURES,
              "warning": "Future outcomes, release-time context, final scores and audit fields must not enter receipt-time predictors.",
              "tables": {name: {"file": f"tables/{files[name]}", "rows": len(frame), "columns":
                        {col: {"dtype": str(frame[col].dtype), **column_info(name, col, TABLE_UNITS)} for col in frame}}
                        for name, frame in tables.items()}}
    for filename, value in [("schema.json", schema), ("source_manifest.json", manifest), ("validation.json", validation)]:
        (out / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    figure_index = draw_coverage_figures(tables, out) if figures else []
    (out / "figure_index.json").write_text(json.dumps(figure_index, ensure_ascii=False, indent=2), encoding="utf-8")
    write_sequence_report(tables, validation, files, figure_index, out)
    outputs = [out / "tables" / file for file in files.values()]
    outputs += [out / name for name in ["schema.json", "data_dictionary.csv", "source_manifest.json", "validation.json",
                                       "report.html", "sequence_examples.csv", "figure_index.json"]]
    for figure in figure_index:
        outputs.extend([out / "figures" / figure["file"], out / "figures" / Path(figure["file"]).with_suffix(".svg")])
    code = ["src/sequences.py", "src/sequence_export.py", "src/sequence_schema.py",
            "src/analysis.py", "src/lineups.py", "scripts/build_euro_sequences.py", "requirements-lock.txt"]
    run = {"schema_version": SCHEMA_VERSION, "stage": "1_data_construction_only", "source_commit": SOURCE_COMMIT,
           "models_fitted": 0, "evaluation_experiments_run": 0, "python": platform.python_version(),
           "pandas": pd.__version__, "numpy": np.__version__, "action_horizons": list(ACTION_HORIZONS),
           "time_horizons": list(TIME_HORIZONS), "tolerance_seconds": EPSILON, "zone_grids": ["6x3", "4x3"],
           "code_sha256": {p: sha256(root / p) for p in code},
           "outputs": {p.relative_to(out).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in outputs}}
    (out / "run_manifest.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return validation, out
