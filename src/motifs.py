"""Actual two/three-pass motifs and post-completion, match-cross-fitted outcomes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from threadpoolctl import threadpool_limits

from .evaluation import load_study_inputs, context_features, assert_split, hierarchical_weights
from .continuation import base_outcomes, values_from_material
from .sequences import valid_point, pitch_zone, window_summary, sha256
from .study import write_json, study_key


def canonical_pattern(players):
    labels = {}
    for player in players:
        if player is None or pd.isna(player):
            raise ValueError("Unknown actual recipient cannot form a successful motif.")
        labels.setdefault(int(player), chr(65+len(labels)))
    return "".join(labels[int(player)] for player in players)


def clean_records(frame):
    """PeriodIndex helpers use Python nulls, not ambiguous pandas.NA booleans."""
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def build_motifs(tables, config):
    if tables["two_pass_sequences"].empty:
        return pd.DataFrame(columns=["motif_id", "order", "pattern"])
    paths = tables["two_pass_sequences"].loc[lambda f: f.memory_target_eligible].copy()
    paths = paths.sort_values(["match_id", "period", "a_event_id"])
    by_first = {r["a_event_id"]: r for r in clean_records(paths)}
    sequences = {r["a_event_id"]: r for r in clean_records(tables["receiver_sequences"])}
    events = {r["event_id"]: r for r in clean_records(tables["events"])}
    previous, period_rows, positions = {}, {}, {}
    for _, group in tables["events"].sort_values(["match_id", "period", "index"]).groupby(["match_id", "period"], sort=False):
        keys = group.event_id.tolist()
        previous.update({current: earlier for current, earlier in zip(keys[1:], keys[:-1])})
        period_rows[(int(group.match_id.iloc[0]),int(group.period.iloc[0]))] = [events[k] for k in keys]
        positions.update({key:i for i,key in enumerate(keys)})
    output = []
    for first in by_first.values():
        chain = [first["a_event_id"], first["b_event_id"]]
        following = by_first.get(first["b_event_id"])
        variants = [(2, chain)]
        if following:
            if (following["match_id"], following["period"], following["team_id"], following["possession_key"]) == (
                    first["match_id"], first["period"], first["team_id"], first["possession_key"]):
                if first["c_id"] == following["b_id"]:
                    variants.append((3, chain+[following["b_event_id"]]))
        for order, pass_ids in variants:
            passes = [events[key] for key in pass_ids]
            first_s, last_s = sequences[pass_ids[0]], sequences[pass_ids[-1]]
            players = [passes[0]["player_id"]]+[r["recipient_id"] for r in passes]
            assert all(passes[i]["recipient_id"] == passes[i+1]["player_id"] for i in range(order-1))
            assert len({r["possession_key"] for r in passes}) == 1
            assert all(r["completed"] and r["open_play"] for r in passes)
            assert all(players[i] != players[i+1] for i in range(order))
            starts = [r["clock_seconds"] for r in passes]
            gaps = np.diff(starts)
            prior_rows = [events.get(previous.get(key)) for key in pass_ids]
            event_gaps = [r["clock_seconds"]-p["clock_seconds"] if p else np.nan for r, p in zip(passes, prior_rows)]
            eps, maximum = config["time_tolerance_seconds"], config["maximum_gap_seconds"]
            between = period_rows[(first["match_id"],first["period"])][positions[pass_ids[0]]+1:positions[pass_ids[-1]]]
            own = [r for r in between if r["team_id"] == first["team_id"]]
            carries = sum(r["event_type"] == "Carry" for r in own)
            dribbles = sum(r["event_type"] == "Dribble" for r in own)
            points = [(r[key_x], r[key_y]) for r in passes for key_x, key_y in [("x", "y"), ("end_x", "end_y")]
                      if valid_point(r[key_x], r[key_y])]
            zones = {pitch_zone(x, y) for x, y in points}
            xs, ys = ([p[0] for p in points], [p[1] for p in points])
            complete_geometry = len(points) == 2*order
            record = {
                "motif_id": f"{first_s['sequence_id']}:m{order}", "order": order,
                "pattern": canonical_pattern(players), "match_id": first["match_id"], "period": first["period"],
                "scope": first["scope"], "team_id": first["team_id"], "team": first["team"],
                "opponent_id": first_s["opponent_id"], "possession_key": first["possession_key"],
                "initial_sequence_id": first_s["sequence_id"], "end_sequence_id": last_s["sequence_id"],
                "pass_event_ids": ";".join(pass_ids), "player_ids": ";".join(str(int(p)) for p in players),
                "player_names": " → ".join([passes[0]["player"]]+[r["recipient"] for r in passes]),
                "first_pass_event_id": pass_ids[0], "last_pass_event_id": pass_ids[-1],
                "first_pass_clock_seconds": starts[0], "last_pass_clock_seconds": starts[-1],
                "completion_clock_seconds": last_s["receipt_clock_seconds"],
                "completion_receipt_event_id": last_s["receipt_event_id"],
                "completion_receipt_observed": last_s["receipt_observed"],
                "completion_receipt_quality": bool(last_s["provider_possession_team_matches"]
                    and not last_s["receipt_after_boundary"] and last_s["receipt_completed"] is not False),
                "gap_between_pass_starts": ";".join(f"{v:.6f}" for v in gaps),
                "max_pass_start_gap": float(max(gaps)),
                "previous_event_ids": ";".join(p["event_id"] if p else "" for p in prior_rows),
                "previous_event_types": ";".join(p["event_type"] if p else "" for p in prior_rows),
                "gaps_to_previous_events": ";".join(f"{v:.6f}" for v in event_gaps),
                "previous_event_5s": bool(all(np.isfinite(v) and -eps <= v <= maximum+eps for v in event_gaps)),
                "between_passes_5s": bool(all(-eps <= v <= maximum+eps for v in gaps)),
                "intervening_carries": carries, "intervening_dribbles": dribbles,
                "no_carry_dribble": carries+dribbles == 0, "all_linked": True,
                "initial_receipt_zone": first_s["receipt_zone_6x3"],
                "initial_receipt_third": ("defensive" if first_s["receipt_x"] < 40 else "middle" if first_s["receipt_x"] < 80 else "attacking"),
                "initial_pressure": first_s["b_pressure_state"],
                "completion_zone": last_s["receipt_zone_6x3"],
                "completion_third": ("defensive" if last_s["receipt_x"] < 40 else "middle" if last_s["receipt_x"] < 80 else "attacking"),
                "completion_pressure": last_s["b_pressure_state"],
                "initial_receipt_x": first_s["receipt_x"], "initial_receipt_y": first_s["receipt_y"],
                "completion_x": last_s["receipt_x"], "completion_y": last_s["receipt_y"],
                "duration_seconds": last_s["receipt_clock_seconds"]-starts[0],
                "net_progression_x": last_s["receipt_x"]-passes[0]["x"] if valid_point(passes[0]["x"], passes[0]["y"]) else np.nan,
                "total_pass_length": float(sum(r["pass_length"] or 0. for r in passes)),
                "unique_zones_6x3": len(zones) if complete_geometry else np.nan,
                "span_x": max(xs)-min(xs) if complete_geometry else np.nan,
                "span_y": max(ys)-min(ys) if complete_geometry else np.nan,
                "geometry_complete": complete_geometry,
            }
            output.append(record)
    result = pd.DataFrame(output)
    if result.empty:
        return pd.DataFrame(columns=["motif_id", "order", "pattern"])
    assert result.motif_id.is_unique
    return result.sort_values(["match_id", "first_pass_clock_seconds", "order", "motif_id"]).reset_index(drop=True)


def shot15_labels(tables):
    """Paper-inspired last-pass-start clock; separate own-goal convention."""
    sequences = tables["receiver_sequences"]
    output = []
    for _, period in tables["events"].groupby(["match_id", "period"], sort=False):
        rows = clean_records(period.sort_values("index"))
        by_id = {r["event_id"]: j for j, r in enumerate(rows)}
        selected = sequences.loc[sequences.a_event_id.isin(by_id)]
        for s in clean_records(selected):
            start = by_id[s["a_event_id"]]
            stop_pos = by_id[s["stop_event_id"]] if s["stop_event_id"] in by_id else len(rows)
            stop = rows[stop_pos] if stop_pos < len(rows) else None
            actions = [r for r in rows[start+1:stop_pos+int(s["stop_action_included"])]
                       if r["team_id"] == s["team_id"] and r["on_ball_action"]]
            window = window_summary(rows[start], s["sequence_id"], actions, stop, s["stop_reason"],
                        s["stop_clock_seconds"], s["a_clock_seconds"], max(r["clock_seconds"] for r in rows), "seconds", 15)
            own_goal = bool(stop and stop["event_type"] in {"Own Goal Against", "Own Goal For"}
                and stop["goal_beneficiary_team_id"] == s["team_id"]
                and stop["clock_seconds"] <= s["a_clock_seconds"]+15+.001)
            known = window["shot_outcome_known"] or own_goal
            quality = (s["provider_possession_team_matches"] and not s["receipt_after_boundary"]
                       and s["receipt_completed"] is not False)
            output.append({"sequence_id": s["sequence_id"],
                "shot15_pass_start": float(window["shot_observed"] or own_goal) if known and quality else np.nan,
                "shot15_source_only": float(window["shot_observed"]) if window["shot_outcome_known"] and quality else np.nan,
                "shot15_own_goal_counted": own_goal,
                "shot15_event_ids": window["shot_event_ids"],
                "shot15_boundary_event_id": s["stop_event_id"]})
    return pd.DataFrame(output)


GEOMETRY = ["duration_seconds", "net_progression_x", "total_pass_length", "unique_zones_6x3",
            "span_x", "span_y", "intervening_carries", "intervening_dribbles"]
NUMERICAL = ["rx", "ry", "rx2", "ry2", "rxy"] + GEOMETRY


def motif_features(frame, source, spec):
    key = "initial_sequence_id" if spec == "initial_receipt" else "end_sequence_id"
    rows = source.set_index("sequence_id").loc[frame[key]].reset_index()
    rows.index = frame.index
    features = context_features(rows)
    if spec == "completion_geometry":
        for c in GEOMETRY:
            features[c] = frame[c].astype(float).fillna(0)
        features["geometry_missing"] = (~frame.geometry_complete).astype(str)
    assert not set(features).intersection({"pattern", "player_ids", "motif_id", "shot10", "R3", "a_id", "b_id", "c_id"})
    return features


def fit_q(train, test, y_train, y_test, source, spec, parameters, binary, max_iter):
    observed = np.isfinite(y_train)
    a = motif_features(train.loc[observed], source, spec)
    b = motif_features(test, source, spec)
    numeric = [c for c in a if c in NUMERICAL]
    categorical = [c for c in a if c not in numeric]
    transform = ColumnTransformer([("numeric", StandardScaler(), numeric),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical)], sparse_threshold=1)
    X, Z = transform.fit_transform(a), transform.transform(b)
    target = np.asarray(y_train)[observed]
    known = np.isfinite(y_test)
    output, scores, audit = {}, {}, []
    for parameter in parameters:
        if binary and len(np.unique(target)) == 1:
            predicted = np.full(len(test), (target.sum()+.5)/(len(target)+1))
            iterations, converged = 0, True
        elif binary:
            model = LogisticRegression(C=parameter, max_iter=max_iter, tol=1e-4)
            model.fit(X, target.astype(int))
            predicted = model.predict_proba(Z)[:, 1]
            iterations = int(model.n_iter_.max()); converged = iterations < max_iter
        else:
            model = Ridge(alpha=parameter, solver="lsqr", tol=1e-6, max_iter=1000)
            model.fit(X, target)
            predicted = np.clip(model.predict(Z), 0, 1)
            iterations = int(np.max(model.n_iter_)); converged = iterations < 1000
        output[parameter] = predicted
        y, p = np.asarray(y_test)[known], np.clip(predicted[known], 1e-12, 1-1e-12)
        loss = (-y*np.log(p)-(1-y)*np.log(1-p)) if binary else (y-p)**2
        scores[parameter] = float(loss.sum())
        audit.append({"parameter": parameter, "training_observed": len(target), "test_observed": int(known.sum()),
                      "encoded_features": X.shape[1], "iterations": iterations, "converged": converged})
    coverage = [{"feature": c, "unknown_rows": int((~b[c].isin(a[c].unique())).sum()),
                 "test_rows": len(test), "training_categories": a[c].nunique()} for c in categorical]
    return output, scores, audit, coverage


def load_dependencies(root):
    tables, hashes = load_study_inputs(root)
    delivery = json.loads((root / "results/memory/delivery_check.json").read_text())
    assert delivery["status"] == "passed"
    run = json.loads((root / "results/memory/run_manifest.json").read_text())
    needed = ["results/continuation/xt_models.json", "results/continuation/tables/outcome_material.csv.gz"]
    for file in needed:
        assert sha256(root / file) == run["outputs"][file]["sha256"], file
        hashes[file] = sha256(root / file)
    for file in ["src/continuation.py", "src/evaluation.py"]:
        assert sha256(root / file) == run["code"][file], file
    material = pd.read_csv(root / needed[1])
    xt_meta = json.loads((root / needed[0]).read_text())
    return tables, material, xt_meta, hashes


def fit_fold(frame, source, material, xt_meta, record, base_config, config, progress=print):
    assert_split(record)
    train = frame.match_id.isin(record["train_match_ids"]).to_numpy()
    test = frame.match_id.isin(record["test_match_ids"]).to_numpy()
    outer_xt = next(m for m in xt_meta["models"] if m["level"] == "outer" and m["fold"] == record["fold"])
    assert outer_xt["training_match_ids"] == record["train_match_ids"]
    actual = {c: frame[c].to_numpy(dtype=float) for c in config["outcomes"] if c != "R3"}
    actual["R3"] = values_from_material(material, outer_xt, frame.end_sequence_id)["R3"]
    result = frame.loc[test].copy()
    result["fold"] = record["fold"]
    for outcome, values in actual.items(): result[outcome] = values[test]
    parameters, audits, coverage = [], [], []
    inner_values = []
    for j, inner in enumerate(record["inner"]):
        model = next(m for m in xt_meta["models"] if m["level"] == "inner" and m["fold"] == record["fold"] and m["inner_fold"] == j)
        assert model["training_match_ids"] == inner["train_match_ids"]
        inner_values.append({**actual, "R3": values_from_material(material, model, frame.end_sequence_id)["R3"]})
    for order in [2, 3]:
        for spec in config["context_specs"]:
            for outcome in config["outcomes"]:
                binary = outcome != "R3"
                grid = base_config["logistic_C_grid"] if binary else base_config["ridge_alpha_grid"]
                totals = {p: 0. for p in grid}
                metadata = {"fold": record["fold"], "order": order, "context": spec, "outcome": outcome}
                for j, inner in enumerate(record["inner"]):
                    a = (frame.match_id.isin(inner["train_match_ids"]) & frame.order.eq(order)).to_numpy()
                    b = (frame.match_id.isin(inner["test_match_ids"]) & frame.order.eq(order)).to_numpy()
                    _, scores, log, _ = fit_q(frame.loc[a], frame.loc[b], inner_values[j][outcome][a],
                        inner_values[j][outcome][b], source, spec, grid, binary, base_config["max_iter"])
                    for p in grid: totals[p] += scores[p]
                    audits.extend({**metadata, **item, "stage": "inner", "inner_fold": j} for item in log)
                chosen = min(totals, key=totals.get)
                a, b = train & frame.order.eq(order).to_numpy(), test & frame.order.eq(order).to_numpy()
                prediction, _, log, cov = fit_q(frame.loc[a], frame.loc[b], actual[outcome][a], actual[outcome][b],
                    source, spec, [chosen], binary, base_config["max_iter"])
                col = f"pred_{spec}_{outcome}"
                result.loc[frame.loc[b].index, col] = prediction[chosen]
                audits.extend({**metadata, **item, "stage": "outer", "inner_fold": -1} for item in log)
                coverage.extend({**metadata, **item} for item in cov)
                parameters.append({**metadata, "selected": chosen, "inner_loss_sum": totals[chosen],
                                   "training_matches": len(record["train_match_ids"])})
                if progress: progress(f"motifs f{record['fold']} {order} passes {spec} {outcome}: {chosen}")
    for spec in config["context_specs"]:
        for outcome in config["outcomes"]:
            result[f"residual_{spec}_{outcome}"] = result[outcome]-result[f"pred_{spec}_{outcome}"]
    return result, parameters, audits, coverage


def summarize(frame, config):
    blocks, weights = hierarchical_weights(frame, config["bootstrap_repetitions"], config["seed"])
    by_block = pd.Series(np.arange(len(blocks)), index=blocks.possession_key)
    b = frame.possession_key.map(by_block).to_numpy(dtype=int)
    summaries, contrasts = [], []
    for order in [2, 3]:
        patterns = config["patterns"][str(order)]
        ref = config["contrasts_reference"][str(order)]
        group = pd.Categorical(frame.pattern, categories=patterns).codes
        for policy in config["policies"]:
            for scope in config["scope_sensitivity"]:
                select = frame.order.eq(order) & frame[policy]
                if scope == "regulation": select &= frame.period.le(2)
                selected = frame.loc[select]
                for outcome in config["outcomes"]:
                    known = select.to_numpy() & frame[outcome].notna().to_numpy()
                    counts = selected.groupby("pattern").size().reindex(patterns, fill_value=0)
                    for spec in config["context_specs"]:
                        values = frame[outcome].to_numpy(dtype=float)
                        residual = frame[f"residual_{spec}_{outcome}"].to_numpy(dtype=float)
                        positions = np.flatnonzero(known)
                        shape = (len(patterns), len(blocks))
                        coords = (group[positions], b[positions])
                        nmat = sparse.coo_matrix((np.ones(len(positions)), coords), shape=shape).tocsr()
                        rmat = sparse.coo_matrix((values[positions], coords), shape=shape).tocsr()
                        emat = sparse.coo_matrix((residual[positions], coords), shape=shape).tocsr()
                        denominator = nmat @ weights.T
                        raw_draws = np.divide(100*(rmat @ weights.T), denominator,
                            out=np.full_like(denominator, np.nan, dtype=float), where=denominator>0)
                        excess_draws = np.divide(100*(emat @ weights.T), denominator,
                            out=np.full_like(denominator, np.nan, dtype=float), where=denominator>0)
                        n = np.asarray(nmat.sum(axis=1)).ravel()
                        raw = np.divide(100*np.asarray(rmat.sum(axis=1)).ravel(), n, out=np.full(len(n), np.nan), where=n>0)
                        excess = np.divide(100*np.asarray(emat.sum(axis=1)).ravel(), n, out=np.full(len(n), np.nan), where=n>0)
                        metadata = {"order": order, "policy": policy, "scope_filter": scope, "context": spec, "outcome": outcome}
                        supported = []
                        for j, pattern in enumerate(patterns):
                            rows = frame.loc[known & frame.pattern.eq(pattern).to_numpy()]
                            observed_matches = rows.match_id.nunique()
                            eligible = n[j] >= config["minimum_observed"] and observed_matches >= config["minimum_matches"]
                            supported.append(eligible)
                            qraw, qadj = np.nanquantile(raw_draws[j], [.025,.975]), np.nanquantile(excess_draws[j], [.025,.975])
                            summaries.append({**metadata, "pattern": pattern, "motifs": int(counts[pattern]),
                                "frequency": counts[pattern]/len(selected) if len(selected) else np.nan,
                                "known": int(n[j]), "matches": observed_matches, "possessions": rows.possession_key.nunique(),
                                "coverage": n[j]/counts[pattern] if counts[pattern] else np.nan,
                                "raw_per100": raw[j], "adjusted_excess_per100": excess[j], "support_eligible": eligible,
                                "raw_ci95_low": qraw[0] if eligible else np.nan, "raw_ci95_high": qraw[1] if eligible else np.nan,
                                "ci95_low": qadj[0] if eligible else np.nan, "ci95_high": qadj[1] if eligible else np.nan,
                                "bootstrap_valid_fraction": float(np.isfinite(excess_draws[j]).mean())})
                        ri = patterns.index(ref)
                        for j, pattern in enumerate(patterns):
                            if pattern == ref: continue
                            adj = excess_draws[j]-excess_draws[ri]
                            rdiff = raw_draws[j]-raw_draws[ri]
                            eligible = supported[j] and supported[ri]
                            qadj, qraw = np.nanquantile(adj, [.025,.975]), np.nanquantile(rdiff, [.025,.975])
                            contrasts.append({**metadata, "pattern": pattern, "reference": ref,
                                "known_pattern": int(n[j]), "known_reference": int(n[ri]),
                                "raw_difference_per100": raw[j]-raw[ri],
                                "adjusted_difference_per100": excess[j]-excess[ri],
                                "ci95_low": qadj[0] if eligible else np.nan, "ci95_high": qadj[1] if eligible else np.nan,
                                "raw_ci95_low": qraw[0] if eligible else np.nan, "raw_ci95_high": qraw[1] if eligible else np.nan,
                                "support_eligible": eligible, "bootstrap_valid_fraction": float(np.isfinite(adj).mean())})
    frequency = frame.groupby(["order", "team_id", "team", "pattern"]).agg(motifs=("motif_id","size"),
        matches=("match_id","nunique")).reset_index()
    frequency["frequency"] = frequency.motifs/frequency.groupby(["order","team_id"]).motifs.transform("sum")
    stratified = []
    for stratum in ["initial_receipt_third", "initial_pressure", "completion_third", "completion_pressure"]:
        for (order, label, pattern), rows in frame.groupby(["order", stratum, "pattern"]):
            for outcome in ["R3", "shot10", "box3"]:
                known = rows.loc[rows[outcome].notna()]
                stratified.append({"order": order, "stratum": stratum, "stratum_value": label, "pattern": pattern,
                    "outcome": outcome, "motifs": len(rows), "known": len(known), "matches": known.match_id.nunique(),
                    "raw_per100": 100*known[outcome].mean(),
                    "initial_adjusted_per100": 100*known[f"residual_initial_receipt_{outcome}"].mean(),
                    "completion_adjusted_per100": 100*known[f"residual_completion_geometry_{outcome}"].mean(),
                    "note": "Descriptive subgroup means; use primary joint bootstrap contrasts for inference."})
    metrics = []
    for order in [2,3]:
        for spec in config["context_specs"]:
            for outcome in config["outcomes"]:
                known = frame.loc[frame.order.eq(order) & frame[outcome].notna()]
                prediction = known[f"pred_{spec}_{outcome}"]
                metrics.append({"order": order, "context": spec, "outcome": outcome, "known": len(known),
                    "mean_outcome": known[outcome].mean(), "mean_prediction": prediction.mean(),
                    "mse": np.mean((known[outcome]-prediction)**2)})
    return {"pattern_effects": pd.DataFrame(summaries), "pattern_contrasts": pd.DataFrame(contrasts),
            "team_frequencies": frequency, "context_strata": pd.DataFrame(stratified), "outcome_model_metrics": pd.DataFrame(metrics)}


def run_motifs(root, config, base_config, progress=print):
    root = Path(root)
    tables, material, xt_meta, hashes = load_dependencies(root)
    hashes[config["base_config"]] = sha256(root/config["base_config"])
    key, provenance = study_key(root, {**config, "base": base_config}, hashes,
        ["src/motifs.py", "src/evaluation.py", "src/continuation.py", "src/sequences.py"])
    out = root/"results/motifs"
    (out/"tables").mkdir(parents=True, exist_ok=True)
    cache = root/".runtime/motif_study"/key
    cache.mkdir(parents=True, exist_ok=True)
    if progress: progress("Constructing actual two/three-pass motifs and post-completion labels.")
    frame = build_motifs(tables, config)
    source = base_outcomes(tables)
    status = material.loc[material.horizon.eq(3)].set_index("sequence_id").status
    source["survival3"] = source.sequence_id.map(status).map({"retained":1., "shot":1., "loss":0.})
    source = source.merge(shot15_labels(tables), on="sequence_id", validate="one_to_one")
    for outcome in [c for c in config["outcomes"] if c != "R3"]:
        frame[outcome] = frame.end_sequence_id.map(source.set_index("sequence_id")[outcome])
    for c in ["shot15_source_only", "shot15_own_goal_counted", "shot15_event_ids", "shot15_boundary_event_id"]:
        frame[c] = frame.end_sequence_id.map(source.set_index("sequence_id")[c])
    predictions, params, audit, coverage = [], [], [], []
    for record in xt_meta["split_manifest"]:
        stem = f"f{record['fold']}"
        if (cache/f"{stem}.json").exists():
            meta = json.loads((cache/f"{stem}.json").read_text())
            pred = pd.read_csv(cache/f"{stem}.csv.gz", low_memory=False)
            p, a, cov = meta["parameters"], meta["audit"], meta["coverage"]
            if progress: progress(f"motifs {stem}: restored hash-keyed checkpoint.")
        else:
            with threadpool_limits(base_config["blas_threads"]):
                pred, p, a, cov = fit_fold(frame, source, material, xt_meta, record, base_config, config, progress)
            pred.to_csv(cache/f"{stem}.csv.gz", index=False, compression={"method":"gzip","mtime":0})
            write_json(cache/f"{stem}.json", {"parameters":p,"audit":a,"coverage":cov})
        predictions.append(pred); params.extend(p); audit.extend(a); coverage.extend(cov)
    oof = pd.concat(predictions, ignore_index=True)
    assert oof.motif_id.is_unique and set(oof.motif_id) == set(frame.motif_id)
    oof.to_csv(out/"tables/oof_motifs.csv.gz", index=False, compression={"method":"gzip","mtime":0})
    with threadpool_limits(base_config["blas_threads"]):
        summaries = summarize(oof, config)
    for name, data in summaries.items(): data.to_csv(out/f"tables/{name}.csv", index=False)
    pd.DataFrame(params).to_csv(out/"tables/selected_parameters.csv",index=False)
    pd.DataFrame(audit).to_csv(out/"tables/fit_audit.csv",index=False)
    pd.DataFrame(coverage).to_csv(out/"tables/q_feature_coverage.csv",index=False)
    write_json(out/"split_manifest.json", {"splits":xt_meta["split_manifest"]})
    write_json(out/"analysis_manifest.json", {**provenance, "run_key":key,
        "motifs_by_order":oof.groupby("order").size().to_dict(),
        "outcome_landmark":"After completion of final pass; paper-inspired 15s outcome clocks from final pass start.",
        "xt_models":"Reused unchanged verified 5 outer + 15 inner train-only step-3 xT fits, with exactly same splits.",
        "bootstrap":"2000 whole-match/whole-possession draws; joint pattern differences, fixed fitted models.",
        "adjustment":"Residual group mean difference: descriptive, not causal standardization.",
        "p_values":"No individual hypothesis tests; pointwise intervals, no familywise/FDR significance claims."})
    write_json(out/"DONE.json", {"status":"complete","run_key":key})
    if progress: progress(f"Motifs complete: {oof.groupby('order').size().to_dict()}; joint cluster contrasts exported.")
