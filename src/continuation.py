"""Cross-fitted future outcomes and context-adjusted descriptive passer metrics."""
from __future__ import annotations
from collections import defaultdict
import json
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import betaln
from sklearn.linear_model import Ridge, LogisticRegression
from threadpoolctl import threadpool_limits

from .analysis import ROOT
from .evaluation import (load_study_inputs, split_records, assert_split, context_features,
                         encoder, hierarchical_weights)
from .sequences import valid_point
from .study import study_key, write_json


CONTINUOUS = {"R1", "R3", "R5", "xg10"}
BINARY = {"receiver_loss", "survival3", "box3", "shot5", "shot10", "shot20"}


def xt_statistics(events):
    result = {}
    for mid, frame in events.groupby("match_id", sort=False):
        move = np.zeros(96); shot = np.zeros(96); goal = np.zeros(96); transitions = np.zeros((96, 96))
        for r in frame.itertuples():
            if not valid_point(r.x, r.y):
                continue
            a = min(7, int(r.y/10))*12 + min(11, int(r.x/10))
            if r.event_type == "Shot" and r.shot_type == "Open Play":
                shot[a] += 1
                goal[a] += r.outcome == "Goal"
            elif r.event_type == "Carry" or (r.event_type == "Pass" and r.open_play):
                move[a] += 1
                if r.completed and valid_point(r.end_x, r.end_y):
                    b = min(7, int(r.end_y/10))*12 + min(11, int(r.end_x/10))
                    transitions[a, b] += 1
        result[int(mid)] = (move, shot, goal, transitions)
    return result


def fit_fold_xt(statistics, train_ids, config):
    train_ids = sorted(map(int, train_ids))
    move, shot, goal, transition = [np.sum([statistics[mid][i] for mid in train_ids], axis=0) for i in range(4)]
    alpha, beta = config["xt_transition_prior"], config["xt_shot_prior"]
    p_shot = (shot + alpha*shot.sum()/(move+shot).sum()) / (move+shot+alpha)
    p_goal = (goal + beta*goal.sum()/shot.sum()) / (shot+beta)
    destination = transition.sum(axis=0)/transition.sum()
    success_rate = transition.sum()/move.sum()
    T = (transition + alpha*success_rate*destination[None, :]) / (move[:, None]+alpha)
    Q = (1-p_shot)[:, None]*T
    reward = p_shot*p_goal
    values = np.linalg.solve(np.eye(96)-Q, reward)
    residual = float(np.abs(values - reward - Q@values).max())
    assert residual < 1e-10 and np.all(values >= 0) and np.all(values <= 1)
    return {"training_match_ids": train_ids, "values": values, "transition": T,
            "bellman_residual": residual, "training_moves": int(move.sum()),
            "training_shots": int(shot.sum()), "training_goals": int(goal.sum())}


def xt_value(model, x, y):
    if not valid_point(x, y):
        return np.nan
    return float(model["values"][min(7, int(y/10))*12 + min(11, int(x/10))])


def outcome_material(tables, horizons=(1, 3, 5)):
    """Select the actual endpoint/terminal event before fitting any state-value model."""
    events = tables["events"]
    lookup = events.set_index("event_id").to_dict("index")
    next_control = {}
    for _, group in events.groupby(["match_id", "period"], sort=False):
        following = None
        for row in reversed(list(group.itertuples())):
            next_control[row.event_id] = following
            if row.controlled_touch:
                following = row.event_id
    links = (tables["sequence_actions"].groupby("sequence_id", sort=False).event_id.agg(list).to_dict()
             if not tables["sequence_actions"].empty else {})
    records = []
    for s in tables["receiver_sequences"].itertuples():
        observed = [lookup[key] | {"event_id": key} for key in links.get(s.sequence_id, [])]
        invalid = (not s.provider_possession_team_matches or s.receipt_after_boundary
                   or (pd.notna(s.receipt_completed) and not s.receipt_completed))
        for k in horizons:
            actions = observed[:k]
            shot = next((a for a in actions if a["event_type"] == "Shot"), None)
            status, reason, x, y, shot_xg, terminal = "censored", "short_or_ambiguous_boundary", np.nan, np.nan, np.nan, None
            if invalid:
                reason = "initial_context_quality"
            elif shot:
                status, reason, shot_xg, terminal = "shot", "first_shot", float(shot["xg"]), shot["event_id"]
            elif (pd.notna(s.stop_event_id) and lookup[s.stop_event_id]["event_type"] in {"Own Goal Against", "Own Goal For"}
                  and (len(actions) < k or (actions and not actions[-1]["completed"]))):
                reason = "own_goal_without_source_shot_xg"
            elif actions:
                last = actions[-1]
                terminal = last["event_id"]
                failed = (last["event_type"] in {"Miscontrol", "Dispossessed", "Clearance"}
                          or last["event_type"] in {"Pass", "Dribble"} and not last["completed"])
                following_id = next_control.get(last["event_id"])
                following = lookup.get(following_id)
                if failed and following and following["team_id"] != s.team_id:
                    status, reason = "loss", "failed_action_next_opponent_control"
                elif len(actions) >= k:
                    if last["event_type"] in {"Pass", "Carry"} and last["completed"] and valid_point(last["end_x"], last["end_y"]):
                        status, reason, x, y = "retained", "completed_move_endpoint", last["end_x"], last["end_y"]
                    elif (failed and following and following["team_id"] == s.team_id
                          and (pd.isna(following["stoppage_reason"]) or following["stoppage_reason"] != "restart")
                          and (pd.isna(s.stop_index) or following["index"] < s.stop_index)
                          and valid_point(following["x"], following["y"])):
                        status, reason, x, y = "retained", "same_team_next_control_location", following["x"], following["y"]
                    else:
                        reason = "endpoint_or_control_unobserved"
                elif s.stop_reason == "opponent_control":
                    status, reason = "loss", "opponent_control_before_horizon"
            elif not invalid and s.stop_reason == "opponent_control":
                status, reason = "loss", "opponent_control_before_any_action"
            records.append({"sequence_id": s.sequence_id, "horizon": k, "status": status, "reason": reason,
                            "end_x": x, "end_y": y, "shot_xg": shot_xg, "terminal_event_id": terminal})
    return pd.DataFrame(records)


def values_from_material(material, model, sequence_ids):
    values = {}
    for k, frame in material.groupby("horizon"):
        target = np.full(len(frame), np.nan)
        shot = frame.status.eq("shot").to_numpy()
        loss = frame.status.eq("loss").to_numpy()
        retained = frame.status.eq("retained").to_numpy()
        target[shot] = frame.loc[shot, "shot_xg"]
        target[loss] = 0.
        target[retained] = [xt_value(model, x, y) for x, y in zip(frame.loc[retained, "end_x"], frame.loc[retained, "end_y"])]
        values[f"R{k}"] = pd.Series(target, index=frame.sequence_id).reindex(sequence_ids).to_numpy()
    return values


def base_outcomes(tables):
    frame = tables["receiver_sequences"].copy().reset_index(drop=True)
    paths = tables["two_pass_sequences"].loc[lambda f: f.memory_target_eligible]
    frame["c_id"] = frame.sequence_id.map(paths.set_index("sequence_id").c_id)
    frame["c"] = frame.sequence_id.map(paths.set_index("sequence_id").c)
    frame["triple_observed"] = frame.c_id.notna()
    frame["receiver_loss"] = frame.receiver_loss_confirmed.to_numpy(dtype=float, na_value=np.nan)
    quality = frame.provider_possession_team_matches & ~frame.receipt_after_boundary & (frame.receipt_completed.fillna(True))
    frame.loc[~quality, "receiver_loss"] = np.nan
    w = tables["future_windows"]
    for t in [5, 10, 20]:
        selected = w.loc[w.horizon_kind.eq("seconds") & w.horizon_value.eq(t)].set_index("sequence_id")
        known = frame.sequence_id.map(selected.shot_outcome_known).fillna(False) & quality
        frame[f"shot{t}"] = np.where(known, frame.sequence_id.map(selected.shot_observed).astype(float), np.nan)
        if t == 10:
            complete = frame.sequence_id.map(selected.observation_complete).fillna(False) & quality
            frame["xg10"] = np.where(complete, frame.sequence_id.map(selected.source_xg_sum), np.nan)
    selected = w.loc[w.horizon_kind.eq("actions") & w.horizon_value.eq(3)].set_index("sequence_id")
    outside = ~((frame.receipt_x >= 102) & frame.receipt_y.between(18, 62))
    known = frame.sequence_id.map(selected.box_entry_outcome_known).fillna(False) & quality & outside
    frame["box3"] = np.where(known, frame.sequence_id.map(selected.box_entry_observed).astype(float), np.nan)
    return frame


def q_path(train, test, y_train, y_test, parameters, binary, include_b=False, max_iter=600, upper=1):
    known_train = np.isfinite(y_train)
    known_test = np.isfinite(y_test)
    tr, te = train.loc[known_train], test
    transform = encoder()
    X = transform.fit_transform(context_features(tr, include_b=include_b))
    Z = transform.transform(context_features(te, include_b=include_b))
    target = np.asarray(y_train)[known_train]
    output, scores, audit = {}, {}, []
    for param in parameters:
        if binary and len(np.unique(target)) < 2:
            predicted = np.full(len(test), (target.sum()+.5)/(len(target)+1))
            iterations = 0
        elif binary:
            model = LogisticRegression(C=param, max_iter=max_iter, tol=1e-4)
            model.fit(X, target.astype(int))
            predicted = model.predict_proba(Z)[:, 1]
            iterations = int(model.n_iter_.max())
        else:
            model = Ridge(alpha=param, solver="lsqr", tol=1e-6, max_iter=1000)
            model.fit(X, target)
            predicted = np.clip(model.predict(Z), 0, upper)
            iterations = int(np.max(model.n_iter_)) if model.n_iter_ is not None else 0
        output[param] = predicted
        p, actual = predicted[known_test], np.asarray(y_test)[known_test]
        loss = -actual*np.log(np.maximum(p, 1e-12))-(1-actual)*np.log(np.maximum(1-p, 1e-12)) if binary else (actual-p)**2
        scores[param] = (float(loss.sum()), len(actual))
        audit.append({"parameter": param, "training_observed": len(target), "validation_observed": len(actual),
                      "features": X.shape[1], "iterations": iterations,
                      "converged": not binary or iterations < max_iter})
    return output, scores, audit


def beta_role_priors(frame, values):
    known = np.isfinite(values)
    d = frame.loc[known, ["a_role", "a_id"]].copy()
    d["y"] = np.asarray(values)[known]
    prior = {}
    for role, rows in d.groupby("a_role"):
        people = rows.groupby("a_id").y.agg(["sum", "count"])
        y, n = people["sum"].to_numpy(), people["count"].to_numpy()
        mean = (y.sum()+.5)/(n.sum()+1)
        def objective(log_ab):
            a, b = np.exp(log_ab)
            return float(-np.sum(betaln(y+a, n-y+b)-betaln(a, b)))
        fit = minimize(objective, np.log([max(.01, mean*10), max(.01, (1-mean)*10)]),
                       method="L-BFGS-B", bounds=[(-7, 8), (-7, 8)])
        a, b = np.exp(fit.x)
        prior[role] = (float(a), float(b))
    return prior


def fit_continuation_fold(frame, material, statistics, record, config, progress=print):
    assert_split(record)
    train_mask = frame.match_id.isin(record["train_match_ids"]).to_numpy()
    test_mask = frame.match_id.isin(record["test_match_ids"]).to_numpy()
    outer_xt = fit_fold_xt(statistics, record["train_match_ids"], config)
    assert set(outer_xt["training_match_ids"]).isdisjoint(record["test_match_ids"])
    models = [{"level": "outer", "fold": record["fold"], **outer_xt}]
    outer_y = values_from_material(material, outer_xt, frame.sequence_id)
    status3 = material.loc[material.horizon.eq(3)].set_index("sequence_id").status
    outer_y["survival3"] = frame.sequence_id.map(status3).map({"shot": 1., "retained": 1., "loss": 0.}).to_numpy()
    for name in BINARY | {"xg10"}:
        if name != "survival3":
            outer_y[name] = frame[name].to_numpy(dtype=float)
    inner_parts = []
    for j, inner in enumerate(record["inner"]):
        model = fit_fold_xt(statistics, inner["train_match_ids"], config)
        assert set(model["training_match_ids"]).isdisjoint(inner["test_match_ids"])
        assert set(model["training_match_ids"]).isdisjoint(record["test_match_ids"])
        models.append({"level": "inner", "fold": record["fold"], "inner_fold": j, **model})
        values = {**outer_y, **values_from_material(material, model, frame.sequence_id)}
        inner_parts.append((frame.match_id.isin(inner["train_match_ids"]).to_numpy(),
                            frame.match_id.isin(inner["test_match_ids"]).to_numpy(), values))
    result = frame.loc[test_mask].copy()
    result["fold"] = record["fold"]
    result["xt_at_receipt"] = [xt_value(outer_xt, x, y) for x, y in zip(result.receipt_x, result.receipt_y)]
    result["xt_at_A_origin"] = [xt_value(outer_xt, x, y) for x, y in zip(result.a_x, result.a_y)]
    tasks = [(name, name, False, False) for name in ["R1", "R3", "R5", "receiver_loss", "survival3", "box3", "shot5", "shot10", "shot20", "xg10"]]
    tasks += [("R3_with_B", "R3", True, False), ("R3_triple", "R3", False, True)]
    parameters, audit = [], []
    for output_name, name, include_b, triple in tasks:
        binary = name in BINARY
        grid = config["logistic_C_grid"] if binary else config["ridge_alpha_grid"]
        totals = {p: 0. for p in grid}
        for j, (im_train, im_test, values) in enumerate(inner_parts):
            a, b = im_train.copy(), im_test.copy()
            if triple:
                a &= frame.triple_observed.to_numpy()
                b &= frame.triple_observed.to_numpy()
            predictions, scores, log = q_path(frame.loc[a], frame.loc[b], values[name][a], values[name][b],
                grid, binary, include_b, config["max_iter"], np.inf if name == "xg10" else 1)
            for p, (loss, _) in scores.items(): totals[p] += loss
            audit.extend({**item, "fold": record["fold"], "outcome": output_name, "level": "inner", "inner_fold": j} for item in log)
        chosen = min(totals, key=totals.get)
        a = train_mask.copy()
        if triple: a &= frame.triple_observed.to_numpy()
        predicted, _, log = q_path(frame.loc[a], frame.loc[test_mask], outer_y[name][a], outer_y[name][test_mask],
            [chosen], binary, include_b, config["max_iter"], np.inf if name == "xg10" else 1)
        result[output_name] = outer_y[name][test_mask]
        result[f"pred_{output_name}"] = predicted[chosen]
        result[f"residual_{output_name}"] = result[output_name]-result[f"pred_{output_name}"]
        known_training = np.isfinite(outer_y[name][a])
        result[f"null_{output_name}"] = float(np.mean(outer_y[name][a][known_training]))
        if triple:
            result.loc[~result.triple_observed, [output_name, f"residual_{output_name}"]] = np.nan
        parameters.append({"fold": record["fold"], "outcome": output_name, "selected": chosen,
                           "include_receiver_B": include_b, "conditional_completed_two_pass": triple})
        audit.extend({**item, "fold": record["fold"], "outcome": output_name, "level": "outer", "inner_fold": -1} for item in log)
        if binary:
            priors = beta_role_priors(frame.loc[train_mask], outer_y[name][train_mask])
            result[f"prior_a_{name}"] = result.a_role.map({r: ab[0] for r, ab in priors.items()})
            result[f"prior_b_{name}"] = result.a_role.map({r: ab[1] for r, ab in priors.items()})
        if progress: progress(f"continuation f{record['fold']} {output_name}: selected {chosen}")
    result["continuation_gain_3"] = result.R3-result.xt_at_receipt
    result["chain_gain_3"] = result.R3-result.xt_at_A_origin
    return result, models, parameters, audit


def effect_table(frame, metrics, keys, level, config, blocks, weights):
    """Cluster-aware intervals; only sufficiently supported entities enter rankings."""
    grouped = frame.groupby(keys, sort=True, dropna=False)
    group_index = grouped.size().index
    group_id = grouped.ngroup().to_numpy()
    n_groups = len(group_index)
    block_ids = pd.Series(np.arange(len(blocks)), index=blocks.possession_key)
    block = frame.possession_key.map(block_ids).to_numpy(dtype=int)
    metadata = grouped.agg(initial_passes=("sequence_id", "size"), initial_matches=("match_id", "nunique"),
                           a=("a", "first"), team=("team", "first"),
                           role=("a_role", lambda x: x.mode().iloc[0])).reset_index()
    if "b_id" in keys: metadata["b"] = grouped.b.first().to_numpy()
    if "c_id" in keys: metadata["c"] = grouped.c.first().to_numpy()
    minimum_n = config[f"{level}_min_passes"]
    minimum_matches = config[f"{level}_min_matches"]
    records = []
    for metric in metrics:
        value_name = "R3" if metric == "R3_with_B" else metric
        if metric == "R3_triple": value_name = "R3_triple"
        values = frame[value_name].to_numpy(dtype=float)
        residuals = frame[f"residual_{metric}"].to_numpy(dtype=float)
        known = np.isfinite(values) & np.isfinite(residuals)
        counts = np.bincount(group_id[known], minlength=n_groups)
        sums = np.bincount(group_id[known], weights=values[known], minlength=n_groups)
        excess = np.bincount(group_id[known], weights=residuals[known], minlength=n_groups)
        known_frame = frame.loc[known]
        matches = known_frame.groupby(keys, dropna=False).match_id.nunique().reindex(group_index).fillna(0).to_numpy(dtype=int)
        possessions = known_frame.groupby(keys, dropna=False).possession_key.nunique().reindex(group_index).fillna(0).to_numpy(dtype=int)
        table = metadata.copy()
        table["metric"] = metric
        table["observed_passes"], table["observed_matches"], table["observed_possessions"] = counts, matches, possessions
        table["coverage"] = counts / table.initial_passes
        table["raw_per100"] = np.divide(100*sums, counts, out=np.full(n_groups, np.nan), where=counts>0)
        table["adjusted_excess_per100"] = np.divide(100*excess, counts, out=np.full(n_groups, np.nan), where=counts>0)
        table["support_eligible"] = (counts >= minimum_n) & (matches >= minimum_matches)
        for col in ["raw_ci95_low", "raw_ci95_high", "ci95_low", "ci95_high", "bootstrap_valid_fraction",
                    "rank_ci95_low", "rank_ci95_high", "top3_frequency", "role_peer_count"]:
            table[col] = np.nan
        eligible = np.flatnonzero(table.support_eligible)
        if len(eligible):
            reduced = np.full(n_groups, -1, dtype=int); reduced[eligible] = np.arange(len(eligible))
            selected = known & (reduced[group_id] >= 0)
            shape = (len(eligible), len(blocks))
            coordinates = (reduced[group_id[selected]], block[selected])
            denominator = sparse.coo_matrix((np.ones(selected.sum()), coordinates), shape=shape).tocsr() @ weights.T
            raw_sum = sparse.coo_matrix((values[selected], coordinates), shape=shape).tocsr() @ weights.T
            adjusted_sum = sparse.coo_matrix((residuals[selected], coordinates), shape=shape).tocsr() @ weights.T
            raw_draws = np.divide(100*raw_sum, denominator, out=np.full_like(raw_sum, np.nan), where=denominator>0)
            adjusted_draws = np.divide(100*adjusted_sum, denominator, out=np.full_like(adjusted_sum, np.nan), where=denominator>0)
            table.loc[eligible, ["raw_ci95_low", "raw_ci95_high"]] = np.nanquantile(raw_draws, [.025, .975], axis=1).T
            table.loc[eligible, ["ci95_low", "ci95_high"]] = np.nanquantile(adjusted_draws, [.025, .975], axis=1).T
            table.loc[eligible, "bootstrap_valid_fraction"] = np.isfinite(adjusted_draws).mean(axis=1)
            if level == "player":
                roles = table.loc[eligible, "role"].to_numpy()
                for role in np.unique(roles):
                    positions = np.flatnonzero(roles == role)
                    estimates = adjusted_draws[positions]
                    direction = -1 if metric == "receiver_loss" else 1
                    sorting = np.where(np.isfinite(estimates), -direction*estimates, np.inf)
                    ranks = np.argsort(np.argsort(sorting, axis=0, kind="stable"), axis=0, kind="stable").astype(float)+1
                    ranks[~np.isfinite(estimates)] = np.nan
                    destinations = eligible[positions]
                    table.loc[destinations, ["rank_ci95_low", "rank_ci95_high"]] = np.nanquantile(ranks, [.025, .975], axis=1).T
                    valid = np.isfinite(ranks).sum(axis=1)
                    table.loc[destinations, "top3_frequency"] = ((ranks <= 3).sum(axis=1)/valid)
                    table.loc[destinations, "role_peer_count"] = len(positions)
        table["ranking_eligible"] = table.support_eligible & table.bootstrap_valid_fraction.ge(.8)
        table["role_rank"] = np.nan
        supported = table.loc[table.ranking_eligible]
        table.loc[supported.index, "role_rank"] = supported.groupby("role").adjusted_excess_per100.rank(
            method="min", ascending=metric == "receiver_loss")
        # Fixed-OOF delete-match sensitivity, not a refit of outcome models.
        bounds = []
        for _, group in known_frame.groupby(keys, sort=True, dropna=False):
            by_match = group.groupby("match_id")[f"residual_{metric}"].agg(["sum", "count"])
            leave = 100*(by_match["sum"].sum()-by_match["sum"])/(by_match["count"].sum()-by_match["count"])
            bounds.append((leave.min(), leave.max()))
        if bounds:
            positions = np.flatnonzero(counts > 0)
            table["leave_match_out_min"], table["leave_match_out_max"] = np.nan, np.nan
            table.loc[positions, ["leave_match_out_min", "leave_match_out_max"]] = np.asarray(bounds)
        if value_name in BINARY:
            alpha = frame.loc[known].groupby(keys, dropna=False)[f"prior_a_{value_name}"].mean().reindex(group_index).fillna(.5).to_numpy()
            beta = frame.loc[known].groupby(keys, dropna=False)[f"prior_b_{value_name}"].mean().reindex(group_index).fillna(.5).to_numpy()
            table["empirical_bayes_rate_per100"] = 100*(sums+alpha)/(counts+alpha+beta)
            table["role_prior_alpha"], table["role_prior_beta"] = alpha, beta
        if metric in {"R3", "R3_with_B", "R3_triple"}:
            table["continuation_gain_per100"] = grouped.continuation_gain_3.mean().to_numpy()*100
            table["chain_gain_per100"] = grouped.chain_gain_3.mean().to_numpy()*100
        records.append(table)
    return pd.concat(records, ignore_index=True)


def continuation_summaries(oof, tables, config, progress=print):
    blocks, weights = hierarchical_weights(oof, config["bootstrap_repetitions"], config["seed"])
    if progress: progress("continuation: hierarchical match/possession bootstrap weights ready")
    player_metrics = ["R1", "R3", "R5", "R3_with_B", "receiver_loss", "survival3", "box3", "shot5", "shot10", "shot20", "xg10"]
    players = effect_table(oof, player_metrics, ["team_id", "a_id"], "player", config, blocks, weights)
    if progress: progress("continuation: player intervals and within-role rank stability exported")
    pairs = effect_table(oof, ["R3", "R3_with_B"], ["team_id", "a_id", "b_id"], "pair", config, blocks, weights)
    triple_frame = oof.loc[oof.triple_observed].copy()
    triples = effect_table(triple_frame, ["R3_triple"], ["team_id", "a_id", "b_id", "c_id"], "triple", config, blocks, weights)
    attempts = tables["pass_attempts"].loc[lambda f: f.open_play & f.player_id.notna()].copy()
    attempts["observed_control"] = attempts.next_control_status.ne("censored_period_end")
    attempts["opponent_next_control"] = attempts.next_control_status.eq("opponent_control")
    attempt_summary = attempts.groupby(["team_id", "player_id"]).agg(
        open_play_attempts=("event_id", "size"), completed_attempts=("completed", "sum"),
        observed_next_controls=("observed_control", "sum"),
        opponent_next_controls=("opponent_next_control", "sum")).reset_index().rename(columns={"player_id": "a_id"})
    attempt_summary["attempt_completion_rate"] = attempt_summary.completed_attempts/attempt_summary.open_play_attempts
    players = players.merge(attempt_summary, on=["team_id", "a_id"], how="left", validate="many_to_one")
    model_metrics = []
    for name in player_metrics+["R3_triple"]:
        actual = oof["R3"] if name == "R3_with_B" else oof[name]
        known = actual.notna()
        y, p = actual.loc[known].to_numpy(), oof.loc[known, f"pred_{name}"].to_numpy()
        null = oof.loc[known, f"null_{name}"].to_numpy()
        model_metrics.append({"outcome": name, "observed": int(known.sum()), "coverage": float(known.mean()),
            "mean_outcome": float(y.mean()), "mean_prediction": float(p.mean()),
            "mse": float(np.mean((y-p)**2)), "training_mean_baseline_mse": float(np.mean((y-null)**2))})
    thresholds = []
    main = players.loc[players.metric.eq("R3")]
    for n in config["player_threshold_sensitivity"]:
        keep = main.loc[main.observed_passes.ge(n) & main.observed_matches.ge(config["player_min_matches"])]
        for role, group in keep.groupby("role"):
            for rank, r in enumerate(group.sort_values("adjusted_excess_per100", ascending=False).itertuples(), 1):
                thresholds.append({"minimum_passes": n, "role": role, "a_id": r.a_id, "a": r.a,
                                   "rank": rank, "adjusted_excess_per100": r.adjusted_excess_per100,
                                   "observed_passes": r.observed_passes})
    coverage = oof.groupby(["match_id", "team_id", "team"]).agg(
        initial_passes=("sequence_id", "size"), R3_observed=("R3", "count"), R3_mean=("R3", "mean"),
        receiver_loss_observed=("receiver_loss", "count"), box3_observed=("box3", "count"),
        shot10_observed=("shot10", "count")).reset_index()
    return {"player_effects": players, "pair_effects": pairs, "triple_effects": triples,
            "all_attempts_by_player": attempt_summary, "outcome_model_metrics": pd.DataFrame(model_metrics),
            "player_threshold_sensitivity": pd.DataFrame(thresholds), "outcome_coverage": coverage}


def run_continuation_stage(root=ROOT, config=None, progress=print):
    from pathlib import Path
    root = ROOT if root is None else Path(root)
    tables, hashes = load_study_inputs(root)
    key, provenance = study_key(root, config, hashes, ["src/continuation.py", "src/evaluation.py"])
    cache = root / ".runtime/memory_study" / key / "continuation"
    cache.mkdir(parents=True, exist_ok=True)
    out = root / "results/continuation"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    frame = base_outcomes(tables)
    material = outcome_material(tables, config["continuation_horizons"])
    material.to_csv(out / "tables/outcome_material.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    statistics = xt_statistics(tables["events"])
    records = [r for r in split_records(tables["matches"].match_id, config) if r["repeat"] == 0]
    predictions, xt_models, parameters, audit = [], [], [], []
    for record in records:
        stem = f"f{record['fold']}"
        checkpoint = cache / f"{stem}.json"
        if checkpoint.exists():
            meta = json.loads(checkpoint.read_text())
            result = pd.read_csv(cache / f"{stem}.csv.gz")
            models, params, log = meta["xt_models"], meta["parameters"], meta["fit_audit"]
            if progress: progress(f"continuation {stem}: restored verified-run checkpoint")
        else:
            if progress: progress(f"continuation {stem}: train-only outer/inner xT and outcome models started")
            with threadpool_limits(config["blas_threads"]):
                result, models, params, log = fit_continuation_fold(frame, material, statistics, record, config, progress)
            result.to_csv(cache / f"{stem}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
            write_json(checkpoint, {"xt_models": models, "parameters": params, "fit_audit": log})
        predictions.append(result); xt_models.extend(models); parameters.extend(params); audit.extend(log)
    oof = pd.concat(predictions, ignore_index=True)
    assert oof.sequence_id.is_unique and set(oof.sequence_id) == set(frame.sequence_id)
    oof.to_csv(out / "tables/oof_continuation.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    with threadpool_limits(config["blas_threads"]):
        summaries = continuation_summaries(oof, tables, config, progress)
    for name, table in summaries.items(): table.to_csv(out / f"tables/{name}.csv", index=False)
    pd.DataFrame(parameters).to_csv(out / "tables/selected_parameters.csv", index=False)
    pd.DataFrame(audit).to_csv(out / "tables/fit_audit.csv", index=False)
    write_json(out / "xt_models.json", {"models": xt_models, "split_manifest": records,
        "note": "Every state-value fit is train-only, including inner validation folds. No pre-existing 48-match model reused."})
    write_json(out / "run_manifest.json", {**provenance, "run_key": key,
        "bootstrap": "2000 whole-match then whole-possession replicates; fitted predictions fixed",
        "primary_metric": "100*mean(R3 - cross-fitted q(reception context))",
        "triple_estimand": "Conditional on completed actual B->C; distinct cross-fitted conditional baseline",
        "support": {"player": [100, 3], "pair": [30, 2], "triple": [20, 2]},
        "empirical_bayes": "Role priors estimated only on training matches; pooled descriptive shrinkage, not Bayesian causal inference",
        "p_values": "No individual hypothesis-test family run; effect sizes and cluster intervals reported"})
    write_json(out / "DONE.json", {"status": "complete", "run_key": key})
    if progress: progress("continuation: all outcomes, player/pair/triple intervals and sensitivity tables complete")
