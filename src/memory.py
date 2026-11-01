"""First/second-order prediction on real paths, using match-disjoint nested CV."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
import numpy as np
import pandas as pd

from .analysis import ROOT
from .lineups import dismissal
from .sequences import source_inventory
from .evaluation import (assert_split, context_features, fit_logistic_path,
                         probability_metrics, paired_match_interval, calibration_table)


def nominal_candidates(sequences, root=ROOT):
    """Candidates are known from Starting XI/substitutions/cards before receipt."""
    matches, source = source_inventory(root)
    file_by_id = {f["match_id"]: f for f in source["files"]}
    result = {}
    for match in matches:
        mid = match["match_id"]
        raw = sorted(json.loads((root / file_by_id[mid]["path"]).read_bytes()), key=lambda e: e["index"])
        index_by_id = {e["id"]: e["index"] for e in raw}
        group = sequences.loc[sequences.match_id.eq(mid)].copy()
        group["context_index"] = group.receipt_event_id.map(index_by_id).fillna(group.a_index)
        roster, cursor = {}, 0
        for r in group.sort_values(["context_index", "a_index"]).itertuples():
            while cursor < len(raw) and raw[cursor]["index"] <= r.context_index:
                e = raw[cursor]
                tid = e["team"]["id"]
                active = roster.setdefault(tid, set())
                kind = e["type"]["name"]
                if kind == "Starting XI":
                    active.update(p["player"]["id"] for p in e["tactics"]["lineup"])
                elif kind == "Substitution":
                    active.discard(e["player"]["id"])
                    active.add(e["substitution"]["replacement"]["id"])
                elif dismissal(e):
                    active.discard(e.get("player", {}).get("id"))
                cursor += 1
            choices = sorted(roster[int(r.team_id)] - {int(r.b_id)})
            assert 1 <= len(choices) <= 11
            result[r.sequence_id] = choices
    return result


def prepare_representation(sequences, paths, representation, candidates=None):
    own = paths.loc[paths.memory_target_eligible].copy()
    # Only one copy of shared metadata/context, explicitly renamed future fields.
    frame = sequences.merge(own[["sequence_id", "c_id", "b_end_zone_6x3", "b_end_zone_4x3",
                                  "opponent_noncontrol_annotations"]], on="sequence_id", validate="one_to_one")
    if representation == "players":
        frame["state_a"] = frame.a_id.astype(int)
        frame["state_b"] = frame.b_id.astype(int)
        frame["target"] = frame.c_id.astype(int)
        frame["candidate_target_available"] = [int(c) in candidates[sid] for sid, c in zip(frame.sequence_id, frame.target)]
        excluded = frame.loc[~frame.candidate_target_available, ["sequence_id", "match_id"]].copy()
        frame = frame.loc[frame.candidate_target_available].reset_index(drop=True)
        allowed = np.full((len(frame), 11), -1, dtype=np.int64)
        for i, sid in enumerate(frame.sequence_id):
            choices = candidates[sid]
            allowed[i, :len(choices)] = choices
    else:
        grid = representation.removeprefix("zones_")
        required = [f"a_start_zone_{grid}", f"receipt_zone_{grid}", f"b_end_zone_{grid}"]
        excluded = frame.loc[frame[required].isna().any(axis=1), ["sequence_id", "match_id"]].copy()
        frame = frame.dropna(subset=required).reset_index(drop=True)
        frame["state_a"] = frame[required[0]].astype(int)
        frame["state_b"] = frame[required[1]].astype(int)
        frame["target"] = frame[required[2]].astype(int)
        classes = 18 if grid == "6x3" else 12
        allowed = np.tile(np.arange(classes), (len(frame), 1))
    frame["row_id"] = np.arange(len(frame))
    frame["count_group"] = frame.team_id.astype(int) if representation == "players" else 0
    return frame, allowed, excluded


class MemoryCounts:
    def __init__(self, train):
        self.global_counts = defaultdict(Counter)
        self.first = defaultdict(Counter)
        self.second = defaultdict(Counter)
        self.n_b, self.n_ab, self.games = Counter(), Counter(), defaultdict(set)
        for r in train.itertuples():
            g, a, b, c = int(r.count_group), int(r.state_a), int(r.state_b), int(r.target)
            self.global_counts[g][c] += 1
            self.first[(g, b)][c] += 1
            self.second[(g, a, b)][c] += 1
            self.n_b[(g, b)] += 1
            self.n_ab[(g, a, b)] += 1
            self.games[g].add(int(r.match_id))

    def prepare(self, frame, candidates):
        g0, g1, g2, nb, nab, games, seen = [], [], [], [], [], [], []
        cache = {}
        for r, choices in zip(frame.itertuples(), candidates):
            g, a, b = int(r.count_group), int(r.state_a), int(r.state_b)
            key = (g, a, b, tuple(choices))
            if key not in cache:
                cache[key] = tuple(np.array([counts.get(int(c), 0) if c >= 0 else 0 for c in choices], dtype=float)
                                   for counts in [self.global_counts[g], self.first[(g, b)], self.second[(g, a, b)]])
            p, q, t = cache[key]
            g0.append(p); g1.append(q); g2.append(t)
            nb.append(self.n_b[(g, b)]); nab.append(self.n_ab[(g, a, b)])
            games.append(len(self.games[g])); seen.append(self.global_counts[g].get(int(r.target), 0) > 0)
        return {"counts": [np.vstack(g0), np.vstack(g1), np.vstack(g2)], "mask": candidates >= 0,
                "n_b": nb, "n_ab": nab, "train_group_matches": games, "target_seen": seen}

    @staticmethod
    def probabilities(prepared, alpha1=10, alpha2=10):
        n0, n1, n2 = prepared["counts"]
        mask = prepared["mask"]
        uniform = mask / mask.sum(axis=1, keepdims=True)
        p0 = (n0 + uniform) / (n0.sum(axis=1, keepdims=True) + 1)
        p1 = (n1 + alpha1*p0) / (n1.sum(axis=1, keepdims=True) + alpha1)
        p2 = (n2 + alpha2*p1) / (n2.sum(axis=1, keepdims=True) + alpha2)
        return p0, p1, p2


def nll(prob, candidates, target):
    return float(-np.log2(np.maximum(np.sum(prob*(candidates == np.asarray(target)[:, None]), axis=1), 1e-15)).sum())


def context_paths(train, test, candidates, prior, representation, order, Cs, config):
    output = {C: prior.copy() for C in Cs}
    audit = []
    groups = sorted(test.team_id.unique()) if representation == "players" else [None]
    for team in groups:
        tr = train if team is None else train.loc[train.team_id.eq(team)]
        mask = np.ones(len(test), dtype=bool) if team is None else test.team_id.eq(team).to_numpy()
        te = test.loc[mask]
        if tr.empty:
            audit.append({"team_id": int(team), "no_team_history_fallback": True})
            continue
        probabilities, log = fit_logistic_path(
            tr, te, context_features(tr, representation=representation, second_order=order == 2),
            context_features(te, representation=representation, second_order=order == 2),
            candidates[mask], prior[mask], Cs, config["max_iter"])
        for C in Cs:
            output[C][mask] = probabilities[C]
        audit.extend({**item, "team_id": team} for item in log)
    return output, audit


def fit_outer(frame, candidates, record, representation, config, progress=print):
    assert_split(record)
    train = frame.loc[frame.match_id.isin(record["train_match_ids"])]
    test = frame.loc[frame.match_id.isin(record["test_match_ids"])]
    validation_parts = []
    audit = []
    for inner in record["inner"]:
        tr = frame.loc[frame.match_id.isin(inner["train_match_ids"])]
        te = frame.loc[frame.match_id.isin(inner["test_match_ids"])]
        allowed = candidates[te.row_id]
        counts = MemoryCounts(tr).prepare(te, allowed)
        validation_parts.append((tr, te, allowed, counts))
    strengths = config["smoothing_grid"]
    scores1 = {a: 0. for a in strengths}
    for _, te, allowed, prepared in validation_parts:
        for alpha in strengths:
            scores1[alpha] += nll(MemoryCounts.probabilities(prepared, alpha)[1], allowed, te.target)
    alpha1 = min(scores1, key=scores1.get)
    scores2 = {a: 0. for a in strengths}
    for _, te, allowed, prepared in validation_parts:
        for alpha in strengths:
            scores2[alpha] += nll(MemoryCounts.probabilities(prepared, alpha1, alpha)[2], allowed, te.target)
    alpha2 = min(scores2, key=scores2.get)
    Cs = config["logistic_C_grid"]
    chosen_C = {}
    for order in [1, 2]:
        scores = {C: 0. for C in Cs}
        for inner_number, (tr, te, allowed, prepared) in enumerate(validation_parts):
            prior = MemoryCounts.probabilities(prepared, alpha1, alpha2)[0]
            predictions, log = context_paths(tr, te, allowed, prior, representation, order, Cs, config)
            for C in Cs:
                scores[C] += nll(predictions[C], allowed, te.target)
            audit.extend({**a, "stage": "inner", "inner_fold": inner_number, "order": order} for a in log)
        chosen_C[order] = min(scores, key=scores.get)
    allowed = candidates[test.row_id]
    prepared = MemoryCounts(train).prepare(test, allowed)
    p0, p1, p2 = MemoryCounts.probabilities(prepared, alpha1, alpha2)
    probabilities = {"M0": p0, "M1": p1, "M2": p2}
    for order in [1, 2]:
        pred, log = context_paths(train, test, allowed, p0, representation, order, [chosen_C[order]], config)
        probabilities[f"M{order}_context"] = pred[chosen_C[order]]
        audit.extend({**a, "stage": "outer", "inner_fold": -1, "order": order} for a in log)
    base = test[["sequence_id", "match_id", "team_id", "team", "opponent_id", "possession_key", "target"]].reset_index(drop=True)
    base["representation"] = representation
    base["repeat"], base["fold"] = record["repeat"], record["fold"]
    base["n_b_training"] = prepared["n_b"]
    base["n_ab_training"] = prepared["n_ab"]
    base["training_group_matches"] = prepared["train_group_matches"]
    base["target_seen_in_training"] = prepared["target_seen"]
    rows = []
    for name, prob in probabilities.items():
        metrics = probability_metrics(prob, allowed, test.target.to_numpy())
        rows.append(pd.concat([base, metrics], axis=1).assign(model=name))
    params = {"representation": representation, "repeat": record["repeat"], "fold": record["fold"],
              "alpha1": alpha1, "alpha2": alpha2, "C1": chosen_C[1], "C2": chosen_C[2],
              "train_rows": len(train), "test_rows": len(test)}
    return pd.concat(rows, ignore_index=True), params, [{**a, **{k: params[k] for k in ["representation", "repeat", "fold"]}} for a in audit], test, allowed, probabilities


def summarize_memory(predictions, config):
    primary = predictions.loc[predictions["repeat"].eq(0)]
    metrics = predictions.groupby(["representation", "repeat", "model"], sort=False).agg(
        rows=("sequence_id", "size"), log_loss_bits=("log_loss_bits", "mean"),
        brier=("brier", "mean"), accuracy=("correct", "mean"), top3_accuracy=("top3_correct", "mean")).reset_index()
    comparisons, by_team, by_match, sensitivity = [], [], [], []
    for representation, frame in primary.groupby("representation", sort=False):
        for first, second in [("M1", "M2"), ("M1_context", "M2_context"), ("M0", "M1")]:
            left = frame.loc[frame.model.eq(first), ["sequence_id", "match_id", "team_id", "opponent_id", "log_loss_bits"]]
            right = frame.loc[frame.model.eq(second), ["sequence_id", "log_loss_bits"]]
            paired = left.merge(right, on="sequence_id", suffixes=("_first", "_second"), validate="one_to_one")
            paired["gain_bits"] = paired.log_loss_bits_first - paired.log_loss_bits_second
            label = f"{second}_vs_{first}"
            comparisons.append({"representation": representation, "comparison": label,
                **paired_match_interval(paired, repetitions=config["bootstrap_repetitions"], seed=config["seed"])})
            for mid, group in paired.groupby("match_id"):
                by_match.append({"representation": representation, "comparison": label, "match_id": mid,
                                 "rows": len(group), "gain_bits": group.gain_bits.mean()})
            for team, group in paired.groupby("team_id"):
                by_team.append({"representation": representation, "comparison": label, "team_id": team,
                                **paired_match_interval(group, repetitions=config["bootstrap_repetitions"], seed=config["seed"])})
            for team in sorted(set(paired.team_id) | set(paired.opponent_id)):
                keep = paired.loc[paired.team_id.ne(team) & paired.opponent_id.ne(team)]
                sensitivity.append({"representation": representation, "comparison": label, "excluded_team_id": team,
                                    "rows": len(keep), "gain_bits": keep.gain_bits.mean(),
                                    "method": "fixed-OOF delete-team evaluation subset; models not refitted"})
    coverage = primary.groupby(["representation", "model"]).agg(
        rows=("sequence_id", "size"), unseen_B=("n_b_training", lambda x: int(x.eq(0).sum())),
        unseen_AB=("n_ab_training", lambda x: int(x.eq(0).sum())),
        unseen_target=("target_seen_in_training", lambda x: int((~x).sum())),
        no_group_history=("training_group_matches", lambda x: int(x.eq(0).sum()))).reset_index()
    return {"model_metrics": metrics, "memory_comparisons": pd.DataFrame(comparisons),
            "memory_by_team": pd.DataFrame(by_team), "memory_by_match": pd.DataFrame(by_match),
            "delete_team_sensitivity": pd.DataFrame(sensitivity), "prediction_coverage": coverage,
            "calibration": calibration_table(predictions)}


def rolling_counts(frame, candidates, matches, representation):
    """Strictly earlier calendar days; smoothing=10 fixed before this sensitivity."""
    records = []
    dates = matches.set_index("match_id").date.to_dict()
    for mid in matches.match_id:
        test = frame.loc[frame.match_id.eq(mid)]
        train = frame.loc[frame.match_id.map(dates).lt(dates[mid])]
        if test.empty:
            continue
        allowed = candidates[test.row_id]
        prepared = MemoryCounts(train).prepare(test, allowed)
        for name, prob in zip(["M0", "M1", "M2"], MemoryCounts.probabilities(prepared, 10, 10)):
            base = test[["sequence_id", "match_id", "team_id", "target"]].reset_index(drop=True)
            base["representation"], base["model"] = representation, name
            base["training_group_matches"] = prepared["train_group_matches"]
            base["has_history"] = np.array(prepared["train_group_matches"]) > 0
            records.append(pd.concat([base, probability_metrics(prob, allowed, test.target.to_numpy())], axis=1))
    return pd.concat(records, ignore_index=True)
