"""Shared, match-blocked validation and receipt-only feature construction."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from .analysis import ROOT
from .sequences import sha256


def load_study_inputs(root=ROOT):
    out = Path(root) / "results/euro_sequences"
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((out / "schema.json").read_text(encoding="utf-8"))
    tables, hashes = {}, {}
    for name in ["matches", "events", "receiver_sequences", "two_pass_sequences", "sequence_actions", "future_windows", "pass_attempts"]:
        spec = schema["tables"][name]
        path = out / spec["file"]
        actual = sha256(path)
        assert actual == manifest["outputs"][spec["file"]]["sha256"], path
        hashes[spec["file"]] = actual
        tables[name] = pd.read_csv(path, dtype={c: v["dtype"] for c, v in spec["columns"].items()})
    return tables, hashes


def fold_map(match_ids, folds, seed):
    ids = np.array(sorted(set(map(int, match_ids))))
    rng = np.random.default_rng(seed)
    ids = rng.permutation(ids)
    return {int(mid): int(i % folds) for i, mid in enumerate(ids)}


def split_records(match_ids, config):
    ids = sorted(set(map(int, match_ids)))
    records = []
    for repeat in range(config["repeats"]):
        assignment = fold_map(ids, config["outer_folds"], config["seed"] + repeat)
        for fold in range(config["outer_folds"]):
            train = [m for m in ids if assignment[m] != fold]
            test = [m for m in ids if assignment[m] == fold]
            inner = fold_map(train, config["inner_folds"], config["seed"] + repeat * 100 + fold * 7 + 11)
            inners = [{"fold": j, "train_match_ids": [m for m in train if inner[m] != j],
                       "test_match_ids": [m for m in train if inner[m] == j]} for j in range(config["inner_folds"])]
            records.append({"repeat": repeat, "fold": fold, "train_match_ids": train, "test_match_ids": test, "inner": inners})
    return records


def assert_split(record):
    train, test = set(record["train_match_ids"]), set(record["test_match_ids"])
    assert train and test and train.isdisjoint(test)
    for inner in record.get("inner", []):
        a, b = set(inner["train_match_ids"]), set(inner["test_match_ids"])
        assert a.isdisjoint(b) and a | b == train
        assert (a | b).isdisjoint(test)


def context_features(frame, *, representation=None, second_order=False, include_b=False):
    """Only variables observed no later than the initial reception."""
    x = frame.receipt_x.astype(float).fillna(60) / 120
    y = frame.receipt_y.astype(float).fillna(40) / 80
    result = pd.DataFrame({
        "reception_zone": frame.receipt_zone_6x3.fillna(-1).astype(int).astype(str),
        "receiver_role": frame.b_role.fillna("unknown").astype(str),
        "pressure": frame.b_pressure_state.fillna("unobserved_receipt").astype(str),
        "team": frame.team_id.astype(str), "opponent": frame.opponent_id.astype(str),
        "period": frame.period.astype(str),
        "time_bin": (frame.receipt_clock_seconds.astype(float) // 600).astype(int).astype(str),
        "score": frame.score_difference_at_receipt.astype(float).clip(-2, 2).astype(int).astype(str),
        "rx": x, "ry": y, "rx2": x*x, "ry2": y*y, "rxy": x*y,
    }, index=frame.index)
    if representation:
        result["actor_B"] = frame.state_b.astype(str)
        if second_order:
            result["previous_A"] = frame.state_a.astype(str)
            result["previous_AB"] = frame.state_a.astype(str) + ":" + frame.state_b.astype(str)
    elif include_b:
        result["receiver_B"] = frame.b_id.astype(str)
    assert not any(c in result for c in ["a_id", "c_id", "b_dx", "receiver_resolution_type"])
    return result


def encoder():
    numerical = ["rx", "ry", "rx2", "ry2", "rxy"]
    return ColumnTransformer(
        [("numeric", StandardScaler(), numerical),
         ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True),
          lambda frame: [c for c in frame if c not in numerical])],
        sparse_threshold=1)


def aligned_probabilities(raw, classes, candidates, prior, epsilon):
    """Respect the same available receiver set; assign positive unseen-player mass."""
    result = np.zeros(candidates.shape, dtype=float)
    index = {int(c): j for j, c in enumerate(classes)}
    for j in range(candidates.shape[1]):
        col = candidates[:, j]
        source = np.array([index.get(int(c), -1) for c in col])
        seen = source >= 0
        result[seen, j] = raw[np.arange(len(raw))[seen], source[seen]]
    mass = result.sum(axis=1, keepdims=True)
    result = np.divide(result, mass, out=prior.copy(), where=mass > 0)
    result = (1-epsilon) * result + epsilon * prior
    result[candidates < 0] = 0.
    result /= result.sum(axis=1, keepdims=True)
    return result


def fit_logistic_path(train, test, features_train, features_test, candidates, prior, Cs, max_iter=600):
    transform = encoder()
    X = transform.fit_transform(features_train)
    Z = transform.transform(features_test)
    target = train.target.to_numpy(dtype=int)
    classes = np.unique(target)
    results, audit = {}, []
    if len(classes) == 1:
        raw = np.ones((len(test), 1))
        for C in Cs:
            results[C] = aligned_probabilities(raw, classes, candidates, prior, 1/(len(train)+1))
        return results, [{"single_class_fallback": True, "training_rows": len(train)}]
    model = LogisticRegression(C=Cs[0], max_iter=max_iter, solver="lbfgs", warm_start=True, tol=1e-4)
    for C in Cs:
        model.set_params(C=C)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, target)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        results[C] = aligned_probabilities(model.predict_proba(Z), model.classes_, candidates, prior, 1/(len(train)+1))
        audit.append({"C": C, "training_rows": len(train), "features": X.shape[1],
                      "iterations": int(np.max(model.n_iter_)), "converged": converged})
    return results, audit


def probability_metrics(probabilities, candidates, target):
    valid = candidates >= 0
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-8)
    assert (probabilities[~valid] == 0).all()
    actual = candidates == np.asarray(target, dtype=int)[:, None]
    assert actual.sum(axis=1).min() == actual.sum(axis=1).max() == 1
    p_true = np.sum(probabilities * actual, axis=1)
    predicted = np.argmax(probabilities, axis=1)
    top3 = np.argsort(-probabilities, axis=1, kind="stable")[:, :3]
    return pd.DataFrame({
        "p_true": p_true, "log_loss_bits": -np.log2(np.maximum(p_true, 1e-15)),
        "brier": (probabilities**2).sum(axis=1)-2*p_true+1,
        "correct": candidates[np.arange(len(target)), predicted] == target,
        "top3_correct": np.take_along_axis(actual, top3, axis=1).any(axis=1),
        "predicted_target": candidates[np.arange(len(target)), predicted],
        "confidence": probabilities.max(axis=1),
    })


def paired_match_interval(frame, value="gain_bits", repetitions=2000, seed=20261101):
    grouped = frame.groupby("match_id")[value].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    weight = rng.multinomial(len(grouped), np.full(len(grouped), 1/len(grouped)), size=repetitions)
    draws = (weight @ grouped["sum"].to_numpy()) / (weight @ grouped["count"].to_numpy())
    means = frame.groupby("match_id")[value].mean().to_numpy()
    match_draws = weight @ means / len(means)
    return {"gain_bits": float(frame[value].mean()), "ci95_low": float(np.quantile(draws, .025)),
            "ci95_high": float(np.quantile(draws, .975)), "match_mean_gain_bits": float(means.mean()),
            "match_mean_ci95_low": float(np.quantile(match_draws, .025)),
            "match_mean_ci95_high": float(np.quantile(match_draws, .975)),
            "matches": len(grouped), "observations": len(frame)}


def calibration_table(predictions):
    rows = []
    keys = ["representation", "model", "repeat"]
    for key, group in predictions.groupby(keys, sort=False):
        bins = np.minimum(9, (group.confidence * 10).astype(int))
        for bucket, g in group.groupby(bins):
            rows.append({**dict(zip(keys, key)), "bin": int(bucket), "n": len(g),
                         "confidence": g.confidence.mean(), "accuracy": g.correct.mean()})
    return pd.DataFrame(rows)


def hierarchical_weights(sequences, repetitions, seed):
    """Resample matches, then whole possession blocks; overlapping rows stay together."""
    blocks = sequences[["match_id", "possession_key"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    matches = sorted(blocks.match_id.unique())
    weights = np.zeros((repetitions, len(blocks)), dtype=np.int16)
    match_draws = rng.multinomial(len(matches), np.full(len(matches), 1/len(matches)), size=repetitions)
    for j, mid in enumerate(matches):
        positions = np.flatnonzero(blocks.match_id.eq(mid).to_numpy())
        for repeat, copies in enumerate(match_draws[:, j]):
            if copies:
                weights[repeat, positions] = rng.multinomial(int(copies)*len(positions), np.full(len(positions), 1/len(positions)))
    return blocks, weights
