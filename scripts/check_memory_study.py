"""Independently rescore step-2/3 exports and check splits, targets and provenance."""
from pathlib import Path
import json
import re
import sys
import unittest
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation import (load_study_inputs, assert_split, probability_metrics,
                            context_features, paired_match_interval, hierarchical_weights)
from src.continuation import values_from_material, base_outcomes
from src.sequences import sha256
from src.study import write_json


def main():
    out = ROOT / "results/memory"
    config = json.loads((ROOT / "configs/memory_study.json").read_text())
    tables, hashes = load_study_inputs(ROOT)
    source_ids = set(tables["two_pass_sequences"].loc[lambda f: f.memory_target_eligible].sequence_id)
    assert len(source_ids) == 33358
    overall = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert overall["inputs"] == hashes and overall["config"] == config
    for path, expected in overall["code"].items():
        assert sha256(ROOT / path) == expected, ("code provenance", path)
    for path, expected in overall["outputs"].items():
        p = ROOT / path
        assert p.is_file() and sha256(p) == expected["sha256"] and p.stat().st_size == expected["bytes"], path
    checks = {"status": "passed", "verified_inputs": len(hashes),
              "output_hashes_verified": len(overall["outputs"]), "primary_paths": len(source_ids)}
    primary_records = None
    total_probabilities = total_fits = 0
    checked_comparisons = 0
    comparisons = pd.read_csv(out / "tables/memory_comparisons.csv")
    memory_audits = []
    for rep in config["representations"]:
        folder = out / rep
        run = json.loads((folder / "run_manifest.json").read_text())
        assert run["inputs"] == hashes and run["config"] == config
        for p, expected in run["code"].items(): assert sha256(ROOT / p) == expected
        records = json.loads((folder / "split_manifest.json").read_text())["splits"]
        assert len(records) == 15
        for record in records: assert_split(record)
        prediction = pd.read_csv(folder / "oof_predictions.csv.gz")
        assert not prediction.duplicated(["repeat", "model", "sequence_id"]).any()
        for (repeat, model), group in prediction.groupby(["repeat", "model"]):
            assert set(group.sequence_id) == source_ids
            assert len(group) == 33358
            for record in [r for r in records if r["repeat"] == repeat]:
                actual = group.loc[group.fold.eq(record["fold"])]
                assert set(actual.match_id) == set(record["test_match_ids"])
                assert set(actual.match_id).isdisjoint(record["train_match_ids"])
        assert np.isfinite(prediction.log_loss_bits).all()
        assert prediction.p_true.between(0, 1, inclusive="right").all()
        assert prediction.brier.between(0, 2).all()
        primary = prediction.loc[prediction["repeat"].eq(0)]
        primary_records = [r for r in records if r["repeat"] == 0]
        rescored_ids = set()
        for record in primary_records:
            p = folder / f"r0_f{record['fold']}_probabilities.npz"
            assert sha256(p) == run["probability_files"][p.name]
            with np.load(p, allow_pickle=False) as saved:
                ids = saved["sequence_id"]
                assert not rescored_ids.intersection(ids)
                rescored_ids.update(ids)
                for model in ["M0", "M1", "M2", "M1_context", "M2_context"]:
                    prob = saved[model]
                    metrics = probability_metrics(prob, saved["candidates"], saved["target"])
                    actual = primary.loc[primary.model.eq(model)].set_index("sequence_id").loc[ids]
                    np.testing.assert_array_equal(actual.target.to_numpy(), saved["target"])
                    for c in metrics:
                        np.testing.assert_allclose(actual[c].to_numpy(), metrics[c].to_numpy(), rtol=1e-9, atol=1e-10)
                    total_probabilities += len(prob)
        assert rescored_ids == source_ids
        for first, second in [("M1", "M2"), ("M1_context", "M2_context"), ("M0", "M1")]:
            p = primary.pivot(index=["sequence_id", "match_id"], columns="model", values="log_loss_bits").reset_index()
            p["gain_bits"] = p[first] - p[second]
            interval = paired_match_interval(p, repetitions=config["bootstrap_repetitions"], seed=config["seed"])
            actual = comparisons.loc[comparisons.representation.eq(rep) & comparisons.comparison.eq(f"{second}_vs_{first}")].iloc[0]
            for c, value in interval.items(): np.testing.assert_allclose(actual[c], value, rtol=1e-9, atol=1e-10)
            checked_comparisons += 1
        audit = pd.read_csv(folder / "fit_audit.csv")
        assert audit.converged.dropna().all()
        total_fits += int(audit.converged.notna().sum())
        memory_audits.append({"representation": rep, "audited_fits": int(audit.converged.notna().sum()),
                              "fallback_rows": int(audit.converged.isna().sum())})
        rolling = pd.read_csv(folder / "rolling_predictions.csv.gz")
        assert not rolling.duplicated(["sequence_id", "model"]).any()
        assert set(rolling.sequence_id) == source_ids
        assert (rolling.loc[rolling.has_history, "training_group_matches"] > 0).all()
    checks.update(outer_memory_experiments=45, probability_rows_rescored=total_probabilities,
                  paired_intervals_recomputed=checked_comparisons, memory_fit_audits=memory_audits)
    print("Memory: splits, OOF probabilities, all scoring rules, convergence and paired intervals verified.", flush=True)

    folder = ROOT / "results/continuation"
    run = json.loads((folder / "run_manifest.json").read_text())
    assert run["inputs"] == hashes and run["config"] == config
    for p, expected in run["code"].items(): assert sha256(ROOT / p) == expected
    meta = json.loads((folder / "xt_models.json").read_text())
    assert meta["split_manifest"] == primary_records and len(meta["models"]) == 20
    for model in meta["models"]:
        record = primary_records[model["fold"]]
        expected = record if model["level"] == "outer" else record["inner"][model["inner_fold"]]
        assert model["training_match_ids"] == expected["train_match_ids"]
        assert set(model["training_match_ids"]).isdisjoint(expected["test_match_ids"])
        assert set(model["training_match_ids"]).isdisjoint(record["test_match_ids"])
        assert model["bellman_residual"] < 1e-10
        assert np.isfinite(model["values"]).all()
    oof = pd.read_csv(folder / "tables/oof_continuation.csv.gz", low_memory=False)
    assert len(oof) == 41927 and oof.sequence_id.is_unique
    assert set(oof.sequence_id) == set(tables["receiver_sequences"].sequence_id)
    material = pd.read_csv(folder / "tables/outcome_material.csv.gz")
    assert len(material) == 3*len(oof) and not material.duplicated(["sequence_id", "horizon"]).any()
    events = tables["events"].set_index("event_id")
    shots = material.loc[material.status.eq("shot")]
    assert events.loc[shots.terminal_event_id, "event_type"].eq("Shot").all()
    np.testing.assert_allclose(shots.shot_xg, events.loc[shots.terminal_event_id, "xg"])
    original = base_outcomes(tables).set_index("sequence_id").loc[oof.sequence_id]
    for c in ["receiver_loss", "box3", "shot5", "shot10", "shot20", "xg10", "triple_observed"]:
        np.testing.assert_allclose(oof[c].to_numpy(dtype=float), original[c].to_numpy(dtype=float), equal_nan=True)
    for record in primary_records:
        subset = oof.loc[oof.fold.eq(record["fold"])]
        assert set(subset.match_id) == set(record["test_match_ids"])
        model = next(m for m in meta["models"] if m["level"] == "outer" and m["fold"] == record["fold"])
        expected = values_from_material(material, model, subset.sequence_id)
        for c in ["R1", "R3", "R5"]:
            np.testing.assert_allclose(subset[c], expected[c], atol=1e-12, equal_nan=True)
    tasks = pd.read_csv(folder / "tables/outcome_model_metrics.csv").outcome.tolist()
    for task in tasks:
        value = "R3" if task == "R3_with_B" else task
        actual, predicted = oof[value], oof[f"pred_{task}"]
        if task == "R3_triple": assert oof.loc[~oof.triple_observed, value].isna().all()
        assert np.isfinite(predicted).all() and predicted.ge(0).all()
        if task != "xg10": assert predicted.le(1).all()
        np.testing.assert_allclose(oof[f"residual_{task}"], actual-predicted, atol=1e-12, equal_nan=True)
        if task != "xg10": assert actual.dropna().between(0, 1).all()
    primary_censored = set(material.loc[material.horizon.eq(3) & material.status.eq("censored")].sequence_id)
    assert oof.set_index("sequence_id").loc[list(primary_censored), "R3"].isna().all()
    audit = pd.read_csv(folder / "tables/fit_audit.csv")
    assert len(audit) == 690 and audit.converged.all()
    assert audit.loc[audit.level.eq("outer")].groupby("outcome").size().eq(5).all()
    total_fits += len(audit)
    known_features = context_features(oof)
    assert not any(c in known_features for c in ["a_id", "a_x", "c_id", "b_dx", "receiver_resolution_type"])
    support_counts = {}
    for level, file, keys in [("player", "player_effects", ["team_id", "a_id"]),
                              ("pair", "pair_effects", ["team_id", "a_id", "b_id"]),
                              ("triple", "triple_effects", ["team_id", "a_id", "b_id", "c_id"])]:
        saved = pd.read_csv(folder / f"tables/{file}.csv")
        sample = oof.loc[oof.triple_observed] if level == "triple" else oof
        for metric, effect in saved.groupby("metric"):
            value = "R3" if metric == "R3_with_B" else metric
            grouped = sample.groupby(keys)
            base = effect.set_index(keys)
            np.testing.assert_allclose(base.initial_passes, grouped.size().reindex(base.index))
            np.testing.assert_allclose(base.observed_passes, grouped[value].count().reindex(base.index))
            np.testing.assert_allclose(base.raw_per100, 100*grouped[value].mean().reindex(base.index), equal_nan=True, atol=1e-10)
            np.testing.assert_allclose(base.adjusted_excess_per100,
                    100*grouped[f"residual_{metric}"].mean().reindex(base.index), equal_nan=True, atol=1e-10)
            expected = effect.observed_passes.ge(config[f"{level}_min_passes"]) & effect.observed_matches.ge(config[f"{level}_min_matches"])
            assert (effect.support_eligible == expected).all()
            assert effect.loc[~expected, "ci95_low"].isna().all()
            ranked = effect.loc[effect.ranking_eligible]
            assert ranked.ci95_low.notna().all() and ranked.ci95_high.ge(ranked.ci95_low).all()
            assert ranked.bootstrap_valid_fraction.ge(.8).all()
            support_counts[f"{level}_{metric}"] = int(len(ranked))
    # Independently reproduce the principal interval for the largest point estimate.
    p = pd.read_csv(folder / "tables/player_effects.csv")
    leader = p.loc[p.metric.eq("R3") & p.ranking_eligible].sort_values("adjusted_excess_per100", ascending=False).iloc[0]
    blocks, weights = hierarchical_weights(oof, config["bootstrap_repetitions"], config["seed"])
    selected = oof.loc[oof.a_id.eq(leader.a_id) & oof.R3.notna()]
    by_block = selected.groupby("possession_key").residual_R3.agg(["sum", "count"]).reindex(blocks.possession_key).fillna(0)
    denom = weights @ by_block["count"].to_numpy()
    draws = np.divide(100*(weights @ by_block["sum"].to_numpy()), denom,
                      out=np.full(len(weights), np.nan), where=denom>0)
    np.testing.assert_allclose([leader.ci95_low, leader.ci95_high], np.nanquantile(draws, [.025, .975]), atol=1e-10)
    checks.update(continuation_initial_passes=len(oof), continuation_outcome_fits=len(audit),
                  train_only_xt_fits=20, main_R3_observed=int(oof.R3.notna().sum()),
                  material_rows=len(material), supported_entities=support_counts,
                  principal_player_interval_independently_recomputed=True, converged_fitted_models=total_fits)
    print("Continuation: train-only xT, source targets, residuals, all entity means/support and primary bootstrap verified.", flush=True)
    dictionary = pd.read_csv(out / "data_dictionary.csv")
    assert dictionary.description.notna().all()
    assert not dictionary.duplicated(["file", "column"]).any()
    for path, rows in dictionary.groupby("file"):
        assert set(pd.read_csv(ROOT / path, nrows=1).columns) == set(rows.column)
    figures = json.loads((out / "figure_index.json").read_text())
    for item in figures:
        path = out / "figures" / item["file"]
        with Image.open(path) as image: image.verify()
        ET.parse(path.with_suffix(".svg"))
    links = 0
    for ref in re.findall(r'(?:href|src)="([^"]+)"', (out / "report.html").read_text(encoding="utf-8")):
        if "://" not in ref and not ref.startswith("#"):
            assert (out / ref).is_file(), ref
            links += 1
    for doc in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        for ref in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            if "://" not in ref and not ref.startswith("#"):
                target = (doc.parent / ref.split("#")[0]).resolve()
                if target != (out / "delivery_check.json").resolve():
                    assert target.exists(), (doc, ref)
    checks.update(figures_png_svg=len(figures), documented_columns=len(dictionary), report_local_links=links,
                  full_unit_suite=f"{unittest.TestLoader().discover(str(ROOT/'tests')).countTestCases()} tests; run separately", completed_roadmap_stages=[1, 2, 3],
                  pending_roadmap_stages=[4, 5, 6, 7])
    write_json(out / "delivery_check.json", checks)
    assert (out / "delivery_check.json").is_file()
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
