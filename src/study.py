"""Run the authorized prediction and continuation stages, with resumable folds."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .analysis import ROOT
from .sequences import sha256
from .evaluation import load_study_inputs, split_records, assert_split
from .memory import nominal_candidates, prepare_representation, fit_outer, rolling_counts, summarize_memory


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist()), encoding="utf-8")


def study_key(root, config, inputs, modules):
    payload = {"config": config, "inputs": inputs, "code": {m: sha256(root / m) for m in modules}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20], payload


def run_memory_stage(root=ROOT, config=None, representation=None, progress=print):
    root = Path(root)
    tables, hashes = load_study_inputs(root)
    key, provenance = study_key(root, config, hashes, ["src/memory.py", "src/evaluation.py", "src/lineups.py"])
    records = split_records(tables["matches"].match_id, config)
    for record in records:
        assert_split(record)
    representations = [representation] if representation else config["representations"]
    choices = nominal_candidates(tables["receiver_sequences"], root) if "players" in representations else None
    for rep in representations:
        out = root / "results/memory" / rep
        out.mkdir(parents=True, exist_ok=True)
        cache = root / ".runtime/memory_study" / key / rep
        cache.mkdir(parents=True, exist_ok=True)
        frame, candidates, excluded = prepare_representation(tables["receiver_sequences"], tables["two_pass_sequences"], rep, choices)
        predictions, parameters, fit_audit = [], [], []
        excluded.to_csv(out / "excluded_targets.csv", index=False)
        for record in records:
            stem = f"r{record['repeat']}_f{record['fold']}"
            checkpoint = cache / f"{stem}.json"
            if checkpoint.exists():
                meta = json.loads(checkpoint.read_text(encoding="utf-8"))
                pred = pd.read_csv(cache / f"{stem}.csv.gz")
                params, audit = meta["parameters"], meta["fit_audit"]
                if progress: progress(f"{rep} {stem}: restored verified-run checkpoint")
            else:
                if progress: progress(f"{rep} {stem}: nested training started")
                with threadpool_limits(config["blas_threads"]):
                    pred, params, audit, test, allowed, probabilities = fit_outer(frame, candidates, record, rep, config, progress)
                pred.to_csv(cache / f"{stem}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
                if record["repeat"] == 0:
                    np.savez_compressed(cache / f"{stem}_probabilities.npz",
                        sequence_id=test.sequence_id.to_numpy(dtype=str), candidates=allowed,
                        target=test.target.to_numpy(dtype=np.int64), **probabilities)
                write_json(checkpoint, {"parameters": params, "fit_audit": audit})
                if progress: progress(f"{rep} {stem}: completed; alpha=({params['alpha1']},{params['alpha2']}), C=({params['C1']},{params['C2']})")
            predictions.append(pred); parameters.append(params); fit_audit.extend(audit)
        prediction = pd.concat(predictions, ignore_index=True)
        assert not prediction.duplicated(["repeat", "model", "sequence_id"]).any()
        prediction.to_csv(out / "oof_predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        pd.DataFrame(parameters).to_csv(out / "selected_parameters.csv", index=False)
        pd.DataFrame(fit_audit).to_csv(out / "fit_audit.csv", index=False)
        rolling = rolling_counts(frame, candidates, tables["matches"], rep)
        rolling.to_csv(out / "rolling_predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        probability_files = {}
        for record in records:
            if record["repeat"] != 0:
                continue
            stem = f"r0_f{record['fold']}_probabilities.npz"
            destination = out / stem
            destination.write_bytes((cache / stem).read_bytes())
            probability_files[stem] = sha256(destination)
        write_json(out / "split_manifest.json", {"splits": records, "unit": "whole match; both teams together"})
        write_json(out / "run_manifest.json", {**provenance, "run_key": key, "representation": rep,
            "rows_per_model_repeat": len(frame), "excluded_targets": len(excluded),
            "probability_files": probability_files,
            "context_features": "receipt context; A/AB added only in second order",
            "rolling_policy": "strictly previous calendar dates; alpha1=alpha2=10 fixed",
            "identity_candidates": "nominal on-field roster known at receipt, excluding B; temporary absences retained"})
        write_json(out / "DONE.json", {"run_key": key, "status": "complete"})
        if progress: progress(f"{rep}: all repeats and rolling sensitivity exported")


def assemble_memory(root, config):
    parts, rolling = [], []
    for rep in config["representations"]:
        out = root / "results/memory" / rep
        assert json.loads((out / "DONE.json").read_text())["status"] == "complete"
        parts.append(pd.read_csv(out / "oof_predictions.csv.gz"))
        rolling.append(pd.read_csv(out / "rolling_predictions.csv.gz"))
    predictions = pd.concat(parts, ignore_index=True)
    summaries = summarize_memory(predictions, config)
    rolling = pd.concat(rolling, ignore_index=True)
    summaries["rolling_metrics"] = rolling.loc[rolling.has_history].groupby(["representation", "model"]).agg(
        observations=("sequence_id", "size"), matches=("match_id", "nunique"),
        log_loss_bits=("log_loss_bits", "mean"), accuracy=("correct", "mean")).reset_index()
    out = root / "results/memory/tables"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in summaries.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    return predictions, summaries
