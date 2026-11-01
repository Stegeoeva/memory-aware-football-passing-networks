"""Verify the exported step-1 files, typed CSV roundtrip, source and local links."""
from pathlib import Path
import json
import re
import sys
import xml.etree.ElementTree as ET

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sequences import sha256, source_inventory
from src.sequence_export import validate_euro_tables
from src.sequence_schema import RECEIPT_FEATURES


def main():
    out = ROOT / "results/euro_sequences"
    run = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((out / "schema.json").read_text(encoding="utf-8"))
    saved_source = json.loads((out / "source_manifest.json").read_text(encoding="utf-8"))
    _, source = source_inventory(ROOT)
    assert source == saved_source
    for path, expected in run["code_sha256"].items():
        assert sha256(ROOT / path) == expected, f"Source changed since build: {path}"
    for path, expected in run["outputs"].items():
        file = out / path
        assert file.is_file() and file.stat().st_size == expected["bytes"], path
        assert sha256(file) == expected["sha256"], path
    tables = {}
    for name, spec in schema["tables"].items():
        dtype = {col: detail["dtype"] for col, detail in spec["columns"].items()}
        frame = pd.read_csv(out / spec["file"], dtype=dtype)
        assert list(frame) == list(spec["columns"]), name
        assert len(frame) == spec["rows"], name
        tables[name] = frame
    # CSV represents both an empty exclusion reason and a null as a blank cell.
    tables["two_pass_sequences"]["memory_target_exclusion_reason"] = (
        tables["two_pass_sequences"].memory_target_exclusion_reason.fillna(""))
    validation = validate_euro_tables(tables, source)
    saved = json.loads((out / "validation.json").read_text(encoding="utf-8"))
    assert validation == saved
    assert run["models_fitted"] == run["evaluation_experiments_run"] == 0
    assert schema["receipt_feature_candidates"] == RECEIPT_FEATURES
    assert not any("release" in col or col.startswith(("stop_", "receiver_", "future_")) for col in RECEIPT_FEATURES)
    dictionary = pd.read_csv(out / "data_dictionary.csv")
    expected_columns = {(name, col) for name, frame in tables.items() for col in frame}
    assert set(zip(dictionary.table, dictionary.column)) == expected_columns
    assert dictionary.description.notna().all() and dictionary.provenance.notna().all()
    figures = json.loads((out / "figure_index.json").read_text(encoding="utf-8"))
    for item in figures:
        path = out / "figures" / item["file"]
        with Image.open(path) as img:
            img.verify()
        ET.parse(path.with_suffix(".svg"))
    link_count = 0
    for doc in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        for ref in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            if "://" not in ref and not ref.startswith("#"):
                assert (doc.parent / ref.split("#")[0]).exists(), (doc, ref)
                link_count += 1
    for ref in re.findall(r'(?:href|src)="([^"]+)"', (out / "report.html").read_text(encoding="utf-8")):
        if "://" not in ref and not ref.startswith("#"):
            assert (out / ref).exists(), ref
            link_count += 1
    result = {"status": "passed", "tables_roundtripped_with_schema": len(tables),
              "columns_documented": len(dictionary), "output_hashes_verified": len(run["outputs"]),
              "source_event_files_verified": 51, "code_hashes_verified": len(run["code_sha256"]),
              "png_and_svg_pairs_verified": len(figures), "local_links_checked": link_count,
              "models_fitted": 0, "evaluation_experiments_run": 0}
    (out / "delivery_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
