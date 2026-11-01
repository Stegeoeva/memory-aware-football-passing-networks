"""Build step-1 data for all 51 EURO 2024 games; no fitting or evaluation."""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sequence_export import build_euro_tables, export_euro_sequences

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-figures", action="store_true", help="Export data without coverage plots")
    args = parser.parse_args()
    started = time.perf_counter()
    tables, source = build_euro_tables(ROOT, progress=lambda message: print(message, flush=True))
    validation, directory = export_euro_sequences(tables, source, ROOT, figures=not args.no_figures)
    print(json.dumps(validation, indent=2), flush=True)
    print(f"Finished in {time.perf_counter()-started:.1f} seconds: {directory}", flush=True)
