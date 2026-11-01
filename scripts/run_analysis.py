"""Run the complete local analysis without starting Jupyter."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
from src.analysis import run_analysis

if __name__ == "__main__":
    dataset, tables, figures = run_analysis(ROOT)
    print(tables["teams"].to_string(index=False))
    print(f"\nValidated {len(dataset['matches'])} matches, {len(tables['graphs'])} aggregate networks; saved {len(figures)} aggregate PNG + SVG figures, plus all fixed-lineup figures.")
    print(ROOT / "results/report.html")
    print(ROOT / "results/fixed_lineups/report.html")
