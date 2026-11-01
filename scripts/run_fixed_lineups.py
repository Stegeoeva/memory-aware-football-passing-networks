"""Render every observed fixed lineup for both teams in each match, including extra time."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.lineups import run_fixed_lineups

if __name__=="__main__":
    _,result,figures,validation=run_fixed_lineups(ROOT)
    import json
    print(json.dumps(validation,indent=2))
    print(f"Saved {len(figures)} PNG and SVG figures.")
    print(ROOT/"results/fixed_lineups/report.html")
