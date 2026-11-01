"""Create the complete event-data and downstream-pass analysis."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.extended import run_extended

if __name__=='__main__':
    import json
    _,_,_,_,validation=run_extended(ROOT)
    print(json.dumps(validation,indent=2))
    print(ROOT/'results/extended/report.html')
