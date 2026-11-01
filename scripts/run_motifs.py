"""Run roadmap step 4 only; preserves all prior experiment files."""
from pathlib import Path
import argparse
import json
import os
import sys
for variable in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ.setdefault(variable,"1")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/motifs.json")
    parser.add_argument("--stage",choices=["all","analysis","report"],default="all")
    args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text(encoding="utf-8"))
    base=json.loads((ROOT/config["base_config"]).read_text(encoding="utf-8"))
    if args.stage in {"all","analysis"}:
        from src.motifs import run_motifs
        run_motifs(ROOT,config,base,progress=lambda v:print(v,flush=True))
    if args.stage in {"all","report"}:
        from src.motif_report import build_report
        build_report(ROOT,config)
