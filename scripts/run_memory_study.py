"""Run steps 2 (memory prediction) and 3 (downstream player effectiveness)."""
from pathlib import Path
import argparse
import json
import os
import sys
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(variable, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/memory_study.json")
    parser.add_argument("--stage", choices=["all", "memory", "continuation", "report"], default="all")
    parser.add_argument("--representation", choices=["zones_6x3", "zones_4x3", "players"])
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    from src.study import run_memory_stage
    if args.stage in {"all", "memory"}:
        run_memory_stage(ROOT, config, args.representation, progress=lambda x: print(x, flush=True))
    if args.stage in {"all", "continuation"}:
        from src.continuation import run_continuation_stage
        run_continuation_stage(ROOT, config, progress=lambda x: print(x, flush=True))
    if args.stage in {"all", "report"}:
        from src.study_report import build_study_report
        build_study_report(ROOT, config)
