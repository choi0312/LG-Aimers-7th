# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lg_aimers_7th.pipeline import run_ensemble_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run LG Aimers 7th demand forecasting pipeline.")
    parser.add_argument("--data-root", type=str, required=True, help="Competition data root directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--n-estimators", type=int, default=1200, help="Number of LightGBM trees")
    parser.add_argument("--use-gpu", action="store_true", help="Use LightGBM GPU if available")
    return parser.parse_args()


def main():
    args = parse_args()
    run_ensemble_pipeline(
        data_root=args.data_root,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        use_gpu=args.use_gpu,
    )


if __name__ == "__main__":
    main()
