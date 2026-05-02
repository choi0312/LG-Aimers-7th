# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lg_aimers_7th.pipeline import run_full_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Run LG Aimers 7th modular demand forecasting pipeline.")
    parser.add_argument("--data-root", type=str, default="data", help="Competition data root. Default: ./data")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory. Default: ./outputs")
    parser.add_argument("--n-estimators", type=int, default=1200)
    return parser.parse_args()

def main():
    args = parse_args()
    run_full_pipeline(args.data_root, args.output_dir, n_estimators=args.n_estimators)

if __name__ == "__main__":
    main()
