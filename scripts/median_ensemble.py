# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lg_aimers_7th.postprocess import median_ensemble

def parse_args():
    parser = argparse.ArgumentParser(description="Median ensemble two DACON submission files.")
    parser.add_argument("--path-a", required=True)
    parser.add_argument("--path-b", required=True)
    parser.add_argument("--out-path", default="outputs/submission_median_ensemble.csv")
    parser.add_argument("--min-clip", type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    out_path = median_ensemble(args.path_a, args.path_b, args.out_path, min_clip=args.min_clip)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
