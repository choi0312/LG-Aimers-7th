# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lg_aimers_7th.postprocess import median_ensemble


def parse_args():
    parser = argparse.ArgumentParser(description="Median ensemble two submission files.")
    parser.add_argument("--path-a", required=True, help="First submission file")
    parser.add_argument("--path-b", required=True, help="Second submission file")
    parser.add_argument("--out-path", default="outputs/submission_median_ensemble.csv", help="Output path")
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = median_ensemble(args.path_a, args.path_b, args.out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
