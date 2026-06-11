from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible Olist delivery analytics outputs.")
    parser.add_argument("raw_dir", nargs="?", default="data/raw", help="Directory containing raw Olist CSV files.")
    parser.add_argument("output_dir", nargs="?", default="data", help="Directory where analytical outputs will be written.")
    parser.add_argument(
        "--sample-orders",
        type=int,
        default=None,
        help="Optional deterministic order sample size for quick test runs.",
    )
    args = parser.parse_args()
    run_pipeline(Path(args.raw_dir), Path(args.output_dir), sample_orders=args.sample_orders)
