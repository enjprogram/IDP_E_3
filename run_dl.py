# run_dl.py
"""
Run this script to train all deep learning models.
Results and TensorBoard logs are cached — re-runs load from saved files.

Usage:
    python run_dl.py                                         # all industries, all models
    python run_dl.py --industries C G K                      # specific industries
    python run_dl.py --models prophet lstm                   # specific models
    python run_dl.py --industries C --models prophet lstm    # combined

After training, launch TensorBoard in a separate terminal:
    tensorboard --logdir results/dl/tensorboard_logs
Then open: http://localhost:6006
"""

import argparse
from src.deep_learning import run_dl_pipeline

parser = argparse.ArgumentParser(description="RFSD Deep Learning Pipeline")
parser.add_argument("--industries", nargs="*", default=None,
                    help="Industry codes e.g. C G K (default: all)")
parser.add_argument("--models", nargs="*",
                    default=["prophet", "lstm", "gru", "nbeats"],
                    help="Models to run (default: all four)")
parser.add_argument("--parquet",    default="data/raw/rfsd_2011_2024.parquet")
parser.add_argument("--lagged-df",  default="results/lagged_df.pkl")
parser.add_argument("--test-ratio", type=float, default=0.2)
args = parser.parse_args()

print("=" * 60)
print("  RFSD Deep Learning Forecasting Pipeline")
print("=" * 60)
print(f"  Models:     {args.models}")
print(f"  Industries: {args.industries or 'all'}")
print(f"  Test ratio: {args.test_ratio}")
print(f"  Features:   lagged_df (DFM factors + lags + ratios)")
print("=" * 60)

run_dl_pipeline(
    parquet_path=args.parquet,
    lagged_df_path=args.lagged_df,
    industries=args.industries,
    test_ratio=args.test_ratio,
    models=args.models,
)

print("\nDone.")
print("\nNext steps:")
print(" 1. TensorBoard: tensorboard --logdir results/dl/tensorboard_logs")
print(" 2. App:         streamlit run app.py")