# scripts/eval_baselines.py
import argparse

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score


def main():
    ap = argparse.ArgumentParser(description="Evaluate baselines vs CEM-TAD+ on a scored CSV.")
    ap.add_argument(
        "--input", required=True, help="Path to scored CSV (must contain Abnormality_score)"
    )
    ap.add_argument("--timestamp_col", default="Time")
    ap.add_argument("--normal_start", required=True)
    ap.add_argument("--normal_end", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=[args.timestamp_col])

    train_mask = (df[args.timestamp_col] >= args.normal_start) & (
        df[args.timestamp_col] <= args.normal_end
    )
    y = (~train_mask).astype(int).values

    # numeric features (exclude score & Top-7)
    feat_cols = [
        c
        for c in df.columns
        if c not in [args.timestamp_col, "Abnormality_score"]
        and not c.startswith("top_feature_")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[feat_cols].to_numpy(dtype=float)
    X_train = df.loc[train_mask, feat_cols].to_numpy(dtype=float)

    # Baseline 1: max |z| per row
    mu = np.nanmean(X_train, axis=0)
    sd = np.nanstd(X_train, axis=0)
    sd[sd == 0] = 1e-12
    Z = (X - mu) / sd
    max_abs_z = np.nanmax(np.abs(Z), axis=1)

    # Baseline 2: Robust Mahalanobis (Ledoit–Wolf)
    lw = LedoitWolf().fit(X_train)
    center, prec = lw.location_, lw.precision_
    dm2 = ((X - center) @ prec * (X - center)).sum(axis=1)

    ens = df["Abnormality_score"].to_numpy(dtype=float)

    def eval_row(name, score):
        auroc = roc_auc_score(y, score)
        ap = average_precision_score(y, score)
        return name, auroc, ap

    rows = [
        eval_row("CEM-TAD+ (0–100)", ens),
        eval_row("Max |z| per row", max_abs_z),
        eval_row("Mahalanobis (LW)", dm2),
    ]

    s_train = df.loc[train_mask, "Abnormality_score"].to_numpy(dtype=float)
    print(
        "Training KPIs:",
        f"mean={s_train.mean():.3f}",
        f"max={s_train.max():.3f}",
        f"p99={np.quantile(s_train, 0.99):.3f}",
    )
    print("\nMethod                 AUROC     AP")
    for n, a, p in rows:
        print(f"{n:22s} {a:7.3f} {p:7.3f}")

    rest = df.loc[~train_mask, "Abnormality_score"].to_numpy(dtype=float)
    thr = max(60.0, float(np.percentile(rest, 95))) if rest.size else 60.0
    print(f"\nSuggested threshold ≈ {thr:.1f}  (max(60, 95th pct of non-train))")


if __name__ == "__main__":
    main()
