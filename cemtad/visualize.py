from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def plot_scores_time(
    df: pd.DataFrame, timestamp_col: str, output_path: str, normal_start: str, normal_end: str
) -> None:
    _ensure_dir(os.path.dirname(output_path))
    t = pd.to_datetime(df[timestamp_col])
    s = df["Abnormality_score"].values
    plt.figure()
    plt.plot(t, s)
    plt.axhline(25, linestyle="--")
    plt.axhline(60, linestyle="--")
    plt.title("Abnormality Score Over Time")
    plt.xlabel("Time")
    plt.ylabel("Score (0–100)")
    ns = pd.to_datetime(normal_start)
    ne = pd.to_datetime(normal_end)
    plt.axvspan(ns, ne, alpha=0.15)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_score_histogram(
    df: pd.DataFrame, timestamp_col: str, normal_start: str, normal_end: str, output_path: str
) -> None:
    _ensure_dir(os.path.dirname(output_path))
    t = pd.to_datetime(df[timestamp_col])
    train_mask = (t >= pd.to_datetime(normal_start)) & (t <= pd.to_datetime(normal_end))
    s_train = df.loc[train_mask, "Abnormality_score"].values
    s_rest = df.loc[~train_mask, "Abnormality_score"].values
    plt.figure()
    plt.hist(s_train, bins=50, alpha=0.7, density=True)
    plt.hist(s_rest, bins=50, alpha=0.5, density=True)
    plt.title("Score Distribution: Train vs Rest")
    plt.xlabel("Score (0–100)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_top_features_bar(df: pd.DataFrame, row_idx: int, output_path: str) -> None:
    _ensure_dir(os.path.dirname(output_path))
    cols = [f"top_feature_{i}" for i in range(1, 8)]
    feats = [
        df.iloc[row_idx][c]
        for c in cols
        if isinstance(df.iloc[row_idx][c], str) and df.iloc[row_idx][c] != ""
    ]
    vals = list(range(len(feats), 0, -1))
    plt.figure()
    plt.barh(feats, vals)
    plt.title(f"Top Contributors @ row {row_idx}")
    plt.xlabel("Rank (proxy)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
