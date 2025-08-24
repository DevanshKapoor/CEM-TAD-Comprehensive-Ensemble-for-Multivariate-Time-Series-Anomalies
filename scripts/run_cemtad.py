
from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd
from cemtad.data import DataProcessor
from cemtad.experts import expert_pca, expert_mahalanobis, expert_isoforest, expert_corrshift, expert_patch_forecaster
from cemtad.ensemble import combine_experts
from cemtad.attribution import top_contributors_from_errors
from cemtad.scoring import robust_calibration_params, alpha_for_q99, raw_to_score, smooth_scores
from cemtad.utils import setup_logger
from cemtad.visualize import plot_scores_time, plot_score_histogram, plot_top_features_bar
from datetime import timedelta
import numpy as np
import pandas as pd

def _assert_train_span(ts: pd.Series, start: str, end: str, min_hours: int = 72):
    t0, t1 = pd.to_datetime(start), pd.to_datetime(end)
    span_h = (t1 - t0) / timedelta(hours=1)
    if span_h < min_hours:
        print(f"[WARN] Training window is only {span_h:.1f} h (<{min_hours} h). "
              "Proceeding but results may be unstable.")

def _warn_if_train_anomalous(scores: pd.Series, max_ok: float = 25.0, mean_ok: float = 10.0):
    scores = pd.to_numeric(scores, errors="coerce").dropna()
    if not len(scores):
        return
    m, M = float(scores.mean()), float(scores.max())
    if (m > mean_ok) or (M > max_ok):
        print(f"[WARN] Training window not fully clean (mean={m:.2f}, max={M:.2f}). "
              "Continuing with calibration but consider refining the window.")

def _avoid_exact_zeros(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if np.allclose(arr, 0):
        # add tiny noise to break ties; deterministic seed
        rng = np.random.RandomState(42)
        arr = arr + eps * rng.randn(*arr.shape)
    return arr

def run_pipeline(
    input_csv: str,
    output_csv: str,
    timestamp_col: str,
    normal_start: str,
    normal_end: str,
    window: int,
    epochs: int,
    corr_window: int,
    patch: int,
    smoothing_alpha: float,
    make_plots: bool,
    plots_dir: str,
) -> str:
    logger = setup_logger()

    # Build bundle & sanity-check training span
    dp = DataProcessor(timestamp_col=timestamp_col, logger=logger)
    bundle = dp.build(input_csv, normal_start, normal_end)
    _assert_train_span(bundle.df[timestamp_col], normal_start, normal_end, min_hours=72)

    # Scale data
    z, x, _ = dp.scale(bundle)
    T, F = z.shape

    # --- Experts ---
    raws = []
    feats = []

    # 1) Patch forecaster
    r_pf, c_pf = expert_patch_forecaster(
        z, bundle.normal_mask, window=window, epochs=epochs, patch=patch, logger=logger
    )
    r_pf = _avoid_exact_zeros(r_pf)
    raws.append(r_pf); feats.append(c_pf)

    # 2) PCA
    r_pca, c_pca = expert_pca(z, bundle.normal_mask, logger=logger)
    r_pca = _avoid_exact_zeros(r_pca)
    raws.append(r_pca); feats.append(c_pca)

    # 3) Mahalanobis (robust)
    try:
        r_m, c_m = expert_mahalanobis(z, bundle.normal_mask, logger=logger)
        r_m = _avoid_exact_zeros(r_m)
        raws.append(r_m); feats.append(c_m)
    except Exception as e:
        logger.warning("Mahalanobis expert skipped: %s", e)

    # 4) Isolation Forest
    try:
        r_if, c_if = expert_isoforest(z, bundle.normal_mask, logger=logger)
        r_if = _avoid_exact_zeros(r_if)
        raws.append(r_if); feats.append(c_if)
    except Exception as e:
        logger.warning("IsolationForest expert skipped: %s", e)

    # 5) Correlation-shift
    r_cs, c_cs = expert_corrshift(z, bundle.normal_mask, window=corr_window, logger=logger)
    r_cs = _avoid_exact_zeros(r_cs)
    raws.append(r_cs); feats.append(c_cs)

    # --- Ensemble & calibration ---
    raw, feat = combine_experts(raws, feats, weights=None)
    med, mad = robust_calibration_params(raw[bundle.normal_mask])
    alpha = alpha_for_q99(raw[bundle.normal_mask], med, mad)
    score = raw_to_score(raw, med, mad, alpha)
    score = smooth_scores(score, alpha=smoothing_alpha)

    # --- Top-7 attribution ---
    feature_names = bundle.feature_cols
    topk = [top_contributors_from_errors(feat[i], feature_names, k=7) for i in range(T)]
    topk_arr = np.array(topk, dtype=object)

    # --- Output ---
    out_df = bundle.df.copy()
    out_df["Abnormality_score"] = score
    for j in range(7):
        out_df[f"top_feature_{j+1}"] = topk_arr[:, j]

    # Warn if training window looks "hot" after calibration
    t = pd.to_datetime(out_df[timestamp_col])
    train_mask = (t >= pd.to_datetime(normal_start)) & (t <= pd.to_datetime(normal_end))
    _warn_if_train_anomalous(out_df.loc[train_mask, "Abnormality_score"])

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    # Small JSON report
    report = {
        "n_rows": int(T),
        "n_features": int(F),
        "normal_rows": int(bundle.normal_mask.sum()),
        "calibration": {"median": float(med), "mad": float(mad), "alpha": float(alpha)},
        "model_used": "cemtad+",
        "window": window,
        "epochs": epochs,
        "corr_window": corr_window,
        "patch": patch,
    }
    with open(output_csv.replace(".csv", "_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Optional plots
    if make_plots:
        os.makedirs(plots_dir, exist_ok=True)
        plot_scores_time(out_df, timestamp_col, os.path.join(plots_dir, "scores_over_time.png"), normal_start, normal_end)
        plot_score_histogram(out_df, timestamp_col, normal_start, normal_end, os.path.join(plots_dir, "score_hist_train_vs_rest.png"))
        idx = int(out_df["Abnormality_score"].idxmax())
        plot_top_features_bar(out_df, idx, os.path.join(plots_dir, f"top_features_at_{idx}.png"))

    logger.info("Wrote: %s", output_csv)
    return output_csv

def parse_args():
    p = argparse.ArgumentParser(description='CEM-TAD+ ensemble for multivariate TS anomaly detection')
    p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--timestamp_col', default='Time')
    p.add_argument('--normal_start', default='2004-01-01 00:00'); p.add_argument('--normal_end', default='2004-01-05 23:59')
    p.add_argument('--window', type=int, default=60); p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--corr_window', type=int, default=120); p.add_argument('--patch', type=int, default=6)
    p.add_argument('--smoothing_alpha', type=float, default=0.15)
    p.add_argument('--make_plots', action='store_true'); p.add_argument('--plots_dir', default='results/figs')
    return p.parse_args()
if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args.input, args.output, args.timestamp_col, args.normal_start, args.normal_end,
                 args.window, args.epochs, args.corr_window, args.patch, args.smoothing_alpha,
                 args.make_plots, args.plots_dir)
