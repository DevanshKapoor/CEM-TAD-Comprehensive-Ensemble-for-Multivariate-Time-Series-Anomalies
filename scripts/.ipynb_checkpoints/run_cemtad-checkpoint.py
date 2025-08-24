
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from cemtad.attribution import top_contributors_from_errors
from cemtad.data import DataProcessor
from cemtad.ensemble import combine_experts
from cemtad.experts import (
    expert_corrshift,
    expert_isoforest,
    expert_mahalanobis,
    expert_patch_forecaster,
    expert_pca,
)
from cemtad.scoring import alpha_for_q99, raw_to_score, robust_calibration_params, smooth_scores
from cemtad.utils import setup_logger
from cemtad.visualize import plot_score_histogram, plot_scores_time, plot_top_features_bar


def run_pipeline(input_csv: str, output_csv: str, timestamp_col: str, normal_start: str, normal_end: str,
                 window: int, epochs: int, corr_window: int, patch: int, smoothing_alpha: float,
                 make_plots: bool, plots_dir: str) -> str:
    logger = setup_logger()
    dp = DataProcessor(timestamp_col=timestamp_col, logger=logger)
    bundle = dp.build(input_csv, normal_start, normal_end)
    z, x, _ = dp.scale(bundle)
    T, F = z.shape
    raws = []; feats = []
    r_pf, c_pf = expert_patch_forecaster(z, bundle.normal_mask, window=window, epochs=epochs, patch=patch, logger=logger)
    raws.append(r_pf); feats.append(c_pf)
    r_pca, c_pca = expert_pca(z, bundle.normal_mask, logger=logger)
    raws.append(r_pca); feats.append(c_pca)
    try:
        r_m, c_m = expert_mahalanobis(z, bundle.normal_mask, logger=logger)
        raws.append(r_m); feats.append(c_m)
    except Exception as e:
        logger.warning('Mahalanobis expert skipped: %s', e)
    try:
        r_if, c_if = expert_isoforest(z, bundle.normal_mask, logger=logger)
        raws.append(r_if); feats.append(c_if)
    except Exception as e:
        logger.warning('IsolationForest expert skipped: %s', e)
    r_cs, c_cs = expert_corrshift(z, bundle.normal_mask, window=corr_window, logger=logger)
    raws.append(r_cs); feats.append(c_cs)
    raw, feat = combine_experts(raws, feats, weights=None)
    med, mad = robust_calibration_params(raw[bundle.normal_mask])
    alpha = alpha_for_q99(raw[bundle.normal_mask], med, mad)
    score = raw_to_score(raw, med, mad, alpha)
    score = smooth_scores(score, alpha=smoothing_alpha)
    feature_names = bundle.feature_cols
    topk = [top_contributors_from_errors(feat[i], feature_names, k=7) for i in range(T)]
    topk_arr = np.array(topk, dtype=object)
    out_df = bundle.df.copy()
    out_df['Abnormality_score'] = score
    for j in range(7):
        out_df[f'top_feature_{j+1}'] = topk_arr[:, j]
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    report = {'n_rows': int(T), 'n_features': int(F), 'normal_rows': int(bundle.normal_mask.sum()),
              'calibration': {'median': float(med), 'mad': float(mad), 'alpha': float(alpha)},
              'model_used': 'cemtad+', 'window': window, 'epochs': epochs, 'corr_window': corr_window, 'patch': patch}
    with open(output_csv.replace('.csv', '_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    if make_plots:
        os.makedirs(plots_dir, exist_ok=True)
        plot_scores_time(out_df, timestamp_col, os.path.join(plots_dir, 'scores_over_time.png'), normal_start, normal_end)
        plot_score_histogram(out_df, timestamp_col, normal_start, normal_end, os.path.join(plots_dir, 'score_hist_train_vs_rest.png'))
        idx = int(out_df['Abnormality_score'].idxmax())
        plot_top_features_bar(out_df, idx, os.path.join(plots_dir, f'top_features_at_{idx}.png'))
    logger.info('Wrote: %s', output_csv)
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
