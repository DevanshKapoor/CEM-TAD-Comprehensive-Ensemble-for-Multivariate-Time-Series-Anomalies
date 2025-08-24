from __future__ import annotations

import numpy as np

from .utils import ewma


def robust_calibration_params(train_raw: np.ndarray):
    train_raw = np.asarray(train_raw, dtype=float)
    med = float(np.nanmedian(train_raw))
    mad = float(np.nanmedian(np.abs(train_raw - med))) * 1.4826
    if mad <= 1e-12:
        mad = 1.0
    return med, mad


def alpha_for_q99(train_raw: np.ndarray, med: float, mad: float) -> float:
    r = np.maximum(0.0, (np.asarray(train_raw, dtype=float) - med) / (mad + 1e-12))
    q99 = np.nanpercentile(r, 99.0)
    if not np.isfinite(q99) or q99 <= 1e-12:
        return 3.0
    return float(-np.log(0.8) / q99)


def raw_to_score(raw: np.ndarray, med: float, mad: float, alpha: float) -> np.ndarray:
    r = np.maximum(0.0, (np.asarray(raw, dtype=float) - med) / (mad + 1e-12))
    score = 100.0 * (1.0 - np.exp(-alpha * r))
    return np.clip(score, 0.0, 100.0)


def smooth_scores(score: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    return ewma(np.asarray(score, dtype=float), alpha=alpha)
