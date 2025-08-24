from __future__ import annotations

import numpy as np


def rank_normalize(x: np.ndarray) -> np.ndarray:
    order = x.argsort().argsort().astype(float)
    return (order + 1.0) / (len(x) + 1.0)


def combine_experts(
    raw_list: list[np.ndarray], contrib_list: list[np.ndarray], weights: list[float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if weights is None:
        weights = [1.0] * len(raw_list)
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    ranks = np.stack([rank_normalize(r) for r in raw_list], axis=1)
    rank_comb = (ranks * w[None, :]).sum(axis=1)
    raw = rank_comb - rank_comb.min() + 1e-6
    contrib_norm = []
    for C in contrib_list:
        C = np.maximum(C, 0.0)
        s = C.sum(axis=1, keepdims=True) + 1e-12
        contrib_norm.append(C / s)
    C_stack = np.stack(contrib_norm, axis=2)
    Cw = (C_stack * w[None, None, :]).sum(axis=2)
    s = Cw.sum(axis=1, keepdims=True) + 1e-12
    contrib = Cw * (raw[:, None] / s)
    return raw, contrib
