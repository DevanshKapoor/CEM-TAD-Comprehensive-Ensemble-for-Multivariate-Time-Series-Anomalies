from __future__ import annotations

import numpy as np


def top_contributors_from_errors(
    per_feature_error: np.ndarray, feature_names: list[str], k: int = 7
) -> list[str]:
    err = np.asarray(per_feature_error, dtype=float)
    err = np.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)
    err = np.maximum(err, 0.0)
    total = float(err.sum())
    if total <= 0.0:
        return [""] * k
    contrib = err / (total + 1e-12)
    items: list[tuple[float, str]] = []
    for i, p in enumerate(contrib):
        if p > 0.01:
            items.append((float(p), feature_names[i]))
    if not items:
        return [""] * k
    items.sort(key=lambda x: (-x[0], x[1]))
    names = [name for _, name in items[:k]]
    while len(names) < k:
        names.append("")
    return names
