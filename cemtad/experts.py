from __future__ import annotations

import numpy as np

from .utils import setup_logger


class PCABaseline:
    def __init__(self, variance_keep: float = 0.95, logger=None) -> None:
        self.variance_keep = variance_keep
        self.logger = logger or setup_logger()
        self.mean_ = None
        self.components_ = None

    def fit(self, train_z: np.ndarray):
        mu = train_z.mean(axis=0, keepdims=True)
        Xc = train_z - mu
        C = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
        vals, vecs = np.linalg.eigh(C)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        cum = np.cumsum(vals) / (vals.sum() + 1e-12)
        k = int(np.searchsorted(cum, self.variance_keep) + 1)
        k = max(1, min(k, vecs.shape[1]))
        self.mean_ = mu
        self.components_ = vecs[:, :k]
        self.logger.info("PCA: k=%d, var≈%.1f%%", k, 100 * cum[k - 1])
        return self

    def reconstruct(self, z: np.ndarray) -> np.ndarray:
        Xc = z - self.mean_
        W = self.components_
        z_low = Xc @ W
        z_hat = self.mean_ + z_low @ W.T
        return z_hat

    def raw_scores(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z_hat = self.reconstruct(z)
        feat_err = (z - z_hat) ** 2
        raw = feat_err.sum(axis=1) + 1e-12
        return raw, feat_err


def expert_pca(z: np.ndarray, train_mask: np.ndarray, logger=None) -> tuple[np.ndarray, np.ndarray]:
    logger = logger or setup_logger()
    model = PCABaseline(variance_keep=0.95, logger=logger).fit(z[train_mask])
    return model.raw_scores(z)


def expert_mahalanobis(
    z: np.ndarray, train_mask: np.ndarray, logger=None
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.covariance import LedoitWolf

    logger = logger or setup_logger()
    z_train = z[train_mask]
    lw = LedoitWolf().fit(z_train)
    mu = lw.location_
    cov = lw.covariance_
    try:
        invcov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        invcov = np.linalg.pinv(cov)
    delta = z - mu
    raw = np.einsum("ij,jk,ik->i", delta, invcov, delta)
    raw = np.maximum(raw, 0.0) + 1e-12
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    contrib = (delta / (std + 1e-12)) ** 2
    s = contrib.sum(axis=1, keepdims=True) + 1e-12
    contrib = contrib * (raw[:, None] / s)
    return raw, contrib


def expert_isoforest(
    z: np.ndarray, train_mask: np.ndarray, logger=None
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.ensemble import IsolationForest

    logger = logger or setup_logger()
    z_train = z[train_mask]
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=7, n_jobs=-1)
    iso.fit(z_train)
    dfun = iso.decision_function(z)
    raw = (dfun.max() - dfun) + 1e-12
    med = np.median(z_train, axis=0)
    iqr = np.percentile(z_train, 75, axis=0) - np.percentile(z_train, 25, axis=0)
    scale = np.where(iqr > 1e-12, iqr, np.std(z_train, axis=0) + 1e-12)
    contrib = ((z - med) / (scale + 1e-12)) ** 2
    s = contrib.sum(axis=1, keepdims=True) + 1e-12
    contrib = contrib * (raw[:, None] / s)
    return raw, contrib


def rolling_corr_shift(z: np.ndarray, base_corr: np.ndarray, window: int):
    T, F = z.shape
    scores = np.zeros(T, dtype=float)
    contrib = np.zeros((T, F), dtype=float)
    if window < 5 or T < window:
        return scores + 1e-12, contrib
    for t in range(window - 1, T):
        seg = z[t - window + 1 : t + 1]
        C = np.corrcoef(seg, rowvar=False)
        C = np.nan_to_num(C, nan=0.0)
        D = np.abs(C - base_corr)
        np.fill_diagonal(D, 0.0)
        scores[t] = float(np.sum(D))
        contrib[t] = D.sum(axis=1)
    scores[: window - 1] = scores[window - 1]
    contrib[: window - 1] = contrib[window - 1]
    return scores + 1e-12, contrib


def expert_corrshift(z: np.ndarray, train_mask: np.ndarray, window: int = 120, logger=None):
    logger = logger or setup_logger()
    base = np.corrcoef(z[train_mask], rowvar=False)
    base = np.nan_to_num(base, nan=0.0)
    return rolling_corr_shift(z, base, window=window)


def expert_patch_forecaster(
    z: np.ndarray,
    train_mask: np.ndarray,
    window: int = 60,
    epochs: int = 20,
    patch: int = 6,
    logger=None,
):
    logger = logger or setup_logger()
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception:
        T, F = z.shape
        raw = np.zeros(T)
        contrib = np.zeros((T, F))
        for t in range(1, T):
            e = (z[t] - z[t - 1]) ** 2
            raw[t] = e.sum()
            contrib[t] = e
        raw += 1e-12
        return raw, contrib
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T, F = z.shape
    w = window
    idx = np.where(train_mask)[0]
    starts = []
    for end in range(w, len(idx)):
        window_idx = idx[end - w : end]
        if np.all(train_mask[window_idx]) and (window_idx[-1] + 1) < T:
            starts.append(window_idx[0])
    if not starts:
        tr = np.where(train_mask)[0]
        starts = list(range(tr[0], max(tr[0], tr[-1] - w)))
    X = np.stack([z[s0 : s0 + w] for s0 in starts], axis=0)
    Y = np.stack([z[s0 + w] for s0 in starts], axis=0)

    class PatchTST(nn.Module):
        def __init__(
            self, f_in: int, d_model: int = 128, nhead: int = 4, layers: int = 2, patch: int = 6
        ):
            super().__init__()
            self.proj = nn.Conv1d(f_in, d_model, kernel_size=patch, stride=patch, padding=0)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.head = nn.Linear(d_model, f_in)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.transpose(1, 2)
            tokens = self.proj(x).transpose(1, 2)
            h = self.encoder(tokens)
            return self.head(h[:, -1, :])

    model = PatchTST(F, d_model=128, nhead=4, layers=2, patch=patch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.SmoothL1Loss()
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=256, shuffle=True, drop_last=False)
    model.train()
    for ep in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            y = model(xb)
            loss = loss_fn(y, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    raw = np.zeros(T)
    contrib = np.zeros((T, F))
    model.eval()
    with torch.no_grad():
        for t in range(w, T):
            import torch as _torch

            xw = _torch.tensor(z[t - w : t][None, ...], dtype=_torch.float32, device=device)
            y = model(xw)[0].detach().cpu().numpy()
            e = (y - z[t]) ** 2
            raw[t] = float(e.sum())
            contrib[t] = e
    raw[:w] = raw[w]
    contrib[:w] = contrib[w]
    return raw + 1e-12, contrib
