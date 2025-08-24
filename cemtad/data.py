from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import setup_logger


@dataclass
class DataBundle:
    df: pd.DataFrame
    timestamps: pd.Series
    feature_cols: list[str]
    numeric_df: pd.DataFrame
    normal_mask: np.ndarray


class RobustScalerLite:
    def __init__(self) -> None:
        self.median_ = None
        self.scale_ = None

    def fit(self, x: np.ndarray) -> RobustScalerLite:
        med = np.nanmedian(x, axis=0)
        q75 = np.nanpercentile(x, 75, axis=0)
        q25 = np.nanpercentile(x, 25, axis=0)
        iqr = q75 - q25
        std = np.nanstd(x, axis=0)
        scale = np.where(iqr > 1e-12, iqr, np.where(std > 1e-12, std, 1.0))
        self.median_ = med
        self.scale_ = scale
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.median_) / (self.scale_ + 1e-12)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


class DataProcessor:
    def __init__(self, timestamp_col: str, logger=None) -> None:
        self.timestamp_col = timestamp_col
        self.logger = logger or setup_logger()
        self.scaler = RobustScalerLite()

    def load(self, path: str) -> pd.DataFrame:
        self.logger.info("Loading CSV: %s", path)
        df = pd.read_csv(path)
        if self.timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{self.timestamp_col}' not found.")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.timestamp_col] = pd.to_datetime(
            df[self.timestamp_col], errors="coerce", infer_datetime_format=True
        )
        if df[self.timestamp_col].isna().any():
            raise ValueError("Failed to parse timestamps; please check the timestamp column.")
        df = df.dropna(axis=1, how="all")
        for c in df.columns:
            if c == self.timestamp_col:
                continue
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.set_index(self.timestamp_col).sort_index()
        inferred = pd.infer_freq(df.index)
        if inferred != "T":
            df = df.resample("T").mean()
        method = "time" if isinstance(df.index, pd.DatetimeIndex) else "linear"
        df = df.ffill().bfill().interpolate(method=method, limit_direction="both")
        return df.reset_index()

    def detect_normal_mask(
        self,
        ts: pd.Series,
        df: pd.DataFrame,
        normal_start: str | None,
        normal_end: str | None,
    ) -> np.ndarray:
        for col in df.columns:
            low = col.lower()
            if low in {"label", "is_anomaly", "anomaly", "is_normal", "split"}:
                vals = df[col].astype(str).str.lower()
                normal_mask = vals.isin({"0", "false", "normal", "train", "training"})
                if normal_mask.sum() > 0:
                    self.logger.info("Using training mask from column '%s'.", col)
                    return normal_mask.values.astype(bool)
        if normal_start and normal_end:
            ns = pd.to_datetime(normal_start)
            ne = pd.to_datetime(normal_end)
            return ((ts >= ns) & (ts <= ne)).values
        cutoff = ts.iloc[0] + pd.Timedelta(hours=120)
        self.logger.warning("No normal window provided; defaulting to first 120 hours.")
        return (ts <= cutoff).values

    def build(
        self, path: str, normal_start: str | None, normal_end: str | None
    ) -> DataBundle:
        raw = self.load(path)
        df = self.clean(raw)
        ts = pd.to_datetime(df[self.timestamp_col])
        feature_cols = [
            c
            for c in df.columns
            if c != self.timestamp_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        numeric_df = df[feature_cols]
        normal_mask = self.detect_normal_mask(ts, df, normal_start, normal_end)
        std = numeric_df[normal_mask].std(axis=0, ddof=0)
        keep = std > 1e-12
        feature_cols = [c for c, k in zip(feature_cols, keep, strict=False) if k]
        numeric_df = numeric_df[feature_cols]
        return DataBundle(
            df=df,
            timestamps=ts,
            feature_cols=feature_cols,
            numeric_df=numeric_df,
            normal_mask=normal_mask,
        )

    def scale(self, bundle: DataBundle):
        x = bundle.numeric_df.values.astype(float)
        self.scaler.fit(x[bundle.normal_mask])
        z_all = (x - self.scaler.median_) / (self.scaler.scale_ + 1e-12)
        return z_all, x, self.scaler
