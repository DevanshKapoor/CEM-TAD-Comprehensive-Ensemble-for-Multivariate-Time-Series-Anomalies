from __future__ import annotations

import logging

import numpy as np


def setup_logger(name: str = "cemtad", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s - %(levelname)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ewma(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    if len(x) == 0:
        return x
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def choose_device(requested: str = "auto") -> str:
    try:
        import torch

        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return requested
    except Exception:
        return "cpu"
