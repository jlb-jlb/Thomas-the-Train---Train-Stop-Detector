# model.py
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["PIPE", "PIPE_NEWEST", "QuantileClipper"]

PIPE = joblib.load(
    Path(__file__).resolve().parent / "saved_models" / "train_stop_detector_13.joblib"
)

PIPE_NEWEST = joblib.load(
    max(
        (Path(__file__).resolve().parent / "saved_models").glob("*.joblib"),
        key=lambda p: p.stat().st_mtime,
    )
)


