import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib


# custom imports
from preprocessing import raw_to_uniform, add_per_sample_features
from feature_engineering import make_feature_table
from postprocessing import proba_to_intervals
from processing.quantile_clipper import QuantileClipper
from log_experiments import log_experiment



################################################



def load_train_data(data_dir: Path):
    trips = []
    for folder in sorted(data_dir.glob("*")):
        if not folder.is_dir(): continue
        npz = folder / "data.npz"
        if not npz.exists(): continue
        group_id = folder.relative_to(data_dir).as_posix()
        trips.append({"data": np.load(npz, allow_pickle=True), "group": group_id})
    return trips



def trip_to_features(npz, group_id, **feat_cfg):
    """
    feat_cfg may contain:
        lag_steps : Sequence[int]
        add_diff  : bool
        diff_steps: Sequence[int]
        add_extra : bool
        window_len: str
    """
    df_u = raw_to_uniform({
        "magnetometer":  npz["magnetometer"],
        "accelerometer": npz["accelerometer"],
        "gravity":       npz["gravity"],
        "gyroscope":     npz["gyroscope"],
    })
    df_u  = add_per_sample_features(df_u)
    df_f  = make_feature_table(df_u, **feat_cfg)   # <‑‑ pass all flags

    # ---------- labels ----------
    y = np.zeros(len(df_f), dtype=int)
    t_c = (df_f.index - df_f.index[0]).total_seconds()
    for s, e in npz["intervals"]:
        y[(t_c >= s) & (t_c <= e)] = 1

    return df_f, y, np.repeat(group_id, len(df_f))


def build_corpus(trips, **feat_cfg):
    X_list, y_list, g_list = [], [], []
    for trip in trips:
        Xf, yf, gf = trip_to_features(trip["data"], trip["group"], **feat_cfg)
        X_list.append(Xf);  y_list.append(yf);  g_list.append(gf)

    X_all = pd.concat(X_list, axis=0)
    y_all = np.concatenate(y_list)
    groups = np.concatenate(g_list)
    return X_all, y_all, groups


def fit_pipeline(X, y):
    pipe = Pipeline([
        ("clip", QuantileClipper(q_lower=1, q_upper=99)),
        ("clf",  HistGradientBoostingClassifier(
                     max_depth=10,
                     learning_rate=0.12,
                     class_weight="balanced",
                     early_stopping=True,
                     random_state=42))
    ])
    pipe.fit(X, y)
    return pipe


def save_model(pipe, model_dir, model_base):
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)
    existing = sorted(model_dir.glob(f"{model_base}_*.joblib"))
    next_idx = (int(existing[-1].stem.split("_")[-1]) + 1) if existing else 1
    path = model_dir / f"{model_base}_{next_idx}.joblib"
    joblib.dump(pipe, path)
    print("✓  Saved model →", path)
    return path




def main(args):
    # best params from hyperparameter tuning
    best_params = {
        'lag_steps': list(range(1, 21)),
        'add_diff': True,
        'add_extra': False,
        'diff_steps': list(range(1,2)),
        'window_len': '1s',
        'hgb__max_depth': 10,
        'hgb__learning_rate': 0.15,
    }

    trips = load_train_data(Path(args.data_dir))
    print(f"Found {len(trips)} recordings")

