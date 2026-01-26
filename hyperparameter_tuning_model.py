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
import json
from itertools import product
from sklearn.model_selection import GroupKFold

# custom imports
from preprocessing import raw_to_uniform, add_per_sample_features
from feature_engineering import make_feature_table, list_base_columns, columns_for_cfg
from postprocessing import proba_to_intervals, mean_interval_iou
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


def tune_postproc(pipe, X_va, y_va, gt_intervals_va, pp_grid):
    best_pp, best_f1 = None, 0
    from itertools import product
    keys, values = zip(*pp_grid.items())
    for combo in product(*values):
        cfg = dict(zip(keys, combo), win_len=1.0)   # keep win_len fixed
        proba = pipe.predict_proba(X_va)[:,1]
        pred_int = proba_to_intervals(
            proba, X_va.index, X_va.index[0], **cfg)
        f1 = mean_interval_iou(pred_int, gt_intervals_va)
        if f1 > best_f1:
            best_f1, best_pp = f1, cfg
    return best_pp, best_f1


def main(args):

    # Hyperparameters TUNING
    FEAT_GRID = [
        # 20 lags
        # dict(lag_steps=list(range(1,21)), add_diff=True,  add_extra=True),
        # dict(lag_steps=list(range(1,21)), add_diff=False,  add_extra=True),
        dict(lag_steps=list(range(1,21)), diff_steps=list(range(1,21)), add_diff=True,  add_extra=False),
        dict(lag_steps=list(range(1,21)), diff_steps=list(range(1,15)), add_diff=True,  add_extra=False),
        dict(lag_steps=list(range(1,21)), diff_steps=list(range(1,11)), add_diff=True,  add_extra=False),
        dict(lag_steps=list(range(1,21)), diff_steps=list(range(1,10)), add_diff=True,  add_extra=False),
        dict(lag_steps=list(range(1,21)), diff_steps=list(range(1,6)), add_diff=True,  add_extra=False),
        dict(lag_steps=list(range(1,21)), diff_steps=list(range(1,5)), add_diff=True,  add_extra=False),
        dict(lag_steps=list(range(1,21)), diff_steps=list(range(1,4)), add_diff=True,  add_extra=False),
        dict(lag_steps=list(range(1,21)), diff_steps=list(range(1,3)), add_diff=True,  add_extra=False),
        dict(lag_steps=list(range(1,21)), diff_steps=list(range(1,2)), add_diff=True,  add_extra=False),
        dict(lag_steps=list(range(1,21)), add_diff=False,  add_extra=False),
        # 15 lags
        dict(lag_steps=list(range(1,16)), add_diff=True,  add_extra=False),
        dict(lag_steps=list(range(1,16)), add_diff=True,  add_extra=True),
        dict(lag_steps=list(range(1,16)), add_diff=False, add_extra=False),
        dict(lag_steps=list(range(1,16)), add_diff=False, add_extra=True),
        # 10 lags
        dict(lag_steps=list(range(1,11)), add_diff=True,  add_extra=False),
        dict(lag_steps=list(range(1,11)), add_diff=True,  add_extra=True),
        dict(lag_steps=list(range(1,11)), add_diff=False, add_extra=False),
        dict(lag_steps=list(range(1,11)), add_diff=False, add_extra=True),
        # 5 lags
        dict(lag_steps=list(range(1,6)), add_diff=True,  add_extra=False),
        dict(lag_steps=list(range(1,6)), add_diff=True,  add_extra=True),
        dict(lag_steps=list(range(1,6)), add_diff=False, add_extra=False),
        dict(lag_steps=list(range(1,6)), add_diff=False, add_extra=True),
        # 3 lags
        dict(lag_steps=[1,2,3], add_diff=True,  add_extra=False),
        dict(lag_steps=[1,2,3], add_diff=True,  add_extra=True),
        dict(lag_steps=[1,2,3], add_diff=False, add_extra=False),
        dict(lag_steps=[1,2,3], add_diff=False, add_extra=True),
        # 2 lags
        dict(lag_steps=[1,2], add_diff=True,  add_extra=False),
        dict(lag_steps=[1,2], add_diff=True,  add_extra=True),
        dict(lag_steps=[1,2], add_diff=False, add_extra=False),
        dict(lag_steps=[1,2], add_diff=False, add_extra=True),
    ]

    HGB_PARAMS = [
        # dict(max_depth=d, learning_rate=lr)
        # for d in [10, 20, 30, 50, 100]
        # for lr in [0.07,0.10,0.12,0.15,0.2,0.25]
        dict(max_depth=d, learning_rate=lr)
        for d in [10]
        for lr in [0.15]
        
    ]

    PP_GRID = {
        "thr":       [0.5,0.55,0.60,0.65,0.70,0.75,0.8,0.9],
        "min_stop":  list(range(1,13)),
        "merge_gap": list(range(2,21)),
        "med_k":     [3,5,7,9,11,13,15,17,19],
    }

    # --------------- load training data ---------------
    trips = load_train_data(Path(args.data_dir))
    print(f"Found {len(trips)} recordings")

    SUP_CFG   = dict(lag_steps=list(range(1,21)), diff_steps=list(range(1,21)), add_diff=True, add_extra=True)
    X_sup, y_all, groups = build_corpus(trips, **SUP_CFG)
    base_cols = list_base_columns()        # 8 core column names
    print("Base columns:", base_cols)
    print("Superset feature matrix:", X_sup.shape)


    #  GRID SEARCH OVER FEATURES +  HGBT Params
    best_f1, best_combo, best_pipe = 0, None, None
    cv = GroupKFold(n_splits=5)
    print("Number of parameter combinations:", len(list(product(FEAT_GRID, HGB_PARAMS))))
    for idx, (feat_cfg, hgb_cfg) in enumerate(product(FEAT_GRID, HGB_PARAMS)):
        cols   = columns_for_cfg(feat_cfg, base_cols)
        X_all  = X_sup[cols]          # fast column‑subset view
        scores = []
        for tr, va in cv.split(X_all, y_all, groups):
            pipe = Pipeline([
                ("clip", QuantileClipper(q_lower=1, q_upper=99)),
                ("clf",  HistGradientBoostingClassifier(
                            class_weight="balanced", early_stopping=True,
                            random_state=42, **hgb_cfg))
            ])
            pipe.fit(X_all.iloc[tr], y_all[tr])
            scores.append(f1_score(y_all[va], pipe.predict(X_all.iloc[va])))
        mean_f1 = np.mean(scores)
        print(f"{feat_cfg} {hgb_cfg}  →  F1={mean_f1:.3f}")
        if mean_f1 > best_f1:
            best_f1, best_combo = mean_f1, (feat_cfg, hgb_cfg)
            best_pipe = pipe         # keep last fitted model on this combo

    print(f"{idx} BEST Stage A F1={best_f1:.3f}\n  feat={best_combo[0]}\n  hgb={best_combo[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a stop detector")
    parser.add_argument("--data_dir", type=str, default="data/train",
                        help="Directory containing training data")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of data to use for testing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)