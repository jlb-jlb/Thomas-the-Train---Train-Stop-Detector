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


def trip_to_features(npz, group_id, lag_steps=list(range(1, 21))):
    df_u  = raw_to_uniform({
               "magnetometer":  npz["magnetometer"],
               "accelerometer": npz["accelerometer"],
               "gravity":       npz["gravity"],
               "gyroscope":     npz["gyroscope"],
           })
    df_u  = add_per_sample_features(df_u)
    df_f  = make_feature_table(df_u, lag_steps=lag_steps)

    # labels
    y = np.zeros(len(df_f), dtype=int)
    t_c = (df_f.index - df_f.index[0]).total_seconds()
    for s,e in npz["intervals"]:
        y[(t_c>=s)&(t_c<=e)] = 1

    return df_f, y, np.repeat(group_id, len(df_f))



def build_corpus(trips):
    X_list, y_list, g_list = [], [], []
    for trip in trips:
        Xf, yf, gf = trip_to_features(trip["data"], trip["group"])
        X_list.append(Xf);  y_list.append(yf);  g_list.append(gf)
    return (pd.concat(X_list, axis=0),
            np.concatenate(y_list),
            np.concatenate(g_list))


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
    trips = load_train_data(Path(args.data_dir))
    print(f"Found {len(trips)} recordings")

    # --------------- build full corpus -----------------
    X_all, y_all, groups = build_corpus(trips)
    print("Feature matrix:", X_all.shape, "  positive rate:", y_all.mean().round(3))

    # --------------- train / val split -----------------
    gss = GroupShuffleSplit(test_size=args.test_size, random_state=args.seed)
    tr_idx, va_idx = next(gss.split(X_all, y_all, groups))
    X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
    y_tr, y_va = y_all[tr_idx],     y_all[va_idx]

    pipe = fit_pipeline(X_tr, y_tr)
    y_pred = pipe.predict(X_va)
    print(f"Validation F1 on unseen trips: {f1_score(y_va, y_pred):.3f}")

    
    val_path = save_model(pipe, "saved_models", "train_stop_detector_val")

    print("✓  Validation model saved →", val_path)

    # --------------- optional per‑trip demo ------------
    if args.demo:
        from postprocessing import POSTPROC_CFG

        for g in np.unique(groups[va_idx])[:3]:
            mask  = groups[va_idx]==g
            proba = pipe.predict_proba(X_all.iloc[va_idx][mask])[:,1]
            ints  = proba_to_intervals(
                        proba, X_all.iloc[va_idx][mask].index,
                        X_all.iloc[va_idx][mask].index[0],
                        **{k:v for k,v in POSTPROC_CFG.items() if k!="thr"})
            print(g, ints[:3], "…")

    # --------------- final model on ALL data ----------
    pipe_final = fit_pipeline(X_all, y_all)
    final_path = save_model(pipe_final, "saved_models", "train_stop_detector")

    #  LOG THE EXPERIMENT
    log_experiment(y_pred, y_va, model_path_val=val_path, model_path_full=final_path,
                   results_dir="results", img_dir="results/img")

    print("All done!")
    return val_path, final_path



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/train")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int,   default=4)
    ap.add_argument("--demo",    action="store_true",
                    help="print first 3 predicted intervals of 3 val trips")
    args = ap.parse_args()
    main(args)

