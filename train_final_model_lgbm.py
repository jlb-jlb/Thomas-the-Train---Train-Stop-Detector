# train_final_model.py   (uses save_model helper)

import argparse, json, joblib
from pathlib import Path
import numpy as np
from sklearn.pipeline   import Pipeline
from sklearn.ensemble   import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier 
from optuna.integration import LightGBMPruningCallback
import optuna, lightgbm as lgb


from sklearn.metrics    import f1_score
from sklearn.model_selection import GroupShuffleSplit

from preprocessing       import raw_to_uniform, add_per_sample_features
from feature_engineering import make_feature_table
from processing.quantile_clipper import QuantileClipper
from postprocessing      import proba_to_intervals
from train_model         import load_train_data, build_corpus, save_model
from log_experiments     import log_experiment



# -------- fixed best hyper‑parameters ---------------------------------

from best_params import BEST_FEATURE_CFG, HGB_PARAMS, BEST_POSTPROC


# SUPER IMPORTANT THE PARAMS!
######################################################################

# ----------------------------------------------------------------------
def fit_pipeline(X, y):
    """
    Same preprocessing (QuantileClipper) but with a LightGBM backend.
    GPU: add  device_type="gpu", gpu_platform_id=0, gpu_device_id=0
    """
    pipe = Pipeline([
        ("clip", QuantileClipper(q_lower=1, q_upper=99)),
about:blank#blocked        ("lgbm", LGBMClassifier(
                    n_estimators   = 1200,
                    num_leaves     = 63,      # similar capacity to depth‑10 tree
                    learning_rate  = 0.03,
                    max_depth      = -1,      # unrestricted
                    subsample      = 0.8,
                    colsample_bytree = 0.8,
                    objective      = "binary",
                    class_weight   = "balanced",
                    random_state   = 42,
                    n_jobs         = -1,      # use all CPU cores
                ))
    ])
    pipe.fit(X, y)
    return pipe

def objective(trial):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
        "num_leaves":    trial.suggest_int("leaves", 31, 255, step=32),
        "feature_fraction": trial.suggest_float("colsample", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("subsample", 0.6, 1.0),
        "bagging_freq": 1,
        "class_weight": "balanced",
        "verbosity": -1,
        "seed": 42,
    }
    gbm = lgb.train(
        params,
        lgb.Dataset(X_tr, y_tr),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(X_va, y_va)],
        valid_names=["val"],
        callbacks=[LightGBMPruningCallback(trial, "binary_logloss")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    pred = gbm.predict(X_va)
    return f1_score(y_va, (pred>=0.5).astype(int))

# ----------------------------------------------------------------------
def main(args):
    trips = load_train_data(Path(args.data_dir))
    print(f"Loaded {len(trips)} recordings")

    X_all, y_all, groups = build_corpus(trips, **BEST_FEATURE_CFG)

    # ---- hold‑out split for metrics ---------------------------------
    gss = GroupShuffleSplit(test_size=0.2, random_state=4)
    tr_idx, va_idx = next(gss.split(X_all, y_all, groups))
    X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
    y_tr, y_va = y_all[tr_idx],     y_all[va_idx]

    pipe = fit_pipeline(X_tr, y_tr)
    y_pred = pipe.predict(X_va)
    val_f1 = f1_score(y_va, y_pred)
    print(f"Validation F1 = {val_f1:.3f}")

    # ---- save validation model (optional) ---------------------------
    val_path = save_model(pipe, args.model_dir, "train_stop_detector_val")

    # ---- refit on ALL data & save FINAL model -----------------------
    pipe.fit(X_all, y_all)
    final_path = save_model(pipe, args.model_dir, "train_stop_detector_final")

    # corresponding post‑proc JSON (same index as model filename)
    cfg_path = final_path.with_suffix(".json")
    combined_cfg = {
        "post_cfg": BEST_POSTPROC,
        "feat_cfg": BEST_FEATURE_CFG
    }
    json.dump(combined_cfg, open(cfg_path, "w"), indent=2)
    print("✓  Saved combined configs →", cfg_path)

    # ---- log experiment --------------------------------------------
    additional = dict(
        feat_cfg   = BEST_FEATURE_CFG,
        hgb_params = HGB_PARAMS,
        postproc   = BEST_POSTPROC,
        val_f1     = round(val_f1, 3),
    )

    log_experiment(
        predictions      = y_pred,
        ground_truth     = y_va,
        features         = list(X_all.columns),
        additional       = additional,
        model_path_val   = val_path,
        model_path_full  = final_path,
        results_dir      = "results",
        img_dir          = "results/img"
    )

    print("All done!  Artefacts stored in", final_path.parent)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",  default="data/train")
    ap.add_argument("--model_dir", default="saved_models")
    args = ap.parse_args()
    main(args)
