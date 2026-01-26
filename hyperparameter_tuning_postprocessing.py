# tune_postproc.py  (verbose version)
import argparse, json, itertools, joblib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline      import Pipeline
from sklearn.ensemble      import HistGradientBoostingClassifier
from sklearn.metrics       import f1_score
from preprocessing         import raw_to_uniform, add_per_sample_features
from feature_engineering   import make_feature_table, list_base_columns, columns_for_cfg
from postprocessing        import proba_to_intervals, mean_interval_iou, interval_f1_at_tau, set_jaccard
from processing.quantile_clipper import QuantileClipper
from train_model           import load_train_data, build_corpus

# ------------------------------------------------------------------ #
BEST_PARAMS = dict(
    lag_steps  = list(range(1,21)),
    add_diff   = True,
    add_extra  = False,
    diff_steps = (1,),
    window_len = "1s",
    hgb__max_depth    = 10,
    hgb__learning_rate= 0.15,
)

PP_GRID = dict(
    thr       = [0.50,0.55,0.60,0.65,0.70,0.75],
    min_stop  = [1,2,3,4,5,6,7,10,11,12],
    merge_gap = [2,3,4,5, 6, 7, 8, 9, 10],
    med_k     = [5,7,9, 11, 13, 15, 17, 19],
    win_len   = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
)

# ------------------------------------------------------------------ #
def fit_classifier(X, y):
    pipe = Pipeline([
        ("clip", QuantileClipper(q_lower=1, q_upper=99)),
        ("clf",  HistGradientBoostingClassifier(
                     max_depth      = BEST_PARAMS["hgb__max_depth"],
                     learning_rate  = BEST_PARAMS["hgb__learning_rate"],
                     class_weight   = "balanced",
                     early_stopping = True,
                     random_state   = 42))
    ])
    pipe.fit(X, y)
    return pipe

# ------------------------------------------------------------------ #
def verbose_tune_postproc(pipe, X_va, gt_intervals_va, grid, score_func=interval_f1_at_tau):
    keys, values = zip(*grid.items())
    combos = list(itertools.product(*values))
    best_cfg, best_iou = None, 0

    print(f"--- Post‑processing sweep ({len(combos)} combos) ---")
    for i, combo in enumerate(combos, 1):
        cfg   = dict(zip(keys, combo))
        proba = pipe.predict_proba(X_va)[:, 1]
        pred  = proba_to_intervals(proba, X_va.index, X_va.index[0], **cfg)
        iou   = score_func(pred, gt_intervals_va)

        mark  = "✓" if iou > best_iou else " "
        print(f"{i:3}/{len(combos)} {mark}  IoU={iou:.3f}  {cfg}")

        if iou > best_iou:
            best_iou, best_cfg = iou, cfg

    print(f"<<< Best IoU={best_iou:.3f}  cfg={best_cfg}")
    return best_cfg, best_iou

# ------------------------------------------------------------------ #
def main(args):
    trips = load_train_data(Path(args.data_dir))
    print(f"Loaded {len(trips)} recordings")

    # -------- single corpus extraction with fixed best feature cfg ---
    feat_cfg = {k: BEST_PARAMS[k] for k in
                ("lag_steps","add_diff","add_extra","diff_steps","window_len")}
    X_all, y_all, groups = build_corpus(trips, **feat_cfg)

    gss = GroupShuffleSplit(test_size=0.2, random_state=4)
    tr, va = next(gss.split(X_all, y_all, groups))
    X_tr, X_va = X_all.iloc[tr], X_all.iloc[va]
    y_tr       = y_all[tr]

    pipe = fit_classifier(X_tr, y_tr)
    print(f"Val F1 (classifier only) = {f1_score(y_all[va], pipe.predict(X_va)):.3f}")

    # ---- build GT interval list for the validation recordings -------
    val_groups = set(np.unique(groups[va]))
    gt_ints_va = [tuple(i)
                  for trip in trips if trip["group"] in val_groups
                  for i in trip["data"]["intervals"]]

    # ---- verbose post‑processing sweep ------------------------------
    best_cfg, best_iou = verbose_tune_postproc(pipe, X_va, gt_ints_va, PP_GRID)

    # ---- refit on ALL data & save artefacts -------------------------
    pipe.fit(X_all, y_all)
    model_dir  = Path(args.model_dir); model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "train_stop_detector_postproc.joblib"
    joblib.dump(pipe, model_path)
    cfg_path   = model_path.with_suffix(".json")
    json.dump(best_cfg, open(cfg_path, "w"))
    print("Saved model →", model_path)
    print("Saved cfg   →", cfg_path)

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",  default="data/train")
    ap.add_argument("--model_dir", default="saved_models")
    args = ap.parse_args()
    main(args)
