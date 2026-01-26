"""
solution.py  – entry‑point for the evaluation server
"""

from pathlib import Path
from typing  import List, Tuple
import json, joblib, numpy as np

# ------------------------------------------------------------------ #
#  re‑use project helpers                                            #
# ------------------------------------------------------------------ #
from preprocessing        import raw_to_uniform, add_per_sample_features
from feature_engineering  import make_feature_table
from postprocessing       import proba_to_intervals

# ------------------------------------------------------------------ #
# 1.  locate *latest* model + its JSON                               #
# ------------------------------------------------------------------ #
BASE_DIR   = Path(__file__).resolve().parent
MODELDIR   = BASE_DIR / "saved_models"
final_jobs = sorted(MODELDIR.glob("train_stop_detector_final_*.joblib"))
if not final_jobs:
    raise FileNotFoundError("no saved_models/train_stop_detector_final_*.joblib found")

MODEL_PATH = final_jobs[-1]                    # newest index
CFG_PATH   = MODEL_PATH.with_suffix(".json")

# ------------------------------------------------------------------ #
# 2.  fixed best feature parameters  (must match training)           #
# ------------------------------------------------------------------ #
MODEL     = joblib.load(MODEL_PATH)
CFG  = json.load(open(CFG_PATH))
POST_CFG = CFG["post_cfg"]
FEAT_CFG = CFG["feat_cfg"]


# FEAT_CFG = dict(
#     lag_steps   = list(range(1, 21)),   # 1‑20‑second lags
#     add_diff    = True,
#     diff_steps  = (1,),
#     add_extra   = False,
#     window_len  = "1s",
# )

# ------------------------------------------------------------------ #
# 3.  main callable                                                  #
# ------------------------------------------------------------------ #
def calculate_stop_intervals(
        magnetometer:  List[Tuple[float,float,float,float]],
        accelerometer: List[Tuple[float,float,float,float]],
        gravity:       List[Tuple[float,float,float,float]],
        gyroscope:     List[Tuple[float,float,float,float]],
) -> List[Tuple[float, float]]:
    """
    Return a list of (start_s, end_s) intervals (seconds since file start)
    during which the train is stopped.
    """

    # ------- raw streams → uniform DataFrame ------------------------
    df_u = raw_to_uniform({
        "magnetometer":  np.asarray(magnetometer),
        "accelerometer": np.asarray(accelerometer),
        "gravity":       np.asarray(gravity),
        "gyroscope":     np.asarray(gyroscope),
    })
    df_u = add_per_sample_features(df_u)

    # ------- 1‑s window feature table (matches training) -----------
    df_f = make_feature_table(df_u, **FEAT_CFG)

    # ------- model inference ---------------------------------------
    proba = MODEL.predict_proba(df_f)[:, 1]

    # ------- probability → stop intervals --------------------------
    intervals = proba_to_intervals(
        proba,
        df_f.index,
        df_f.index[0],
        **POST_CFG
    )

    return intervals
