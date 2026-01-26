import joblib
import pandas as pd
import numpy as np

# from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Tuple
from pathlib import Path

# from processing.quantile_clipper import QuantileClipper

THIS_DIR = Path(__file__).resolve().parent  # folder that houses solution.py
MODEL_PATH = THIS_DIR / "saved_models" / "train_stop_detector_9.joblib"
# MODEL_PATH = "/home/user/saved_models/train_stop_detector_9.joblib"


def to_df(data):
    sensors = ["magnetometer", "accelerometer", "gravity", "gyroscope"]
    columns = {
        "magnetometer": ["timestamp", "magne_x", "magne_y", "magne_z"],
        "accelerometer": ["timestamp", "accel_x", "accel_y", "accel_z"],
        "gravity": ["timestamp", "gravi_x", "gravi_y", "gravi_z"],
        "gyroscope": ["timestamp", "gyros_x", "gyros_y", "gyros_z"],
    }
    dfs = [
        pd.DataFrame(data[sensor], columns=columns[sensor])
        .assign(timestamp=lambda df: pd.to_datetime(df["timestamp"], unit="s"))
        .set_index("timestamp")
        for sensor in sensors
    ]
    df = pd.concat(dfs, axis=1)

    # Convert the Index to last year
    df.index = df.index - pd.Timestamp("1970-01-01") + pd.Timestamp("2024-01-01")
    return df


def resample_and_interpolate(df, freq="10ms", method="linear"):
    df_uniform = df.copy()
    df_uniform = df_uniform.resample(freq).mean().interpolate(method=method)
    return df_uniform


def create_features(df_uniform):
    acc = df_uniform[["accel_x", "accel_y", "accel_z"]].values
    gra = df_uniform[["gravi_x", "gravi_y", "gravi_z"]].values
    gyr = df_uniform[["gyros_x", "gyros_y", "gyros_z"]].values
    df_uniform["dyn_acc"] = np.linalg.norm(acc - gra, axis=1)
    df_uniform["gyro_mag"] = np.linalg.norm(gyr, axis=1)
    df_uniform["mag_mag"] = np.linalg.norm(
        df_uniform[["magne_x", "magne_y", "magne_z"]].values, axis=1
    )
    return df_uniform


import numpy as np
from scipy.signal import medfilt  # median filter


def proba_to_intervals(
    proba,  # 1‑D array, same length as time_index
    time_index,  # X_val.index  (DateTimeIndex)
    recording_start,  # df_feat.index[0]  (Timestamp)
    thr=0.60,
    win_len=1.0,  # seconds
    min_stop=1.0,
    merge_gap=15.0,
    med_k=7,
    return_datetime=False,
):  # if True, also return Timestamp pairs
    """
    Convert window‑level probabilities to stop intervals.

    Returns
    -------
    list[(start_s, end_s)]                      if return_datetime == False
    list[(start_s, end_s, start_dt, end_dt)]    if return_datetime == True
        where *_s  are seconds since `recording_start`
              *_dt are pandas.Timestamp
    """
    # ------------------------------------------------------------------
    # 0.  seconds since the *trip start*
    # ------------------------------------------------------------------
    t_c_abs = (time_index - recording_start).total_seconds().values
    file_len = t_c_abs[-1] + win_len / 2  # clip right edge
    # print("file_len", file_len)

    # ------------------------------------------------------------------
    # 1.  binary mask (+ median smoothing)
    # ------------------------------------------------------------------
    mask = (proba >= thr).astype(int)
    if med_k > 1:
        mask = medfilt(mask, kernel_size=med_k)

    # ------------------------------------------------------------------
    # 2.  locate start / end indices where mask flips
    # ------------------------------------------------------------------
    edges = np.diff(mask, prepend=0, append=0)
    start_idx = np.flatnonzero(edges == 1)
    end_idx = np.flatnonzero(edges == -1)

    # ------------------------------------------------------------------
    # 3.  build raw intervals, clip to [0, file_len], drop short ones
    # ------------------------------------------------------------------
    intervals = []
    for s_i, e_i in zip(start_idx, end_idx):
        t0 = max(0.0, t_c_abs[s_i] - win_len / 2)
        t1 = min(file_len, t_c_abs[e_i - 1] + win_len / 2)
        if t1 - t0 >= min_stop:
            intervals.append([t0, t1])

    # ------------------------------------------------------------------
    # 4.  merge gaps shorter than merge_gap
    # ------------------------------------------------------------------
    merged = []
    for s, e in intervals:
        if not merged:
            merged.append([s, e])
        elif s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    if not return_datetime:
        return [tuple(ix) for ix in merged]

    # also return the wall‑clock times
    merged_dt = [
        (
            s,
            e,
            recording_start + pd.Timedelta(seconds=s),
            recording_start + pd.Timedelta(seconds=e),
        )
        for s, e in merged
    ]
    return merged_dt


def calculate_stop_intervals(
    magnetometer: List[Tuple[float, float, float, float]],
    accelerometer: List[Tuple[float, float, float, float]],
    gravity: List[Tuple[float, float, float, float]],
    gyroscope: List[Tuple[float, float, float, float]],
) -> List[Tuple[float, float]]:
    # each list has tuples with 4 values
    # first value is the time the sample was taken at
    # 2nd-4th value are the x, y, and z components
    #
    # returns a list of intervals at which the train has stopped

    # pipeline the data

    # transform the data into a pandas dataframe
    df = to_df(
        {
            "magnetometer": magnetometer,
            "accelerometer": accelerometer,
            "gravity": gravity,
            "gyroscope": gyroscope,
        }
    )

    # resample the data to a uniform time series
    df_uniform = resample_and_interpolate(df)
    df_uniform = create_features(df_uniform)

    # create windows of 1 second
    win = "1s"  # pandas offset alias: 1‑second

    agg = {
        "dyn_acc": ["mean", "std", "max", "median"],
        "gyro_mag": ["mean", "std", "max"],
        "mag_mag": ["std"],  # variance in magnetic field ↑ when doors open
    }
    df_feat = df_uniform.resample(win).agg(agg)
    df_feat.columns = ["_".join(c) for c in df_feat.columns]  # flatten index
    df_feat = df_feat.dropna()

    X_test = df_feat  # .values
    # load the model
    # MODEL_PATH = Path(__file__).with_name("saved_models/train_stop_detector_9.joblib")
    pipe = joblib.load(MODEL_PATH)
    predictions = pipe.predict_proba(X_test)[:, 1]

    recording_start = df_feat.index[0]  # pandas.Timestamp

    pred_intervals = proba_to_intervals(
        predictions,
        X_test.index,
        recording_start,
        thr=0.70,
        win_len=1.0,
        min_stop=10.0,
        merge_gap=3.0,
        return_datetime=False,
    )  # to get both numeric & datetime

    pred_intervals = [(float(s), float(e)) for s, e in pred_intervals]

    return pred_intervals
