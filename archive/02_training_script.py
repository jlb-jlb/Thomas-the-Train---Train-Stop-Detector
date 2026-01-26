# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from scipy.signal import medfilt  # median filter

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# %%
# read all files from data/train
data_dir = "data/train"


# Updated function to include start_station, end_station, and passed_stations
def load_train_data(data_dir):
    train_data = {}

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            train_name = folder.split("_")[0]
            if train_name not in train_data:
                train_data[train_name] = []

            data_file = os.path.join(folder_path, "data.npz")
            readme_file = os.path.join(folder_path, "README.txt")

            if os.path.exists(data_file) and os.path.exists(readme_file):
                with open(readme_file, "r") as f:
                    readme_content = f.read()

                # Extract journey details from the readme content
                journey_parts = readme_content.split("Journey with ")[1].split(" from ")
                start_station = journey_parts[1].split(" to ")[0]
                end_station = journey_parts[1].split(" to ")[1].split(" via ")[0]
                passed_stations = journey_parts[1].split(" via ")[1].split(", ")
                # reverse the order of passed stations
                passed_stations = passed_stations[::-1]
                # print(data_file)
                train_data[train_name].append(
                    {
                        "data": np.load(data_file),
                        "readme": readme_content,
                        "start_station": start_station,
                        "end_station": end_station,
                        "passed_stations": passed_stations,
                    }
                )

    return train_data


# %%
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


# %%
# Resample and interpolate the data
def resample_and_interpolate(df, freq="10ms"):
    df_uniform = df.copy()
    df_uniform = df_uniform.resample(freq).mean().interpolate()
    return df_uniform


# %%
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


# %%
def aggregated_features(df_uniform, window="1s"):

    agg = {
        "dyn_acc": ["mean", "std", "max", "median"],
        "gyro_mag": ["mean", "std", "max"],
        "mag_mag": ["std"],  # variance in magnetic field ↑ when doors open
    }
    df_feat = df_uniform.resample(window).agg(agg)
    df_feat.columns = ["_".join(c) for c in df_feat.columns]  # flatten index
    df_feat = df_feat.dropna()


# %%
def prepare_ground_truth(df_feat, intervals):
    gt = intervals.copy()
    y = np.zeros(len(df_feat), dtype=int)
    t_centres = (df_feat.index - df_feat.index[0]).total_seconds()
    for start, end in gt:
        y[(t_centres >= start) & (t_centres <= end)] = 1
    print(y.mean())
    return y


# %%
class QuantileClipper(BaseEstimator, TransformerMixin):
    """
    Clips each column to [q_lower, q_upper] computed on the *training* set.
    Works with pandas DataFrames or NumPy arrays.
    """

    def __init__(self, q_lower=0.5, q_upper=99.5):
        self.q_lower = q_lower
        self.q_upper = q_upper

    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        X_ = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.lo_ = X_.quantile(self.q_lower / 100, axis=0).values
        self.hi_ = X_.quantile(self.q_upper / 100, axis=0).values
        return self

    # ------------------------------------------------------------------
    def transform(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_clipped = np.clip(X_arr, self.lo_, self.hi_)
        return (
            pd.DataFrame(X_clipped, index=X.index, columns=X.columns)
            if isinstance(X, pd.DataFrame)
            else X_clipped
        )


# %%


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
    print("file_len", file_len)

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


# %%
def trip_to_features(npz, group_id):  # data_sample from np.load(...)  # e.g. "S2_01"
    # ---------- concat on native timestamps, resample to 20 Hz ----------
    df_raw = to_df(npz)
    df = df_raw.resample("10ms").mean().interpolate()  # 20 Hz

    # ---------- per‑sample magnitudes ----------
    acc = df[["accel_x", "accel_y", "accel_z"]].values
    gra = df[["gravi_x", "gravi_y", "gravi_z"]].values
    gyr = df[["gyros_x", "gyros_y", "gyros_z"]].values

    df["dyn_acc"] = np.linalg.norm(acc - gra, axis=1)
    df["gyro_mag"] = np.linalg.norm(gyr, axis=1)
    df["mag_mag"] = np.linalg.norm(df[["magne_x", "magne_y", "magne_z"]].values, axis=1)

    # ---------- 1‑second windows ----------
    win = "1s"
    agg = {
        "dyn_acc": ["mean", "std", "max", "median"],
        "gyro_mag": ["mean", "std", "max"],
        "mag_mag": ["std"],
    }
    feat = df.resample(win).agg(agg)
    feat.columns = ["_".join(c) for c in feat.columns]
    feat = feat.dropna()

    # ---------- labels ----------
    y = np.zeros(len(feat), dtype=int)
    t_centres = (feat.index - feat.index[0]).total_seconds()
    for s, e in npz["intervals"]:
        y[(t_centres >= s) & (t_centres <= e)] = 1

    return feat, y, np.repeat(group_id, len(feat))


def save_model(model_name, model_dir, model):
    import joblib
    import os

    os.makedirs(model_dir, exist_ok=True)
    models = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
    if models:
        latest_model = max(models, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        latest_index = int(latest_model.split("_")[-1].split(".")[0])
        new_index = latest_index + 1
    else:
        new_index = 1
    model_path = os.path.join("saved_models", f"{model_name}_{new_index}.joblib")
    joblib.dump(pipe, model_path)
    print("Model saved to", model_path)
    return model_path


# %%
if __name__ == "__main__":
    print("Loading data...")
    train_data = load_train_data(data_dir)
    print("Data loaded.")
    # print(train_data)
    X_list, y_list, g_list = [], [], []

    for train_name, trips in train_data.items():  # your dict from load_train_data
        for idx, trip in enumerate(trips):
            group_id = f"{train_name}_{idx}"  # e.g. "S2_03"
            Xf, yf, gf = trip_to_features(trip["data"], group_id)
            X_list.append(Xf)
            y_list.append(yf)
            g_list.append(gf)
    X_all = pd.concat(X_list, axis=0)
    y_all = np.concatenate(y_list)
    groups = np.concatenate(g_list)
    print(X_all.shape, y_all.shape, groups.shape)
    print(X_all.head())
    print(X_all.info())

    gss = GroupShuffleSplit(test_size=0.2, random_state=4)  # 25 % of trips for val
    train_idx, val_idx = next(gss.split(X_all, y_all, groups))
    print(train_idx.shape, val_idx.shape)
    print(f"train_idx: {train_idx[:5]} ...")
    print(f"val_idx:   {val_idx[:5]} ...")

    X_train, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
    y_train, y_val = y_all[train_idx], y_all[val_idx]

    pipe = Pipeline(
        [
            ("clip", QuantileClipper(q_lower=1, q_upper=99)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=10,
                    learning_rate=0.12,
                    class_weight="balanced",
                    early_stopping=True,
                    random_state=42,
                ),
            ),
        ]
    )
    print("Fitting...")
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_val)
    print(f"val‑F1 = {f1_score(y_val, predictions):.3f}")
    cm = confusion_matrix(y_val, predictions)
    print(f"Confusion Matrix: \n\r{cm}")

    save_model("train_stop_detector", "saved_models", pipe)

    # Train the model on all data
    print("Training on all data...")
    pipe.fit(X_all, y_all)
    print("Training complete.")

    # Save the model
