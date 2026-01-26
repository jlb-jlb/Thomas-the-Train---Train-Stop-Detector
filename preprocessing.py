# preprocessing.py
import numpy as np, pandas as pd

__all__ = ["raw_to_uniform", "add_per_sample_features"]


def raw_to_uniform(
    streams: dict, target_freq="10ms"  # dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    streams = {"magnetometer": arr, "accelerometer": arr, ...}
    Each arr has shape (n, 4): t, x, y, z.
    Returns one DataFrame indexed at `target_freq`.
    """
    cols = {
        "magnetometer": ["t", "magne_x", "magne_y", "magne_z"],
        "accelerometer": ["t", "accel_x", "accel_y", "accel_z"],
        "gravity": ["t", "gravi_x", "gravi_y", "gravi_z"],
        "gyroscope": ["t", "gyros_x", "gyros_y", "gyros_z"],
    }
    dfs = (
        pd.DataFrame(streams[s], columns=cols[s])
        .assign(t=lambda d: pd.to_datetime(d["t"], unit="s"))
        .set_index("t")
        for s in cols
    )
    df = pd.concat(dfs, axis=1)
    df.index = (
        df.index
        - pd.Timestamp("1970-01-01")   # 
        + pd.Timestamp("2024-01-01")   # 
    )
    return df.resample(target_freq).mean().interpolate()


def add_per_sample_features(df: pd.DataFrame) -> pd.DataFrame:
    acc = df[["accel_x", "accel_y", "accel_z"]].to_numpy()
    gra = df[["gravi_x", "gravi_y", "gravi_z"]].to_numpy()
    gyr = df[["gyros_x", "gyros_y", "gyros_z"]].to_numpy()
    df["dyn_acc"] = np.linalg.norm(acc - gra, axis=1)
    df["gyro_mag"] = np.linalg.norm(gyr, axis=1)
    df["mag_mag"] = np.linalg.norm(
        df[["magne_x", "magne_y", "magne_z"]].to_numpy(), axis=1
    )
    return df
