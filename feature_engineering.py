# feature_engineering.py
import pandas as pd
import numpy as np
from typing import Sequence
import logging

__all__ = ["make_feature_table"]





def extra_stats(df_u: pd.DataFrame,  win='2s') -> pd.DataFrame:
    """Example: rolling energy in 0‑3 Hz band of dyn_acc."""
    roll = df_u['dyn_acc'].rolling(int(pd.Timedelta(win)/df_u.index.freq))
    df_u['dyn_acc_var2s'] = roll.var()
    return df_u


def make_feature_table(
    df_uniform: pd.DataFrame, 
    window_len: str = '1s', 
    lag_steps: Sequence[int] = (1,2),
    diff_steps: Sequence[int] = (1,),
    add_diff: bool = True,
    add_extra: bool = False,
) -> pd.DataFrame:
    
    info_str = f"Window: {window_len}, Lags: {lag_steps}, Diffs: {diff_steps}, Extra: {add_extra}"
    logging.debug(info_str)
    # print(info_str)
    # add extra True: calls extra_stats before resampling
    if add_extra:
        df_uniform = extra_stats(df_uniform.copy())


    agg = {
        "dyn_acc": ["mean", "std", "max", "median"],
        "gyro_mag": ["mean", "std", "max"],
        "mag_mag": ["std"],
    }
    if add_extra:
        agg["dyn_acc_var2s"] = ["mean"]

    df_feat = df_uniform.resample(window_len).agg(agg)
    df_feat.columns = ["_".join(c) for c in df_feat.columns]  # flatten index
    base_cols = df_feat.columns.tolist()

    
    lagged_features = {
        f"{c}_lag{l}": df_feat[c].shift(l) for l in lag_steps for c in base_cols
    }
    df_feat = pd.concat([df_feat, pd.DataFrame(lagged_features, index=df_feat.index)], axis=1)
    

    # first‑order differences (current – previous second)
    if add_diff:
        diff_features = {
            f"{c}_diff{d}": df_feat[c].diff(d) for d in diff_steps for c in base_cols
        }
        df_feat = pd.concat([df_feat, pd.DataFrame(diff_features, index=df_feat.index)], axis=1)



    return df_feat.dropna()



def list_base_columns() -> list[str]:
    """
    Build the minimum viable DataFrame that make_feature_table()
    can aggregate, then return the 8 base column names it produces
    (dyn_acc_mean, dyn_acc_std, …, mag_mag_std).
    """
    # a two‑row stub that already contains the three per‑sample features
    stub = pd.DataFrame(
        {
            "dyn_acc":  [0.0, 0.0],
            "gyro_mag": [0.0, 0.0],
            "mag_mag":  [0.0, 0.0],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="1s"),
    )

    tmp = make_feature_table(
        stub,
        lag_steps=(),          # no lags
        add_diff=False,        # no diffs
        add_extra=False,       # no extra stats
    )

    # keep only the base stats (no _lag / _diff suffixes)
    return [c for c in tmp.columns if "_lag" not in c and "_diff" not in c]


def columns_for_cfg(feat_cfg: dict, base_cols: list[str]) -> list[str]:
    """Given a feature config, return the column names required."""
    cols = []
    # base & extra stats ------------------------------------------------
    for c in base_cols:
        cols.append(c)
        if feat_cfg.get("add_diff", True):
            cols.append(f"{c}_diff1")
        for l in feat_cfg["lag_steps"]:
            cols.append(f"{c}_lag{l}")
    if feat_cfg.get("add_extra", False):
        cols.append("dyn_acc_var2s_mean")
        if feat_cfg.get("add_diff", True):
            cols.append("dyn_acc_var2s_mean_diff1")
        for l in feat_cfg["lag_steps"]:
            cols.append(f"dyn_acc_var2s_mean_lag{l}")
    return cols