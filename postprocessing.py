# postprocessing.py
import numpy as np, pandas as pd
from scipy.signal import medfilt

__all__ = ["proba_to_intervals", "POSTPROC_CFG", "mean_interval_iou"]

POSTPROC_CFG = dict(
    thr=0.70,
    win_len=1.0,
    min_stop=10.0,
    merge_gap=3.0,
    med_k=7,
)


def proba_to_intervals(
    proba,  # 1‑D array, same length as time_index
    time_index,  # X_val.index  (DateTimeIndex)
    recording_start,  # df_feat.index[0]  (Timestamp)
    *,
    thr: float = 0.60,  # threshold
    win_len: float = 1.0,  # seconds
    min_stop: float = 1.0,  # seconds
    merge_gap: float = 15.0,  # seconds
    med_k: int = 7,  # median filter kernel size must be odd
    return_datetime=False,  # if True, also return Timestamp pairs
):
    """
    Convert window-level probabilities to stop intervals.
    Parameters
    ----------
    proba : 1-D array
        Probabilities for each time window.
        time_index : pd.DatetimeIndex
        Time index for the probabilities.
        recording_start : pd.Timestamp
        Start time of the recording.
        thr : float
    """
    t_c = (time_index - recording_start).total_seconds().values
    file_len = t_c[-1] + win_len / 2

    #  Binary mask (+ median smoothing)
    mask = (proba >= thr).astype(int)
    if med_k > 1:
        mask = medfilt(mask, kernel_size=med_k)

    # locate start / end indices where mask flip
    edges = np.diff(mask, prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    # Build raw intervals, clip to [0, file_len], drop short ones
    intervals = []
    for s_i, e_i in zip(starts, ends):
        t0 = max(0, t_c[s_i] - win_len / 2)
        t1 = min(file_len, t_c[e_i - 1] + win_len / 2)
        if t1 - t0 >= min_stop:
            intervals.append([t0, t1])

    # merge gaps shorter than merge_gap
    merged = []
    for s, e in intervals:
        # if the gap between the current interval and the last merged interval is greater than merge_gap
        # or if there are no merged intervals yet, add the current interval to merged
        # otherwise, merge the current interval with the last merged interval
        if not merged or s - merged[-1][1] > merge_gap:
            merged.append([s, e])
        else:
            merged[-1][1] = e
    return [(float(s), float(e)) for s, e in merged]



def interval_f1_at_tau(pred, gt, tau=0.5):
    tp = 0
    matched = set()
    for p_s, p_e in pred:
        hit = False
        for j,(g_s, g_e) in enumerate(gt):
            if j in matched: continue
            inter = max(0, min(p_e, g_e) - max(p_s, g_s))
            union = max(p_e, g_e) - min(p_s, g_s)
            if union and inter/union >= tau:
                tp += 1; matched.add(j); hit = True; break
    fp = len(pred) - tp
    fn = len(gt)   - tp
    return 0 if tp==0 else 2*tp/(2*tp+fp+fn)



# EVALUATION METRICS USED FOR FINE TUNING THE POSTPROCESSING

def interval_f1(pred_intervals, gt_intervals, iou_thr=0.5) -> float:
    tp = 0,
    for p_s, p_e in pred_intervals:
        for g_s, g_e in gt_intervals:
            inter = max(0, min(p_e, g_e) - max(p_s, g_s))
            union = max(p_e, g_e) - min(p_s, g_s)
            if inter / union >= iou_thr:
                tp += 1
                break
    fp = len(pred_intervals) - tp
    fn = len(gt_intervals) - tp
    return 0 if tp==0 else 2*tp/(2*tp + fp + fn)


def mean_interval_iou(pred, gt):
    """
    Greedy match each GT interval with the pred that has max IoU.
    Returns the mean IoU over all GT intervals.
    """
    pred = pred.copy()
    ious = []
    for g_s, g_e in gt:
        best, best_i = 0, -1
        for i, (p_s, p_e) in enumerate(pred):
            inter = max(0, min(p_e, g_e) - max(p_s, g_s))
            union = max(p_e, g_e) - min(p_s, g_s)
            iou   = inter / union
            if iou > best:
                best, best_i = iou, i
        if best_i >= 0:
            ious.append(best)
            pred.pop(best_i)        # remove, 1‑to‑1 matching
        else:
            ious.append(0.0)        # missed GT stop
    return float(np.mean(ious))


def set_jaccard(pred, gt, grid=1.0):
    """
    Discretise the timeline on a `grid` (s), mark points covered by
    pred and gt, and compute J = |A∩B| / |A∪B|.
    """
    if not pred and not gt:
        return 1.0
    t_max = max(max(e for _, e in pred+gt), 1)
    t = np.arange(0, t_max, grid)
    mask_p = np.zeros_like(t, dtype=bool)
    mask_g = np.zeros_like(t, dtype=bool)
    for s,e in pred:
        mask_p[(t>=s)&(t<e)] = True
    for s,e in gt:
        mask_g[(t>=s)&(t<e)] = True
    inter = np.sum(mask_p & mask_g)
    union = np.sum(mask_p | mask_g)
    return 0.0 if union == 0 else inter / union