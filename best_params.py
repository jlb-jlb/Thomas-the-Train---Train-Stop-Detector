# BEST_FEATURE_CFG = dict(
#     lag_steps   = list(range(1, 21)),
#     add_diff    = True,
#     diff_steps  = (1,),
#     add_extra   = False,
#     window_len  = "1s",
# )
## didn't do hyperparam training on 
BEST_FEATURE_CFG = dict(
    lag_steps   = list(range(1, 21)),
    add_diff    = True,
    diff_steps  = list(range(1, 21)),
    add_extra   = False,
    window_len  = "1s",
)

HGB_PARAMS = dict(
    max_depth      = 10,
    learning_rate  = 0.15,
    class_weight   = "balanced",
    early_stopping = True,
    random_state   = 42,
)


# had to try a few different values for the post-processing parameters, cause tuning was not 100% reliable
BEST_POSTPROC = dict(
    thr       = 0.5,
    min_stop  = 3,
    merge_gap = 2,
    med_k     = 5,
    win_len   = 1.0,
) 
# --> 0.702

# BEST_POSTPROC = dict(
#     thr       = 0.6,
#     min_stop  = 5,
#     merge_gap = 2,
#     med_k     = 5,
#     win_len   = 1.0,
# ) 
# -->  0.6849

# BEST_POSTPROC = dict(
#     thr       = 0.75,
#     min_stop  = 2,
#     merge_gap = 3,
#     med_k     = 5,
#     win_len   = 1.0,
# ) 
# --> 0.6175