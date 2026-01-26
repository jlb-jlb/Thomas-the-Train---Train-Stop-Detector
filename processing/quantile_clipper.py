import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
