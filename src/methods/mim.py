"""
Feature selection method implementations.
Each class follows the sklearn BaseEstimator/TransformerMixin interface.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.metrics import mutual_info_score

class MIMFeatureSelector(BaseEstimator, TransformerMixin):
    """
    MIM (Mutual Information Maximization) - Discrete MI version.
    Procedure:
      1) Discretize continuous features into n_bins (uniform-width).
      2) Rank features by I(X_i; Y) using mutual_info_score.
      3) Keep top ratio features (fixed budget).
    """
    def __init__(self, ratio=0.5, n_bins=3, random_state=42):
        self.ratio = ratio
        self.n_bins = n_bins
        self.random_state = random_state
        self.selected_indices_ = np.array([], dtype=int)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_keep = max(1, int(np.ceil(n_features * self.ratio)))

        # Discretize X (fit on training split inside CV/pipeline)
        est = KBinsDiscretizer(
            n_bins=self.n_bins, encode="ordinal", strategy="uniform", subsample=None
        )
        try:
            X_disc = est.fit_transform(X).astype(int)
        except Exception:
            X_disc = X.astype(int)

        # Ensure y is integer-coded
        if not np.issubdtype(y.dtype, np.integer):
            y = LabelEncoder().fit_transform(y)

        mi = np.zeros(n_features, dtype=float)
        for i in range(n_features):
            mi[i] = mutual_info_score(X_disc[:, i], y)

        self.selected_indices_ = np.argsort(mi)[::-1][:n_keep]
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]
