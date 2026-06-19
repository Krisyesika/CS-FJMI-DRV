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
    Mutual Information Maximisation, the simplest filter baseline:
    features are ranked independently by I(X_i; Y) with no redundancy
    term, and the top-ranked features are kept up to a fixed budget.

    Continuous features are discretized into n_bins uniform-width bins
    before scoring, since mutual_info_score expects discrete inputs.
    """
    def __init__(self, ratio=0.5, n_bins=3, random_state=42):
        self.ratio = ratio
        self.n_bins = n_bins
        self.random_state = random_state
        self.selected_indices_ = np.array([], dtype=int)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_keep = max(1, int(np.ceil(n_features * self.ratio)))

        # Fit on this split only, so the binning stays consistent with
        # how this selector behaves inside a CV/pipeline.
        est = KBinsDiscretizer(
            n_bins=self.n_bins, encode="ordinal", strategy="uniform", subsample=None
        )
        try:
            X_disc = est.fit_transform(X).astype(int)
        except ValueError:
            # Falls back to raw values if binning fails, e.g. on
            # constant or degenerate columns.
            X_disc = X.astype(int)

        if not np.issubdtype(y.dtype, np.integer):
            y = LabelEncoder().fit_transform(y)

        mi = np.zeros(n_features, dtype=float)
        for i in range(n_features):
            mi[i] = mutual_info_score(X_disc[:, i], y)

        self.selected_indices_ = np.argsort(mi)[::-1][:n_keep]
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]
