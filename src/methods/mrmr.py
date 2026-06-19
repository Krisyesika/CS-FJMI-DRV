"""
Feature selection method implementations.
Each class follows the sklearn BaseEstimator/TransformerMixin interface.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.metrics import mutual_info_score

class MRMRFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Minimum Redundancy Maximum Relevance (Peng et al., 2005),
    discretized for speed.

    Each candidate is scored by its relevance to the target minus its
    average redundancy with the features already selected:
    Score(f) = I(f; y) - (1/|S|) * sum_{s in Selected} I(f; s).
    Averaging by |S| is what distinguishes this from MIFS, where the
    redundancy term is summed rather than averaged and so grows
    unchecked as the selected set gets larger.
    """
    def __init__(self, ratio=0.5, n_bins=3, random_state=42):
        self.ratio = ratio
        self.n_bins = n_bins
        self.random_state = random_state
        self.selected_indices_ = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_keep = int(np.ceil(n_features * self.ratio))
        n_keep = max(1, n_keep)

        # Same binning scheme as CMIM/JMI/MIFS, so MI estimates stay
        # comparable across baselines.
        est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform', subsample=None)
        try:
            X_disc = est.fit_transform(X).astype(int)
        except ValueError:
            # Falls back to raw values if binning fails, e.g. on
            # constant or degenerate columns.
            X_disc = X.astype(int)

        if not np.issubdtype(y.dtype, np.integer):
            y = LabelEncoder().fit_transform(y)

        # Relevance: I(X_i; Y) for every feature.
        relevance = np.zeros(n_features)
        for i in range(n_features):
            relevance[i] = mutual_info_score(X_disc[:, i], y)

        selected = []
        mask = np.ones(n_features, dtype=bool)

        # First feature: highest standalone relevance, since the
        # redundancy term is still zero with an empty selected set.
        best_idx = np.argmax(relevance)
        selected.append(best_idx)
        mask[best_idx] = False

        # Running redundancy accumulator: redundancy(f) = sum_{s in S} I(f; s).
        # Gains one term each round from the feature just selected, then
        # gets divided by |S| at scoring time to keep its influence from
        # growing unbounded as more features are added.
        redundancy_accum = np.zeros(n_features)

        for _ in range(n_keep - 1):
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                break

            last_selected_feat = X_disc[:, selected[-1]]

            for idx in valid_indices:
                cand_data = X_disc[:, idx]
                mi_val = mutual_info_score(cand_data, last_selected_feat)
                redundancy_accum[idx] += mi_val

            n_selected = len(selected)
            current_redundancy_avg = redundancy_accum[valid_indices] / n_selected
            mrmr_scores = relevance[valid_indices] - current_redundancy_avg

            best_idx_local = np.argmax(mrmr_scores)
            best_feat = valid_indices[best_idx_local]

            selected.append(best_feat)
            mask[best_feat] = False

        self.selected_indices_ = np.array(selected)
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]
