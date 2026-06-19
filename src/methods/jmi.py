"""
Feature selection method implementations.
Each class follows the sklearn BaseEstimator/TransformerMixin interface.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.metrics import mutual_info_score

class JMIFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Joint Mutual Information feature selection (Yang & Moody, 1999),
    discretized for speed.

    Each candidate's score accumulates I(f_cand, f_sel; Y) over every
    feature already selected, so the chosen subset jointly favours
    relevance to the target and complementarity with the features
    already in the set: Score(f) = sum_{s in Selected} I(f, s; Y).
    """
    def __init__(self, ratio=0.5, n_bins=3, random_state=42):
        self.ratio = ratio
        self.n_bins = n_bins
        self.random_state = random_state
        self.selected_indices_ = []

    def _entropy(self, values):
        """Shannon entropy H(X) for an array of discrete values."""
        counts = np.bincount(values)
        probs = counts[counts > 0] / len(values)
        return -np.sum(probs * np.log2(probs))

    def _joint_entropy_2(self, x, y):
        """Joint entropy H(X, Y), via integer packing of (x, y)."""
        combined = x + (np.max(x) + 1) * y
        return self._entropy(combined)

    def _joint_entropy_3(self, x, z, y):
        """Joint entropy H(X, Z, Y), via integer packing of (x, z, y)."""
        mult1 = np.max(x) + 1
        mult2 = mult1 * (np.max(z) + 1)
        combined = x + mult1 * z + mult2 * y
        return self._entropy(combined)

    def _calculate_joint_mutual_info(self, f_cand, f_sel, y, h_y):
        """
        Joint mutual information I(f_cand, f_sel; Y), via
        H(f_cand, f_sel) + H(Y) - H(f_cand, f_sel, Y).
        """
        h_xz = self._joint_entropy_2(f_cand, f_sel)
        h_xzy = self._joint_entropy_3(f_cand, f_sel, y)
        return h_xz + h_y - h_xzy

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_keep = int(np.ceil(n_features * self.ratio))
        n_keep = max(1, n_keep)

        # Discretizing keeps entropy estimation O(N), same rationale
        # as the CMIM selector.
        est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform', subsample=None)
        try:
            X_disc = est.fit_transform(X).astype(int)
        except ValueError:
            # Falls back to raw values if binning fails, e.g. on
            # constant or degenerate columns.
            X_disc = X.astype(int)

        if not np.issubdtype(y.dtype, np.integer):
            y = LabelEncoder().fit_transform(y)

        h_y = self._entropy(y)

        # First feature is chosen by plain MI(X; Y).
        mi_scores = np.zeros(n_features)
        for i in range(n_features):
            mi_scores[i] = mutual_info_score(X_disc[:, i], y)

        selected = []
        mask = np.ones(n_features, dtype=bool)

        best_idx = np.argmax(mi_scores)
        selected.append(best_idx)
        mask[best_idx] = False

        # Running JMI accumulator: JMI(X_k) = sum_{j in S} I(X_k, X_j; Y).
        # Starts at zero and gains one term each time a new feature is
        # added to the selected set, so every candidate's score only
        # needs the contribution from the newest selection each round
        # rather than being recomputed from scratch.
        jmi_scores = np.zeros(n_features)

        for _ in range(n_keep - 1):
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                break

            last_selected_feat = X_disc[:, selected[-1]]

            for idx in valid_indices:
                cand_data = X_disc[:, idx]
                joint_mi_term = self._calculate_joint_mutual_info(cand_data, last_selected_feat, y, h_y)
                jmi_scores[idx] += joint_mi_term

            best_idx_local = np.argmax(jmi_scores[valid_indices])
            best_feat = valid_indices[best_idx_local]

            selected.append(best_feat)
            mask[best_feat] = False

        self.selected_indices_ = np.array(selected)
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]
