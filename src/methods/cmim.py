"""
Feature selection method implementations.
Each class follows the sklearn BaseEstimator/TransformerMixin interface.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.metrics import mutual_info_score

class CMIMFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Conditional Mutual Information Maximisation (Fleuret, 2004),
    discretized for speed.

    At each step the next feature is chosen to maximise its mutual
    information with the target conditioned on the features already
    selected. Following Fleuret's fast CMIM, conditioning is only
    checked against the most recently selected feature, with each
    feature's running score capped to the minimum conditional MI seen
    so far.
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

    def _joint_entropy(self, x, y):
        """
        Joint entropy H(X, Y), computed by folding (x, y) into a single
        integer label. Assumes x and y are small non-negative integers
        (i.e. already discretized).
        """
        combined = x + (np.max(x) + 1) * y
        return self._entropy(combined)

    def _conditional_mi(self, f_cand, target, f_cond):
        """
        Conditional mutual information I(f_cand; target | f_cond), via
        H(f_cand, f_cond) + H(target, f_cond) - H(f_cand, target, f_cond) - H(f_cond).
        """
        h_z = self._entropy(f_cond)
        h_xz = self._joint_entropy(f_cand, f_cond)
        h_yz = self._joint_entropy(target, f_cond)

        # Fold f_cand and target together first so the 3-way joint
        # entropy can reuse _joint_entropy against f_cond.
        combined_xy = f_cand + (np.max(f_cand) + 1) * target
        h_xyz = self._joint_entropy(combined_xy, f_cond)

        return h_xz + h_yz - h_xyz - h_z

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_keep = int(np.ceil(n_features * self.ratio))
        n_keep = max(1, n_keep)

        # Discretizing keeps entropy estimation O(N) instead of relying
        # on continuous density estimates.
        est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform', subsample=None)
        try:
            X_disc = est.fit_transform(X).astype(int)
        except ValueError:
            # Falls back to raw values if binning fails, e.g. on
            # constant or degenerate columns.
            X_disc = X.astype(int)

        if not np.issubdtype(y.dtype, np.integer):
            y = LabelEncoder().fit_transform(y)

        # Initial scores are plain MI(X; Y), matching standard MIM.
        scores = np.zeros(n_features)
        for i in range(n_features):
            scores[i] = mutual_info_score(X_disc[:, i], y)

        selected = []
        current_mask = np.ones(n_features, dtype=bool)

        for _ in range(n_keep):
            valid_indices = np.where(current_mask)[0]
            if len(valid_indices) == 0:
                break

            best_idx_local = np.argmax(scores[valid_indices])
            best_feat = valid_indices[best_idx_local]

            selected.append(best_feat)
            current_mask[best_feat] = False

            # Update remaining scores against the feature just selected.
            # CMIM caps each score at the conditional MI with the newest
            # selection, so a feature's score reflects the most
            # redundant pairing seen so far rather than every pairing.
            f_sel_data = X_disc[:, best_feat]
            remaining_feats = np.where(current_mask)[0]

            for f_idx in remaining_feats:
                cond_mi = self._conditional_mi(X_disc[:, f_idx], y, f_sel_data)
                scores[f_idx] = min(scores[f_idx], cond_mi)

        self.selected_indices_ = np.array(selected)
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]
