"""
Feature selection method implementations.
Each class follows the sklearn BaseEstimator/TransformerMixin interface.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.metrics import mutual_info_score

class FHFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Fuzzy entropy feature selection, following Luukka (2011).

    For each class an ideal vector is computed as the (generalised)
    mean of its samples. Every sample's similarity to every class's
    ideal vector is then turned into a fuzzy entropy value, and
    features are ranked by the total entropy of their similarities.
    Low entropy means a feature separates the classes cleanly, so the
    n_keep features with the lowest entropy are retained.
    """
    def __init__(self, ratio=0.5, p=1.0, measure='luca', random_state=42):
        self.ratio = ratio
        self.p = p
        self.measure = measure
        self.random_state = random_state
        self.selected_indices_ = []

    def fit(self, X, y):
        # X is expected to already be scaled to [0, 1] upstream, as
        # required by the fuzzy similarity measure below.
        n_samples, n_features = X.shape
        n_keep = int(np.ceil(n_features * self.ratio))
        n_keep = max(1, n_keep)

        classes = np.unique(y)
        n_classes = len(classes)

        # Ideal vector per class: the per-feature mean of that class's
        # samples (p=1 reduces to the arithmetic mean, matching Luukka's
        # default).
        ideals = np.zeros((n_classes, n_features))
        for i, c in enumerate(classes):
            X_c = X[y == c]
            ideals[i, :] = np.mean(X_c, axis=0)

        # Entropy is computed per feature, vectorised across samples and
        # classes rather than the triple loop in the original script.
        entropies = np.zeros(n_features)
        delta = 1e-10

        if self.p != 1.0:
            X_pow = X ** self.p
            ideals_pow = ideals ** self.p
        else:
            X_pow = X
            ideals_pow = ideals

        for f in range(n_features):
            x_col = X_pow[:, f].reshape(-1, 1)   # (n_samples, 1)
            v_row = ideals_pow[:, f].reshape(1, -1)  # (1, n_classes)

            # Broadcast to get every sample's similarity to every
            # class's ideal vector in one shot: (n_samples, n_classes).
            # Generalised Lukasiewicz similarity, sim = (1 - |v^p - x^p|)^(1/p).
            diff = np.abs(v_row - x_col)

            if self.p == 1.0:
                sim = 1.0 - diff
            else:
                sim = (1.0 - diff) ** (1.0 / self.p)

            sim = np.clip(sim, 0.0, 1.0)
            sim_flat = sim.flatten()

            if self.measure == 'luca':
                # De Luca & Termini entropy; clip away from 0/1 to
                # avoid log(0).
                sim_flat[sim_flat == 1] = 1 - delta
                sim_flat[sim_flat == 0] = delta
                H = -sim_flat * np.log(sim_flat) - (1 - sim_flat) * np.log(1 - sim_flat)
            elif self.measure == 'park':
                # Parkash et al. sine-based entropy.
                H = np.sin(np.pi / 2 * sim_flat) + np.sin(np.pi / 2 * (1 - sim_flat)) - 1
            else:
                raise ValueError(f"Unknown entropy measure: '{self.measure}'")

            entropies[f] = np.sum(H)

        # Higher total entropy means a feature's similarity pattern is
        # less decisive across classes, i.e. a worse feature. Keep the
        # n_keep features with the lowest entropy.
        self.selected_indices_ = np.argsort(entropies)[:n_keep]

        return self

    def transform(self, X):
        return X[:, self.selected_indices_]
