"""
Feature selection method implementations.
Each class follows the sklearn BaseEstimator/TransformerMixin interface.
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.metrics import mutual_info_score

class FJMIURFeatureSelector(BaseEstimator, TransformerMixin):
    """
    FJMI with Uncertainty Region instance selection (FJMI-UR).

    Runs in two phases:
      1. ISUR: reduce the training set to its uncertainty region by
         keeping every minority-class sample and only the majority-class
         samples nearest to them, before any feature scoring happens.
      2. FJMI with auto-threshold: greedy forward selection on the
         reduced set, scoring each candidate feature by how much it
         adds on top of the current selection via fuzzy interaction
         information, and stopping once that score stops improving.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.selected_indices_ = []
        self.selected_instance_count_ = 0

    # ISUR helpers

    def _distfunc(self, X, Y):
        """
        For every row in X, find the index of its nearest row in Y
        (Euclidean distance). Used to pick the majority-class samples
        closest to each minority-class sample.
        """
        D = cdist(X, Y, metric='euclidean')
        return np.argmin(D, axis=1)

    def _ISUR(self, X, y):
        """
        Instance selection based on the uncertainty region: keep all
        minority-class samples, and for each one keep its nearest
        majority-class neighbour. Only meaningful for binary class
        splits; with more than two classes the smallest class is
        treated as the minority and everything else as the majority.
        """
        classes = np.unique(y)
        if len(classes) < 2:
            return X, y

        class_counts = [np.sum(y == c) for c in classes]
        min_cls_idx = np.argmin(class_counts)
        min_cls_label = classes[min_cls_idx]

        mask_min = (y == min_cls_label)
        X_min = X[mask_min]
        y_min = y[mask_min]

        mask_max = ~mask_min
        X_max = X[mask_max]
        y_max = y[mask_max]

        if len(X_max) == 0:
            return X, y

        sel_indices = self._distfunc(X_min, X_max)
        X_max_sel = X_max[sel_indices]
        y_max_sel = y_max[sel_indices]

        X_new = np.vstack((X_min, X_max_sel))
        y_new = np.hstack((y_min, y_max_sel))

        return X_new, y_new

    # FJMI helpers (shared logic with the plain FJMI baseline)

    def _relation(self, x):
        diff = np.abs(x[:, None] - x[None, :])
        return np.exp(-diff)

    def _decision(self, y):
        return (y[:, None] == y[None, :]).astype(float)

    def _fuzzy_entropy(self, rel_matrix):
        n = rel_matrix.shape[0]
        cardinalities = np.sum(rel_matrix, axis=1)
        probs = cardinalities / n
        probs = np.clip(probs, 1e-12, 1.0)
        return -np.mean(np.log2(probs))

    def _F_MI(self, relx, rely):
        Hx = self._fuzzy_entropy(relx)
        Hy = self._fuzzy_entropy(rely)
        relxy = np.minimum(relx, rely)
        Hxy = self._fuzzy_entropy(relxy)
        return Hx + Hy - Hxy

    def _iF_MI(self, relx, rely, relz):
        """
        Fuzzy interaction information I(f; SN; C) among a candidate
        feature, the current selection summary SN, and the class.
        """
        Hx = self._fuzzy_entropy(relx)
        Hy = self._fuzzy_entropy(rely)
        Hz = self._fuzzy_entropy(relz)

        relxy = np.minimum(relx, rely)
        Hxy = self._fuzzy_entropy(relxy)

        relxz = np.minimum(relx, relz)
        Hxz = self._fuzzy_entropy(relxz)

        relyz = np.minimum(rely, relz)
        Hyz = self._fuzzy_entropy(relyz)

        relxyz = np.minimum(np.minimum(relx, rely), relz)
        Hxyz = self._fuzzy_entropy(relxyz)

        return (Hx + Hy + Hz + Hxyz) - (Hxy + Hxz + Hyz)

    def fit(self, X, y):
        X_reduced, y_reduced = self._ISUR(X, y)
        self.selected_instance_count_ = X_reduced.shape[0]
        self._fjmi_logic(X_reduced, y_reduced)
        return self

    def _fjmi_logic(self, X, y):
        n_features = X.shape[1]

        RS = []
        FRM = None  # running relation summary of the selected features
        best_vals = []

        available = np.ones(n_features, dtype=bool)
        rel_C = self._decision(y)

        # First feature: maximise standalone I(f; C).
        max_mi = -np.inf
        first_feature = -1

        for i in range(n_features):
            rel_f = self._relation(X[:, i])
            i_cx = self._F_MI(rel_f, rel_C)
            if i_cx > max_mi:
                max_mi = i_cx
                first_feature = i

        if first_feature == -1:
            return

        RS.append(first_feature)
        best_vals.append(max_mi)
        available[first_feature] = False
        FRM = self._relation(X[:, first_feature])

        # Forward selection with auto-threshold stopping: each round
        # picks the candidate that maximises I(f; C) + I(SN; C) - I(f; SN; C),
        # i.e. its individual relevance plus the current selection's
        # relevance, minus their shared (redundant) information.
        for _ in range(1, n_features):
            max_IcSNx = -np.inf
            best_feature_next = -1

            candidates = np.where(available)[0]
            if len(candidates) == 0:
                break

            I_CSN = self._F_MI(FRM, rel_C)

            for i in candidates:
                rel_f = self._relation(X[:, i])
                I_SNxc = self._iF_MI(rel_f, FRM, rel_C)
                I_Cx = self._F_MI(rel_f, rel_C)

                JRes = I_Cx + I_CSN - I_SNxc

                if JRes > max_IcSNx:
                    max_IcSNx = JRes
                    best_feature_next = i

            # Stop once the best candidate no longer improves on the
            # previous round's score.
            if max_IcSNx <= best_vals[-1]:
                break

            RS.append(best_feature_next)
            best_vals.append(max_IcSNx)
            available[best_feature_next] = False

            rel_next = self._relation(X[:, best_feature_next])
            FRM = np.minimum(FRM, rel_next)

        self.selected_indices_ = np.array(RS)

    def transform(self, X):
        return X[:, self.selected_indices_]
