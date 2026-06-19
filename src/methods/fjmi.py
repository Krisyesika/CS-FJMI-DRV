"""
Feature selection method implementations.
Each class follows the sklearn BaseEstimator/TransformerMixin interface.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.metrics import mutual_info_score

class FJMIFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Fuzzy Joint Mutual Information feature selection.
    Reference: Salem et al., "Feature selection and threshold method
    based on fuzzy joint mutual information" (2021).

    Features are added greedily to maximise joint MI with the class
    label, where the running set of selected features is summarised by
    a single fuzzy relation matrix (FRM) via intersection. Selection
    stops once the score from adding a feature no longer improves on
    the previous round, matching Algorithm 1 in the reference paper.

    This baseline intentionally uses the full N x N relation matrix per
    feature rather than any subsampling or prototype reduction, so its
    runtime reflects the true O(N^2) cost the proposed method improves
    on.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.selected_indices_ = []

    def _relation(self, x):
        """
        Eq. (15): fuzzy similarity relation E_f(x_i, x_j) = exp(-|x_i - x_j|).
        O(N^2) memory; for N=11500 this is roughly 1 GB per feature, by
        design (see class docstring).
        """
        diff = np.abs(x[:, None] - x[None, :])
        return np.exp(-diff)

    def _decision(self, y):
        """Eq. (16): crisp equivalence relation induced by the class label."""
        return (y[:, None] == y[None, :]).astype(float)

    def _fuzzy_entropy(self, rel_matrix):
        """Eq. (5): fuzzy entropy H(f) of a relation matrix."""
        n = rel_matrix.shape[0]
        cardinalities = np.sum(rel_matrix, axis=1)
        probs = cardinalities / n
        probs = np.clip(probs, 1e-12, 1.0)
        return -np.mean(np.log2(probs))

    def _F_MI(self, relx, rely):
        """Eq. (8): fuzzy mutual information I(X; Y) = H(X) + H(Y) - H(X, Y)."""
        Hx = self._fuzzy_entropy(relx)
        Hy = self._fuzzy_entropy(rely)
        relxy = np.minimum(relx, rely)
        Hxy = self._fuzzy_entropy(relxy)
        return Hx + Hy - Hxy

    def _Joint_MI(self, rel_f, FRM, rel_C):
        """
        Joint MI of a candidate feature and the current FRM against the
        class, I(E_f, FRM; E_C), following Eq. (17). The joint relation
        of (f, FRM) is their intersection (elementwise minimum), after
        which this reduces to a standard fuzzy MI computation.
        """
        rel_joint_features = np.minimum(rel_f, FRM)
        return self._F_MI(rel_joint_features, rel_C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Algorithm 1, lines 1-5: initialise the selected set and FRM.
        RS = []
        FRM = None
        current_score = 0.0
        previous_score = 0.0

        available_features = np.ones(n_features, dtype=bool)
        rel_C = self._decision(y)

        # Algorithm 1, lines 6-12: pick the first feature by max I(f; C).
        # Relation matrices are built on demand per feature rather than
        # all at once, since holding every N x N matrix in memory at the
        # same time isn't feasible at this scale.
        candidates_score = np.zeros(n_features)
        for i in range(n_features):
            rel_f = self._relation(X[:, i])
            candidates_score[i] = self._F_MI(rel_f, rel_C)

        best_first_idx = np.argmax(candidates_score)
        max_first_score = candidates_score[best_first_idx]

        RS.append(best_first_idx)
        available_features[best_first_idx] = False
        FRM = self._relation(X[:, best_first_idx])
        current_score = max_first_score

        # Algorithm 1, lines 13-22: greedily grow the subset while the
        # joint score keeps improving.
        #
        # Note on the stopping behaviour: a feature is added to RS
        # before its score is compared against the previous round, so
        # the feature that first causes the score to drop is still kept
        # in the final selection, with the while loop only exiting on
        # the following iteration. This matches the original algorithm
        # as written rather than stopping one feature earlier.
        while current_score > previous_score:
            previous_score = current_score

            best_new_score = -np.inf
            best_new_idx = -1

            candidate_indices = np.where(available_features)[0]
            if len(candidate_indices) == 0:
                break

            for f_i in candidate_indices:
                rel_f = self._relation(X[:, f_i])
                j_score = self._Joint_MI(rel_f, FRM, rel_C)

                if j_score > best_new_score:
                    best_new_score = j_score
                    best_new_idx = f_i

            if best_new_idx != -1:
                RS.append(best_new_idx)
                available_features[best_new_idx] = False

                # FRM <- M(ind({FRM, M(f_i)})) = min(FRM, M(f_i))
                rel_best = self._relation(X[:, best_new_idx])
                FRM = np.minimum(FRM, rel_best)

                current_score = best_new_score
            else:
                break

        self.selected_indices_ = np.array(RS)
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]
