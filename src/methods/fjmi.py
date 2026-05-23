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
    FJMI (Fuzzy Joint Mutual Information) Implementation.
    Reference: Salem et al., "Feature selection and threshold method based on fuzzy joint mutual information" (2021).
    
    Key Features:
    1. Auto-Threshold: Stops when I(NewFeature, FRM; C) <= Previous Score.
    2. No Subsampling: Uses full N x N matrices to demonstrate true time complexity.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.selected_indices_ = []

    # Helper Functions (Vectorized)
    
    def _relation(self, x):
        """
        Eq (15): E_f(xi, xj) = exp(-|xi - xj|)
        """
        # x shape: (n_samples,)
        # Output: (n_samples, n_samples)
        # Note: This is O(N^2) memory. For N=11500, this is ~1GB RAM per feature.
        diff = np.abs(x[:, None] - x[None, :])
        return np.exp(-diff)

    def _decision(self, y):
        """Eq (16): Crisp relation for class."""
        return (y[:, None] == y[None, :]).astype(float)

    def _fuzzy_entropy(self, rel_matrix):
        """Eq (5): H(f) calculation."""
        n = rel_matrix.shape[0]
        cardinalities = np.sum(rel_matrix, axis=1)
        probs = cardinalities / n
        probs = np.clip(probs, 1e-12, 1.0)
        return -np.mean(np.log2(probs))

    def _F_MI(self, relx, rely):
        """Eq (8): I(X; Y)"""
        Hx = self._fuzzy_entropy(relx)
        Hy = self._fuzzy_entropy(rely)
        relxy = np.minimum(relx, rely)
        Hxy = self._fuzzy_entropy(relxy)
        return Hx + Hy - Hxy

    def _Joint_MI(self, rel_f, FRM, rel_C):
        """
        Calculate Joint MI: I(E_f, FRM; E_C)
        Using Eq (17) concept.
        The paper implies: Joint Relation of (f, FRM) is intersection (min).
        So I(A, B; C) = I(Min(A, B); C) based on their logic of Indiscernibility.
        """
        # Indiscernibility of f and FRM (Intersection)
        rel_joint_features = np.minimum(rel_f, FRM)
        
        # Calculate MI between (f AND FRM) and Class
        # I(Joint; C) = H(Joint) + H(C) - H(Joint, C)
        return self._F_MI(rel_joint_features, rel_C)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # PHASE 1: Initialization (Algorithm 1 Lines 1-5)
        RS = [] # RS <- Empty
        FRM = None # FRM <- Empty
        current_score = 0.0
        previous_score = 0.0
        
        # Available features mask
        available_features = np.ones(n_features, dtype=bool)
        
        # Precompute Class Relation (relc)
        rel_C = self._decision(y)
        
        # PHASE 2: Select First Feature (Algorithm 1 Lines 6-12)
        # "Select feature with max I(f; C)"
        
        candidates_score = np.zeros(n_features)
        
        # Note: Calculating relation matrix for ALL features at once might kill RAM.
        # We calculate on demand inside loop.
        for i in range(n_features):
            rel_f = self._relation(X[:, i])
            candidates_score[i] = self._F_MI(rel_f, rel_C)
            
        best_first_idx = np.argmax(candidates_score)
        max_first_score = candidates_score[best_first_idx]
        
        # Update State
        RS.append(best_first_idx)
        available_features[best_first_idx] = False
        
        # FRM <- M(f_i)
        FRM = self._relation(X[:, best_first_idx])
        
        # current <- score
        current_score = max_first_score
        
        # PHASE 3: Select Best Feature Subset (Algorithm 1 Lines 13-22)
        # "while current > previous do"
        
        while current_score > previous_score:
            previous_score = current_score
            
            best_new_score = -np.inf
            best_new_idx = -1
            
            # Identify candidates
            candidate_indices = np.where(available_features)[0]
            if len(candidate_indices) == 0:
                break
            
            # Loop candidates (Line 15)
            for f_i in candidate_indices:
                rel_f = self._relation(X[:, f_i])
                
                # Calculate I(E_f, FRM; E_C) (Line 16)
                # Note: The paper calls this Joint MI.
                # In Algorithm 1 line 16: candidate[] <- I(E_f, FRM; E_C)
                j_score = self._Joint_MI(rel_f, FRM, rel_C)
                
                if j_score > best_new_score:
                    best_new_score = j_score
                    best_new_idx = f_i
            
            # Check stopping condition implicitly via the loop check next time
            # Algorithm 1 Line 17: Select max
            # Line 18: RS <- RS U {f_i} (It adds it regardless, then checks condition next loop)
            # Line 22: current <- score
            
            # Implementasi Note: Jika best_new_score <= previous_score, 
            # Algoritma asli akan menambahkan fitur tersebut, mengupdate current, 
            # lalu loop 'while' akan berhenti di iterasi BERIKUTNYA.
            # Artinya fitur terakhir yang membuat skor turun TETAP terpilih.
            
            if best_new_idx != -1:
                RS.append(best_new_idx)
                available_features[best_new_idx] = False
                
                # Update FRM (Line 20-21)
                # P = {FRM, M(f_i)} -> FRM = M(ind(P)) = min(FRM, M(f_i))
                rel_best = self._relation(X[:, best_new_idx])
                FRM = np.minimum(FRM, rel_best)
                
                # Update current (Line 22)
                current_score = best_new_score
            else:
                break

        self.selected_indices_ = np.array(RS)
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]
