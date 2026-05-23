"""
Feature selection method implementations.
Each class follows the sklearn BaseEstimator/TransformerMixin interface.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.metrics import mutual_info_score

class FJMIURFeatureSelector(BaseEstimator, TransformerMixin):
    """
    FJMIUR Implementation based on Author's Snippet & Paper.
    Phase 1: ISUR (Instance Selection) to reduce samples to uncertainty region.
    Phase 2: FJMI with Auto-Threshold on the reduced dataset.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.selected_indices_ = []
        self.selected_instance_count_ = 0

    # ISUR Helper Functions
    def _distfunc(self, X, Y):
        """
        Calculates distance and selects indices in Y closest to X.
        Equivalent to: sel_indx = distfunc(mindata, maxdata)
        """
        # cdist computes distance between each pair of the two collections of inputs.
        # D shape: (len(X), len(Y))
        D = cdist(X, Y, metric='euclidean')
        # For each x in X (minority), find index of closest y in Y (majority)
        selected = np.argmin(D, axis=1)
        return selected

    def _ISUR(self, X, y):
        """
        Instance Selection based on Uncertainty Region.
        Logic: Keep all minority samples. Select nearest majority samples.
        """
        classes = np.unique(y)
        if len(classes) < 2:
            return X, y # Cannot select if only 1 class

        # Identify minority class
        class_counts = [np.sum(y == c) for c in classes]
        min_cls_idx = np.argmin(class_counts)
        min_cls_label = classes[min_cls_idx]

        # Split data
        mask_min = (y == min_cls_label)
        X_min = X[mask_min]
        y_min = y[mask_min]
        
        mask_max = ~mask_min
        X_max = X[mask_max]
        y_max = y[mask_max]

        if len(X_max) == 0: return X, y

        # Select majority instances closest to minority instances
        # "sel_indx = distfunc(mindata[:, :-1], maxdata[:, :-1])"
        sel_indices = self._distfunc(X_min, X_max)
        
        X_max_sel = X_max[sel_indices]
        y_max_sel = y_max[sel_indices]

        # Combine
        X_new = np.vstack((X_min, X_max_sel))
        y_new = np.hstack((y_min, y_max_sel))
        
        return X_new, y_new

    # FJMI Helper Functions (Same as FJMI)
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
        # I(f; SN; C) -> Interaction Information
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
        # 1. Apply ISUR to reduce instances
        X_reduced, y_reduced = self._ISUR(X, y)
        self.selected_instance_count_ = X_reduced.shape[0]
        
        # 2. Apply FJMI on reduced data
        self._fjmi_logic(X_reduced, y_reduced)
        return self

    def _fjmi_logic(self, X, y):
        n_features = X.shape[1]
        
        # Initialization
        RS = []
        FRM = None # SN in snippet
        current_val = 0.0
        best_vals = []
        
        # Mask for available features
        available = np.ones(n_features, dtype=bool)
        
        # Precompute Class Relation
        rel_C = self._decision(y)
        
        # --- First Feature Selection (Max I(f; C)) ---
        max_mi = -np.inf
        first_feature = -1
        
        # Loop all features to find first
        for i in range(n_features):
            rel_f = self._relation(X[:, i])
            i_cx = self._F_MI(rel_f, rel_C)
            if i_cx > max_mi:
                max_mi = i_cx
                first_feature = i
        
        if first_feature == -1: # Should not happen unless empty
            return

        RS.append(first_feature)
        best_vals.append(max_mi)
        available[first_feature] = False
        
        # Initialize SN (FRM)
        FRM = self._relation(X[:, first_feature])
        
        # --- Forward Selection Loop (Auto Threshold) ---
        # "for n in range(1, data.shape[1] - 1)"
        for _ in range(1, n_features):
            max_IcSNx = -np.inf # Max Gain
            best_feature_next = -1
            
            candidates = np.where(available)[0]
            if len(candidates) == 0: break
            
            # Constant for this iter: I(SN; C)
            I_CSN = self._F_MI(FRM, rel_C)
            
            for i in candidates:
                rel_f = self._relation(X[:, i])
                
                # I(f; SN; C) -> Interaction
                I_SNxc = self._iF_MI(rel_f, FRM, rel_C)
                
                # I(f; C)
                I_Cx = self._F_MI(rel_f, rel_C)
                
                # JRes = I(f;C) + I(SN;C) - I(f;SN;C)
                JRes = I_Cx + I_CSN - I_SNxc
                
                if JRes > max_IcSNx:
                    max_IcSNx = JRes
                    best_feature_next = i
            
            # --- Auto-Threshold Stopping Condition ---
            # "if max_IcSNx <= best_val[-1]: break"
            if max_IcSNx <= best_vals[-1]:
                break
                
            # Update State
            RS.append(best_feature_next)
            best_vals.append(max_IcSNx)
            available[best_feature_next] = False
            
            # Update SN: "SN = np.minimum(SN, relation(data[:, bestFeature]))"
            rel_next = self._relation(X[:, best_feature_next])
            FRM = np.minimum(FRM, rel_next)
            
        self.selected_indices_ = np.array(RS)

    def transform(self, X):
        return X[:, self.selected_indices_]
