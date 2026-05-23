"""
Feature selection method implementations.
Each class follows the sklearn BaseEstimator/TransformerMixin interface.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.metrics import mutual_info_score

class FJMIIVFeatureSelector(BaseEstimator, TransformerMixin):
    """
    FJMIIV Implementation - STRICT REPLICATION.
    Includes internal normdata logic (Class-dependent ideal vector construction).
    """
    def __init__(self, measure='luca', p=1, random_state=42):
        self.measure = measure
        self.p = p
        self.random_state = random_state
        self.selected_indices_ = []

    def _normdata_logic(self, X, y):
        """
        Replicates 'normdata' logic: Shift -> Ideal -> Normalize.
        """
        m, t = X.shape
        classes = np.unique(y) 
        l = len(classes)
        
        # 1. Shift Data
        mins_v = np.min(X, axis=0)
        shift = np.abs(mins_v)
        X_shifted = X + shift
        
        # 2. Calculate Ideal Vectors from SHIFTED Data
        idealvec_s = np.zeros((l, t))
        for k_idx, k in enumerate(classes):
            mask = (y == k)
            if np.sum(mask) > 0:
                idealvec_s[k_idx, :] = np.mean(X_shifted[mask], axis=0)
        
        # 3. Normalize by Max of SHIFTED Data
        maxs_v = np.max(X_shifted, axis=0)
        maxs_v[maxs_v == 0] = 1.0 # Safety
        
        X_norm = X_shifted / maxs_v
        ideal_norm = idealvec_s / maxs_v
        
        return X_norm, ideal_norm

    def _simRelation_logic(self, X, y):
        X_norm, ideal_vecs = self._normdata_logic(X, y)
        
        m, t = X_norm.shape
        l = ideal_vecs.shape[0]
        
        X_exp = X_norm[:, np.newaxis, :]
        Ideal_exp = ideal_vecs[np.newaxis, :, :]
        
        diff = np.abs(Ideal_exp**self.p - X_exp**self.p)
        sim_3d = (1 - diff**self.p)**(1/self.p)
        
        sim_tml = sim_3d.transpose(2, 0, 1)
        sim_reshaped = sim_tml.reshape(t, m * l, order='F')
        return sim_reshaped.T

    def _FH(self, sim):
        if self.measure == 'luca':
            delta = 1e-10
            s = np.clip(sim, delta, 1 - delta)
            H = np.sum(-s * np.log(s) - (1 - s) * np.log(1 - s))
        elif self.measure == 'park':
            H = np.sum(np.sin(np.pi / 2 * sim) + np.sin(np.pi / 2 * (1 - sim)) - 1)
        return H

    def _FFMI2(self, f1, f2, HC):
        H1 = self._FH(f1)
        f12 = np.minimum(f1, f2)
        H12 = self._FH(f12)
        if self.measure == 'luca': return H1 + HC - H12
        elif self.measure == 'park':
            H2 = self._FH(f2)
            return H1 + H2 - H12

    def _FMI3(self, f1, f2, f3, HC, ICX):
        f12 = np.minimum(f1, f2)
        f123 = np.minimum(f12, f3)
        if self.measure == 'luca':
            H2 = self._FH(f2)
            f23 = np.minimum(f2, f3)
            H23 = self._FH(f23)
            H12 = self._FH(f12) 
            H123 = self._FH(f123)
            return (ICX + H2 + H123) - (H12 + H23)
        elif self.measure == 'park':
            H1 = self._FH(f1); H2 = self._FH(f2); H3 = self._FH(f3)
            H12 = self._FH(f12); f13 = np.minimum(f1, f3); H13 = self._FH(f13)
            f23 = np.minimum(f2, f3); H23 = self._FH(f23); H123 = self._FH(f123)
            return (H1 + H2 + H3 + H123) - (H12 + H23 + H13)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_encoded = LabelEncoder().fit_transform(y)
        y_1based = y_encoded + 1
        
        sim = self._simRelation_logic(X, y_1based)
        
        y_col = y_1based.reshape(-1, 1)
        relc_matrix = self._simRelation_logic(y_col, y_1based)
        relc = relc_matrix[:, 0]
        
        HC = self._FH(relc)
        
        I_Cx = np.zeros(n_features)
        max_mi = -np.inf
        first_feature = -1
        
        for i in range(n_features):
            I_Cx[i] = self._FFMI2(sim[:, i], relc, HC)
            if I_Cx[i] > max_mi:
                max_mi = I_Cx[i]; first_feature = i
                
        best_fs = [first_feature]
        best_vals = [max_mi]
        SN = sim[:, first_feature]
        selected_mask = np.zeros(n_features, dtype=bool)
        selected_mask[first_feature] = True
        
        for n in range(2, n_features + 1):
            max_IcSNx = -np.inf
            best_feature_next = -1
            candidates = np.where(~selected_mask)[0]
            if len(candidates) == 0: break
            
            I_CSN = self._FFMI2(SN, relc, HC)
            
            for i in candidates:
                relxn = sim[:, i]
                I_SNxc = self._FMI3(relxn, SN, relc, HC, I_Cx[i])
                JRes = I_Cx[i] + I_CSN - I_SNxc
                if JRes > max_IcSNx:
                    max_IcSNx = JRes; best_feature_next = i
            
            if max_IcSNx <= best_vals[-1]: break
            best_fs.append(best_feature_next)
            best_vals.append(max_IcSNx)
            SN = np.minimum(SN, sim[:, best_feature_next])
            selected_mask[best_feature_next] = True
            
        self.selected_indices_ = np.array(best_fs)
        return self

    def transform(self, X):
        return X[:, self.selected_indices_] if len(self.selected_indices_) > 0 else np.empty((X.shape[0], 0))
