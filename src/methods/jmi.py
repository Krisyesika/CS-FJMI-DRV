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
    JMI Implementation (Fast Version with Discretization).
    Score(f) = Sum_{s in Selected} I(f, s; Y)
    """
    def __init__(self, ratio=0.5, n_bins=3, random_state=42):
        self.ratio = ratio
        self.n_bins = n_bins
        self.random_state = random_state
        self.selected_indices_ = []

    def _entropy(self, values):
        """Hitung Shannon Entropy H(X)"""
        counts = np.bincount(values)
        probs = counts[counts > 0] / len(values)
        return -np.sum(probs * np.log2(probs))

    def _joint_entropy_2(self, x, y):
        """Hitung H(X, Y) dengan integer packing"""
        # x + (max_x + 1) * y
        combined = x + (np.max(x) + 1) * y
        return self._entropy(combined)

    def _joint_entropy_3(self, x, z, y):
        """Hitung H(X, Z, Y) dengan integer packing"""
        mult1 = np.max(x) + 1
        mult2 = mult1 * (np.max(z) + 1)
        combined = x + mult1 * z + mult2 * y
        return self._entropy(combined)

    def _calculate_joint_mutual_info(self, f_cand, f_sel, y, h_y):
        """
        Hitung I(f_cand, f_sel; Y)
        Rumus: H(f_cand, f_sel) + H(Y) - H(f_cand, f_sel, Y)
        """
        h_xz = self._joint_entropy_2(f_cand, f_sel)
        h_xzy = self._joint_entropy_3(f_cand, f_sel, y)
        return h_xz + h_y - h_xzy

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_keep = int(np.ceil(n_features * self.ratio))
        n_keep = max(1, n_keep)
        
        # 1. Discretize Data (Sama seperti CMIM, wajib untuk kecepatan)
        est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform', subsample=None)
        try:
            X_disc = est.fit_transform(X).astype(int)
        except:
            X_disc = X.astype(int)

        # Pastikan y integer
        if not np.issubdtype(y.dtype, np.integer):
            y = LabelEncoder().fit_transform(y)
            
        # Precompute H(Y)
        h_y = self._entropy(y)

        # 2. Inisialisasi: Pilih fitur pertama berdasarkan MI(X; Y)
        mi_scores = np.zeros(n_features)
        for i in range(n_features):
            mi_scores[i] = mutual_info_score(X_disc[:, i], y)
            
        selected = []
        mask = np.ones(n_features, dtype=bool) 
        
        # Fitur pertama: Max MI
        best_idx = np.argmax(mi_scores)
        selected.append(best_idx)
        mask[best_idx] = False
        
        # Array untuk menyimpan akumulasi skor JMI
        # JMI(X_k) = Sum_{j in S} I(X_k, X_j; Y)
        # Kita mulai dengan array kosong, karena JMI adalah sum pairwise.
        # Saat fitur pertama dipilih, kita tambahkan I(X_k, X_first; Y) ke skor.
        jmi_scores = np.zeros(n_features)

        # 3. Iterative Selection (Incremental Update)
        for _ in range(n_keep - 1):
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0: break
            
            # Fitur yang baru saja terpilih
            last_selected_feat = X_disc[:, selected[-1]]
            
            # Update skor JMI hanya menambahkan term baru: I(X_cand, X_new; Y)
            for idx in valid_indices:
                cand_data = X_disc[:, idx]
                joint_mi_term = self._calculate_joint_mutual_info(cand_data, last_selected_feat, y, h_y)
                jmi_scores[idx] += joint_mi_term
            
            # Pilih fitur dengan total skor JMI tertinggi
            best_idx_local = np.argmax(jmi_scores[valid_indices])
            best_feat = valid_indices[best_idx_local]
            
            selected.append(best_feat)
            mask[best_feat] = False

        self.selected_indices_ = np.array(selected)
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]
