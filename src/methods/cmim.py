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
    CMIM Implementation (Fast Version with Discretization).
    Memilih fitur yang memaksimalkan I(f; Y | Selected).
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

    def _joint_entropy(self, x, y):
        """Hitung Joint Entropy H(X, Y) dengan label encoding cepat"""
        # Trik cepat: buat label tunggal dari (x, y)
        # Asumsi x dan y adalah integer kecil (discretized)
        combined = x + (np.max(x) + 1) * y
        return self._entropy(combined)

    def _conditional_mi(self, f_cand, target, f_cond):
        """
        Hitung I(f_cand; target | f_cond)
        Rumus: H(f_cand, f_cond) + H(target, f_cond) - H(f_cand, target, f_cond) - H(f_cond)
        """
        # H(f_cond)
        h_z = self._entropy(f_cond)
        # H(f_cand, f_cond)
        h_xz = self._joint_entropy(f_cand, f_cond)
        # H(target, f_cond)
        h_yz = self._joint_entropy(target, f_cond)
        
        # H(f_cand, target, f_cond) -> Combine 3 variables
        # Combine f_cand dan target dulu
        combined_xy = f_cand + (np.max(f_cand) + 1) * target
        h_xyz = self._joint_entropy(combined_xy, f_cond)

        return h_xz + h_yz - h_xyz - h_z

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_keep = int(np.ceil(n_features * self.ratio))
        n_keep = max(1, n_keep)
        
        # 1. Discretize Data (Wajib untuk kecepatan CMIM)
        # Menggunakan simple binning agar perhitungan Entropi cepat (O(N) bukan O(N^2))
        est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform', subsample=None)
        try:
            X_disc = est.fit_transform(X).astype(int)
        except:
            # Fallback jika gagal (misal data konstan)
            X_disc = X.astype(int)

        # Pastikan y integer
        if not np.issubdtype(y.dtype, np.integer):
            y = LabelEncoder().fit_transform(y)

        # 2. Inisialisasi: Hitung MI(X; Y) untuk semua fitur (Skor awal MIM)
        # CMIM Score awal adalah MI standar
        scores = np.zeros(n_features)
        for i in range(n_features):
            scores[i] = mutual_info_score(X_disc[:, i], y)
            
        selected = []
        # Mask untuk fitur yang belum terpilih
        current_mask = np.ones(n_features, dtype=bool) 
        
        # 3. Iterative Selection
        for _ in range(n_keep):
            # Cari fitur dengan skor terbaik di antara yang belum terpilih
            valid_indices = np.where(current_mask)[0]
            if len(valid_indices) == 0: break
            
            best_idx_local = np.argmax(scores[valid_indices])
            best_feat = valid_indices[best_idx_local]
            
            selected.append(best_feat)
            current_mask[best_feat] = False
            
            # 4. Update Score untuk fitur sisa (Conditional update)
            # Score_j = min(Score_j, I(X_j; Y | X_newly_selected))
            # Optimization: Fleuret's Fast CMIM logic
            # Kita update score terhadap fitur yang BARU saja dipilih (best_feat)
            
            f_sel_data = X_disc[:, best_feat]
            
            # Loop hanya untuk fitur yang belum terpilih
            remaining_feats = np.where(current_mask)[0]
            
            for f_idx in remaining_feats:
                # Hitung Conditional MI terhadap fitur yang baru masuk
                cond_mi = self._conditional_mi(X_disc[:, f_idx], y, f_sel_data)
                
                # CMIM criterion: ambil minimum dari (skor lama, cond_mi baru)
                # Artinya: informasi fitur ini "dibatasi" oleh fitur yang paling mirip dengannya di set terpilih
                scores[f_idx] = min(scores[f_idx], cond_mi)

        self.selected_indices_ = np.array(selected)
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]
