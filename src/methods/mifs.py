"""
Feature selection method implementations.
Each class follows the sklearn BaseEstimator/TransformerMixin interface.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.metrics import mutual_info_score

class MIFSFeatureSelector(BaseEstimator, TransformerMixin):
    """
    MIFS (Mutual Information Feature Selection) Implementation.
    Criterion: Maximize [ I(f;y) - beta * Sum(I(f;s)) ]
    Reference: Battiti, R. (1994). Using mutual information for selecting features...
    """
    def __init__(self, ratio=0.5, n_bins=3, beta=0.5, random_state=42):
        self.ratio = ratio
        self.n_bins = n_bins
        self.beta = beta
        self.random_state = random_state
        self.selected_indices_ = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_keep = int(np.ceil(n_features * self.ratio))
        n_keep = max(1, n_keep)
        
        # 1. Discretize Data (Wajib untuk kecepatan perhitungan Entropy/MI)
        est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform', subsample=None)
        try:
            X_disc = est.fit_transform(X).astype(int)
        except:
            X_disc = X.astype(int)

        # Pastikan y integer
        if not np.issubdtype(y.dtype, np.integer):
            y = LabelEncoder().fit_transform(y)
            
        # 2. Hitung Relevance: I(X_i; Y) untuk semua fitur
        relevance = np.zeros(n_features)
        for i in range(n_features):
            relevance[i] = mutual_info_score(X_disc[:, i], y)
            
        selected = []
        mask = np.ones(n_features, dtype=bool) 
        
        # Fitur pertama: Fitur dengan Relevance tertinggi (Max I(X;Y))
        best_idx = np.argmax(relevance)
        selected.append(best_idx)
        mask[best_idx] = False
        
        # Array untuk menyimpan akumulasi Redundancy
        # Redundancy(f) = Sum_{s in S} I(f; s)
        redundancy_accum = np.zeros(n_features)

        # 3. Iterative Selection
        for _ in range(n_keep - 1):
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0: break
            
            # Fitur yang baru saja terpilih
            last_selected_feat = X_disc[:, selected[-1]]
            
            # Update Redundancy: Tambahkan I(f_cand; last_selected) ke akumulasi
            for idx in valid_indices:
                cand_data = X_disc[:, idx]
                # Hitung MI antara kandidat dan fitur terakhir yg dipilih
                mi_vals = mutual_info_score(cand_data, last_selected_feat)
                redundancy_accum[idx] += mi_vals
            
            # Hitung Skor MIFS
            # Score = Relevance - beta * Sum(Redundancy)
            # Perbedaan dengan mRMR: MIFS TIDAK membagi dengan |S|
            
            mifs_scores = relevance[valid_indices] - (self.beta * redundancy_accum[valid_indices])
            
            # Pilih fitur dengan skor MIFS tertinggi
            best_idx_local = np.argmax(mifs_scores)
            best_feat = valid_indices[best_idx_local]
            
            selected.append(best_feat)
            mask[best_feat] = False

        self.selected_indices_ = np.array(selected)
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]
