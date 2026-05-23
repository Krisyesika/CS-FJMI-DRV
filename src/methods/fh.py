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
    FH (Fuzzy Entropy) Feature Selection - Luukka's Original Logic.
    
    Logic:
    1. Calculate Ideal Vectors (Generalized Mean) for each class.
    2. Calculate Similarity of ALL samples to ALL Ideal Vectors.
    3. Calculate Entropy of these similarities.
    4. Remove features with HIGHEST entropy (Keep LOWEST).
    """
    def __init__(self, ratio=0.5, p=1.0, measure='luca', random_state=42):
        self.ratio = ratio
        self.p = p
        self.measure = measure
        self.random_state = random_state
        self.selected_indices_ = []

    def fit(self, X, y):
        # Input X sudah di-scale [0,1] oleh Pipeline, sesuai requirements FH.
        n_samples, n_features = X.shape
        n_keep = int(np.ceil(n_features * self.ratio))
        n_keep = max(1, n_keep)
        
        classes = np.unique(y)
        n_classes = len(classes)
        
        # 1. Calculate Ideal Vectors (Arithmetic Mean, p=1 default)
        # Sesuai skrip Luukka: idealvec_s[k, :] = data[idx].iloc[:, :-1].mean(axis=0)
        ideals = np.zeros((n_classes, n_features))
        for i, c in enumerate(classes):
            X_c = X[y == c]
            ideals[i, :] = np.mean(X_c, axis=0) # Asumsi m=1 (Arithmetic) sesuai default skrip
            
        # 2. Calculate Entropy per Feature (Vectorized for Speed)
        # Skrip asli melakukan 3 loop (samples, features, classes).
        # Kita vektorisasi agar cepat di Python.
        
        entropies = np.zeros(n_features)
        delta = 1e-10 # Sesuai skrip Luukka
        
        # Pre-compute powers if p != 1
        if self.p != 1.0:
            X_pow = X ** self.p
            ideals_pow = ideals ** self.p
        else:
            X_pow = X
            ideals_pow = ideals

        for f in range(n_features):
            # Ambil kolom fitur ke-f dari semua sampel (Shape: N_samples x 1)
            x_col = X_pow[:, f].reshape(-1, 1)
            
            # Ambil nilai ideal fitur ke-f untuk semua kelas (Shape: 1 x N_classes)
            v_row = ideals_pow[:, f].reshape(1, -1)
            
            # Broadcasting: Hitung selisih semua sampel vs semua ideal sekaligus
            # Result Shape: (N_samples, N_classes)
            # Formula PDF/Script: (1 - |v^p - x|^p)^(1/p) (Asumsi x di skrip typo tidak dipangkatkan, kita ikut PDF)
            # Namun skrip Luukka: (1 - abs(ideal**p - data)**p)**(1/p).
            # Ikuti logika PDF Eq 5 yang lebih umum: 1 - |x^p - v^p| untuk p=1.
            
            diff = np.abs(v_row - x_col) # |v^p - x^p| jika p sudah diapply
            
            if self.p == 1.0:
                sim = 1.0 - diff
            else:
                # Koreksi sesuai PDF Eq 5 generalized Lukasiewicz
                sim = (1.0 - diff) ** (1.0/self.p) 
            
            # Clip untuk keamanan numerik
            sim = np.clip(sim, 0.0, 1.0)
            
            # Flatten sim matrix untuk fitur ini (m * l values)
            sim_flat = sim.flatten()
            
            # 3. Calculate Entropy (Measure)
            if self.measure == 'luca':
                # De Luca & Termini
                # Handle log(0)
                sim_flat[sim_flat == 1] = 1 - delta
                sim_flat[sim_flat == 0] = delta
                
                H = -sim_flat * np.log(sim_flat) - (1 - sim_flat) * np.log(1 - sim_flat)
                
            elif self.measure == 'park':
                # Parkash et al.
                H = np.sin(np.pi / 2 * sim_flat) + np.sin(np.pi / 2 * (1 - sim_flat)) - 1
            
            # Sum entropy for this feature
            entropies[f] = np.sum(H)

        # 4. Select Features
        # Skrip Luukka: "Find maximum feature... Removing feature"
        # Artinya fitur dengan Entropy Tertinggi adalah fitur terburuk.
        # Kita ingin MENYIMPAN fitur terbaik (Entropy Terendah).
        
        # Sort ascending (kecil ke besar) dan ambil n_keep teratas
        self.selected_indices_ = np.argsort(entropies)[:n_keep]
        
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]
