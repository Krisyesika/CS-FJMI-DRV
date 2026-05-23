"""
Shared utility functions used across all feature selection methods.
"""
import os
import numpy as np
import pandas as pd
import scipy.io
from collections import Counter
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
from scipy import sparse

RANDOM_STATE = 42

def load_mat_data(file_path):
    """Loader file .mat (Gene Expression)."""
    try:
        mat = scipy.io.loadmat(file_path)
    except Exception as e:
        print(f"[ERROR] Gagal membaca .mat: {e}")
        return None, None
    
    X_raw, y_raw = None, None
    for key in mat:
        if key.upper() == 'X': X_raw = mat[key]
        if key.upper() == 'Y': y_raw = mat[key]
        
    if X_raw is None: 
        keys = [k for k in mat.keys() if not k.startswith('__')]
        if len(keys) >= 2: X_raw, y_raw = mat[keys[0]], mat[keys[1]]

    if X_raw is None or y_raw is None: return None, None
    if y_raw.ndim > 1: y_raw = y_raw.flatten()
        
    n_features = X_raw.shape[1]
    feat_names = [f"Gene_{i}" for i in range(n_features)]
    
    X_df = pd.DataFrame(X_raw, columns=feat_names)
    if not np.issubdtype(y_raw.dtype, np.number):
        y_raw = LabelEncoder().fit_transform(y_raw)
    
    print(f"   [INFO] MAT Loaded. Shape: {X_df.shape}. Classes: {np.unique(y_raw)}")
    return X_df, y_raw

def calculate_merit(X, y, selected_features):
    if selected_features is None or len(selected_features) == 0: return 0.0
    X_sel = X[:, selected_features]
    if sparse.issparse(X_sel): X_sel = X_sel.toarray()
    
    var = np.var(X_sel, axis=0)
    keep = np.where(var > 0)[0]
    if keep.size == 0: return 0.0
    X_sel = X_sel[:, keep]
    k = X_sel.shape[1]
    
    corr = np.corrcoef(X_sel, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    
    if k > 1:
        iu = np.triu_indices(k, k=1)
        r_ff = float(np.mean(corr[iu])) if iu[0].size > 0 else 0.0
    else: r_ff = 0.0

    try:
        mi = mutual_info_classif(X_sel, y, discrete_features=False, random_state=RANDOM_STATE)
        r_fc = float(np.mean(mi))
    except: r_fc = 0.0

    denom = np.sqrt(k + k * (k - 1) * max(r_ff, 0.0))
    return (k * r_fc) / denom if denom > 0 else 0.0

def kuncheva_index(subsets, M: int) -> float:
    subsets = [set(s) for s in subsets if len(s) > 0]
    if len(subsets) < 2: return np.nan
    ks = {len(s) for s in subsets}
    k = ks.pop() if len(ks) == 1 else int(np.mean([len(s) for s in subsets]))
    
    if M <= 0 or k == 0: return np.nan
    overlaps = [len(a & b) for a, b in combinations(subsets, 2)]
    r_bar = float(np.mean(overlaps))
    num = r_bar - (k*k)/M
    den = k - (k*k)/M
    if abs(den) < 1e-9: return np.nan
    return float(num/den)

def mean_pairwise_overlap(subsets):
    overlaps = [len(set(a) & set(b)) for a, b in combinations(subsets, 2)]
    return float(np.mean(overlaps)) if overlaps else np.nan

def save_result_incrementally(new_row_dict, file_path):
    clean_row = {}
    for k, v in new_row_dict.items():
        if isinstance(v, (np.integer, np.floating)): clean_row[k] = v.item()
        elif isinstance(v, str): clean_row[k] = v
        else: clean_row[k] = v
            
    df_new = pd.DataFrame([clean_row])
    if os.path.exists(file_path):
        try:
            with pd.ExcelFile(file_path, engine="openpyxl") as reader:
                if 'Results' in reader.sheet_names:
                    df_old = pd.read_excel(reader, sheet_name='Results')
                    df_combined = pd.concat([df_old, df_new], ignore_index=True)
                else: df_combined = df_new
        except: df_combined = df_new
    else: df_combined = df_new

    try:
        with pd.ExcelWriter(file_path, engine="openpyxl", mode='w') as writer:
            df_combined.to_excel(writer, index=False, sheet_name='Results')
    except Exception as e:
        print(f"Excel Save Error: {e}. Saving CSV.")
        df_combined.to_csv(file_path.replace(".xlsx", ".csv"), index=False)

def prune_rare_classes(X_df, y, min_count=2):
    cnt = Counter(y)
    keep = {c for c, n in cnt.items() if n >= min_count}
    mask = np.isin(y, list(keep))
    removed = {c:n for c,n in cnt.items() if c not in keep}
    return X_df.loc[mask].reset_index(drop=True), y[mask], removed

def detect_column_roles(X_df):
    cat_cols = X_df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = []
    for c in X_df.columns:
        if c in cat_cols: continue
        is_effectively_int = False
        if pd.api.types.is_integer_dtype(X_df[c]): is_effectively_int = True
        elif pd.api.types.is_float_dtype(X_df[c]):
            valid_vals = X_df[c].dropna()
            if len(valid_vals) > 0 and np.all(np.mod(valid_vals, 1) == 0): is_effectively_int = True
        if is_effectively_int and X_df[c].nunique() < 20: cat_cols.append(c)
        else: num_cols.append(c)
    return num_cols, cat_cols, []

class FixedColumnSelector(TransformerMixin, BaseEstimator):
    def __init__(self, idx=None): self.idx = idx
    def fit(self, X, y=None): return self
    def transform(self, X):
        if self.idx is None or len(self.idx) == 0: return np.empty((len(X), 0))
        return X[:, self.idx]

class SelectiveSMOTE(BaseEstimator):
    def __init__(self, random_state=42, balance_ratio_min=0.8, k_neighbors_max=5):
        self.random_state = random_state
        self.balance_ratio_min = balance_ratio_min
        self.k_neighbors_max = k_neighbors_max
        self.sampler = None
    def fit(self, X, y): return self
    def fit_resample(self, X, y):
        if sparse.issparse(X): X = X.toarray()
        cnt = Counter(y)
        if len(cnt) < 2: return X, y
        M = max(cnt.values())
        eligible = {c:M for c,n in cnt.items() if n >= 2 and n/M < self.balance_ratio_min}
        if not eligible: return X, y
        k = max(1, min(self.k_neighbors_max, min(cnt.values())-1))
        self.sampler = SMOTE(sampling_strategy=eligible, k_neighbors=k, random_state=self.random_state)
        return self.sampler.fit_resample(X, y)

def build_model_spaces():
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=3000)
    svm = SVC(probability=True, random_state=RANDOM_STATE)
    knn = KNeighborsClassifier()
    nb = GaussianNB()
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)

    def grid_wrap(base_grid):
        return [{**g, "imb": ["passthrough"]} for g in base_grid]

    return {
        "Random Forest": (rf, grid_wrap([{"clf__n_estimators": [100, 500], "clf__class_weight": [None, "balanced"]}])),
        "Logistic Regression": (lr, grid_wrap([{"clf__C": [0.1, 1, 10], "clf__solver": ["lbfgs"], "clf__class_weight": [None, "balanced"]}])),
        "SVM": (svm, grid_wrap([{"clf__C": [1, 10], "clf__kernel": ["rbf"], "clf__class_weight": [None, "balanced"]}])),
        "KNN": (knn, grid_wrap([{"clf__n_neighbors": [3, 5, 7], "clf__weights": ["uniform", "distance"]}])),
        "Naive Bayes": (nb, grid_wrap([{"clf__var_smoothing": [1e-9, 1e-8]}])),
        "Decision Tree": (dt, grid_wrap([{"clf__max_depth": [5, 10, None], "clf__min_samples_leaf": [1, 5], "clf__class_weight": [None, "balanced"]}])),
    }

# 4. EXECUTION LOOP
