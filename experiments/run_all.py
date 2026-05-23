"""
run_all.py — Reproduce all experiments from the paper.

Usage:
    python experiments/run_all.py                          # Run all methods
    python experiments/run_all.py --method CS-FJMI-DRV    # Run one method
    python experiments/run_all.py --dataset Arrhythmia     # Run one dataset
    python experiments/run_all.py --config experiments/config.yaml

Output:
    results/tables/Result_<METHOD>_<TIMESTAMP>.xlsx
"""

import os
import sys
import time
import json
import warnings
import argparse
from datetime import datetime
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.io
import yaml
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import (
    OrdinalEncoder, MinMaxScaler, LabelEncoder, label_binarize
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.metrics import geometric_mean_score
from imblearn.pipeline import Pipeline as ImbPipeline

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.methods import (
    CMIMFeatureSelector, FHFeatureSelector,
    FJMIFeatureSelector, FJMIIVFeatureSelector, FJMIURFeatureSelector,
    JMIFeatureSelector, MIFSFeatureSelector, MIMFeatureSelector,
    MRMRFeatureSelector, cs_fjmi_drv_select,
)
from src.utils import (
    load_mat_data, calculate_merit, kuncheva_index,
    mean_pairwise_overlap, save_result_incrementally,
    prune_rare_classes, detect_column_roles,
)

warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path="experiments/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Fixed Column Selector ─────────────────────────────────────────────────────

class FixedColumnSelector(TransformerMixin, BaseEstimator):
    """Selects pre-computed feature indices inside a sklearn Pipeline."""
    def __init__(self, idx=None):
        self.idx = idx

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.idx is None or len(self.idx) == 0:
            return np.empty((len(X), 0))
        return X[:, self.idx]


# ── Preprocessing ─────────────────────────────────────────────────────────────

def build_preprocessor(X_df, cfg):
    num_cols, cat_cols, _ = detect_column_roles(X_df)
    transformers = [("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", MinMaxScaler()),
    ]), num_cols)]
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols))
    return ColumnTransformer(transformers, remainder="drop")


# ── Classifiers ───────────────────────────────────────────────────────────────

def build_classifiers(cfg, random_state):
    clf_cfg = cfg["classifiers"]

    def wrap_grid(base_grid):
        return [{**g, "imb": ["passthrough"]} for g in base_grid]

    return {
        "Random Forest": (
            RandomForestClassifier(random_state=random_state, n_jobs=-1),
            wrap_grid([{
                "clf__n_estimators": clf_cfg["random_forest"]["n_estimators"],
                "clf__class_weight": clf_cfg["random_forest"]["class_weight"],
            }]),
        ),
        "SVM": (
            SVC(probability=True, random_state=random_state),
            wrap_grid([{
                "clf__C": clf_cfg["svm"]["C"],
                "clf__kernel": clf_cfg["svm"]["kernel"],
                "clf__class_weight": clf_cfg["svm"]["class_weight"],
            }]),
        ),
        "KNN": (
            KNeighborsClassifier(),
            wrap_grid([{
                "clf__n_neighbors": clf_cfg["knn"]["n_neighbors"],
                "clf__weights": clf_cfg["knn"]["weights"],
            }]),
        ),
    }


# ── Selector factory ──────────────────────────────────────────────────────────

def get_selector(method_name, cfg):
    b = cfg["baselines"]
    m = cfg["cs_fjmi_drv"]
    ratio   = b["selection_ratio"]
    n_bins  = b["n_bins"]
    rs      = cfg["experiment"]["random_seed"]

    selectors = {
        "MIM":   MIMFeatureSelector(ratio=ratio, n_bins=n_bins, random_state=rs),
        "MIFS":  MIFSFeatureSelector(ratio=ratio, n_bins=n_bins, beta=b["beta_mifs"], random_state=rs),
        "mRMR":  MRMRFeatureSelector(ratio=ratio, n_bins=n_bins, random_state=rs),
        "JMI":   JMIFeatureSelector(ratio=ratio, n_bins=n_bins, random_state=rs),
        "CMIM":  CMIMFeatureSelector(ratio=ratio, n_bins=n_bins, random_state=rs),
        "FH":    FHFeatureSelector(ratio=ratio, p=b["fh_p"], measure=b["fh_measure"], random_state=rs),
        "FJMI":  FJMIFeatureSelector(random_state=rs),
        "FJMI-IV":  FJMIIVFeatureSelector(measure=b["fjmiiv_measure"], p=b["fjmiiv_p"], random_state=rs),
        "FJMI-UR":  FJMIURFeatureSelector(random_state=rs),
    }
    return selectors.get(method_name)


# ── Main experiment loop ──────────────────────────────────────────────────────

def run_experiment(method_name, cfg, data_dir="data", out_dir="results/tables"):
    rs          = cfg["experiment"]["random_seed"]
    n_outer     = cfg["experiment"]["n_outer_folds"]
    n_inner     = cfg["experiment"]["n_inner_folds"]
    m_cfg       = cfg["cs_fjmi_drv"]
    is_proposed = (method_name == "CS-FJMI-DRV")

    np.random.seed(rs)
    os.makedirs(out_dir, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(out_dir, f"Result_{method_name}_{timestamp}.xlsx")

    print(f"\n{'='*65}")
    print(f"  Method : {method_name}")
    print(f"  Output : {output_file}")
    print(f"{'='*65}")

    classifiers = build_classifiers(cfg, rs)
    datasets    = cfg["datasets"]

    for ds_idx, (ds_name, ds_info) in enumerate(datasets.items(), 1):
        filepath = os.path.join(data_dir, ds_info["file"])
        if not os.path.exists(filepath):
            print(f"[{ds_idx:02d}/{len(datasets)}] SKIP {ds_name} — file not found: {filepath}")
            continue

        try:
            print(f"\n[{ds_idx:02d}/{len(datasets)}] {ds_name}")

            # Load data
            if filepath.endswith(".mat"):
                X_df, y_all = load_mat_data(filepath)
                removed_summary = "N/A"
            else:
                df = pd.read_excel(filepath)
                df.columns = df.columns.astype(str)
                target = ds_info.get("target", "class")
                if target not in df.columns:
                    print(f"  Column '{target}' not found. Available: {list(df.columns[:5])}")
                    continue
                df       = df.replace("?", np.nan).dropna(subset=[target])
                y_all    = df[target].values
                X_df     = df.drop(columns=[target]).apply(pd.to_numeric, errors="coerce")
                if not np.issubdtype(y_all.dtype, np.number):
                    y_all = LabelEncoder().fit_transform(y_all.astype(str))

            if X_df is None or len(np.unique(y_all)) < 2:
                print("  Invalid data — skipping.")
                continue

            X_df, y_all, removed_cls = prune_rare_classes(X_df, y_all)
            removed_summary = str(removed_cls) if removed_cls else "none"

            M_raw     = X_df.shape[1]
            n_classes = len(np.unique(y_all))
            n_samples = len(y_all)
            print(f"  Shape: {X_df.shape} | Classes: {n_classes} | Samples: {n_samples}")

            # Cross-validation
            cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=rs)

            for clf_name, (clf_base, param_grid) in classifiers.items():
                print(f"  {clf_name}")

                # Metric accumulators
                accs, f1s_mac, precs_mac, recs_mac = [], [], [], []
                f1s_w, precs_w, recs_w              = [], [], []
                bal_accs, gmeans, mccs              = [], [], []
                roc_mac, roc_mic, pr_mac, pr_mic    = [], [], [], []
                fs_times, total_times, merits       = [], [], []
                sel_unions = []

                for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X_df, y_all), 1):
                    t_start = time.time()
                    X_tr = X_df.iloc[tr_idx]
                    X_te = X_df.iloc[te_idx]
                    y_tr = y_all[tr_idx]
                    y_te = y_all[te_idx]

                    preprocessor = build_preprocessor(X_tr, cfg)
                    X_tr_t = preprocessor.fit_transform(X_tr, y_tr)
                    if sparse.issparse(X_tr_t):
                        X_tr_t = X_tr_t.toarray()

                    # Feature selection
                    t0_fs = time.time()
                    if is_proposed:
                        sel_idx, _, _ = cs_fjmi_drv_select(
                            X_tr_t, y_tr,
                            n_feats_per_class=m_cfg["n_feats_per_class"],
                            gamma=m_cfg["gamma"],
                            rm_mode=m_cfg["rm_mode"],
                            stop_mode=m_cfg["stop_mode"],
                            delta=m_cfg["delta"],
                        )
                    else:
                        selector = get_selector(method_name, cfg)
                        selector.fit(X_tr_t, y_tr)
                        sel_idx = list(selector.selected_indices_)
                    fs_time = time.time() - t0_fs
                    fs_times.append(fs_time)

                    sel_unions.append([int(x) + 1 for x in sel_idx])
                    merits.append(calculate_merit(X_tr_t[:, sel_idx], y_tr) if sel_idx else 0.0)

                    # Classification
                    pipeline = ImbPipeline([
                        ("pre", preprocessor),
                        ("sel", FixedColumnSelector(sel_idx)),
                        ("imb", "passthrough"),
                        ("clf", clf_base),
                    ])
                    gs = GridSearchCV(
                        pipeline, param_grid,
                        cv=StratifiedKFold(n_inner, shuffle=True, random_state=rs),
                        scoring="f1_macro", n_jobs=-1, error_score="raise",
                    )
                    try:
                        gs.fit(X_tr, y_tr)
                        final_model = CalibratedClassifierCV(gs.best_estimator_, method="sigmoid", cv="prefit")
                        final_model.fit(X_tr, y_tr)
                        y_pred = final_model.predict(X_te)
                        try:
                            y_prob = final_model.predict_proba(X_te)
                        except Exception:
                            y_prob = np.zeros((len(y_te), n_classes))

                        accs.append(accuracy_score(y_te, y_pred))
                        precs_mac.append(precision_score(y_te, y_pred, average="macro", zero_division=0))
                        recs_mac.append(recall_score(y_te, y_pred, average="macro", zero_division=0))
                        f1s_mac.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
                        precs_w.append(precision_score(y_te, y_pred, average="weighted", zero_division=0))
                        recs_w.append(recall_score(y_te, y_pred, average="weighted", zero_division=0))
                        f1s_w.append(f1_score(y_te, y_pred, average="weighted", zero_division=0))
                        bal_accs.append(balanced_accuracy_score(y_te, y_pred))
                        gmeans.append(geometric_mean_score(y_te, y_pred, average="macro"))
                        mccs.append(matthews_corrcoef(y_te, y_pred))

                        classes_ref = np.unique(y_all)
                        y_bin = label_binarize(y_te, classes=classes_ref)
                        if y_prob.shape[1] != len(classes_ref):
                            pad = np.zeros((len(y_prob), len(classes_ref) - y_prob.shape[1]))
                            y_prob = np.hstack([y_prob, pad])

                        for lst, fn, kw in [
                            (roc_mac, roc_auc_score, {"average": "macro", "multi_class": "ovr"}),
                            (roc_mic, roc_auc_score, {"average": "micro", "multi_class": "ovr"}),
                            (pr_mac,  average_precision_score, {"average": "macro"}),
                            (pr_mic,  average_precision_score, {"average": "micro"}),
                        ]:
                            try:
                                lst.append(fn(y_bin, y_prob, **kw))
                            except Exception:
                                lst.append(np.nan)

                        print(f"    Fold {fold_idx}: Acc={accs[-1]*100:.2f}%  "
                              f"F1={f1s_mac[-1]*100:.2f}%  "
                              f"Feats={len(sel_idx)}  FS={fs_time:.1f}s")

                    except Exception as e:
                        print(f"    Fold {fold_idx} failed: {e}")
                        for lst in [accs, precs_mac, recs_mac, f1s_mac, precs_w, recs_w,
                                    f1s_w, bal_accs, gmeans, mccs, roc_mac, roc_mic, pr_mac, pr_mic]:
                            lst.append(np.nan)

                    total_times.append(time.time() - t_start)

                # Aggregate
                def fmt(arr, pct=False):
                    clean = [x for x in arr if not np.isnan(x)]
                    if not clean:
                        return "N/A"
                    m, s = np.mean(clean), np.std(clean)
                    return f"{m*100:.2f} ± {s*100:.2f}" if pct else f"{m:.4f} ± {s:.4f}"

                sel_fold1     = sel_unions[0] if sel_unions else []
                reduction_pct = (1.0 - len(sel_fold1) / M_raw) * 100.0 if M_raw > 0 else 0.0
                kappa         = kuncheva_index(sel_unions, M=M_raw) if len(sel_unions) > 1 else np.nan
                mpo           = mean_pairwise_overlap(sel_unions) if len(sel_unions) > 1 else np.nan

                result_row = {
                    "Method":               method_name,
                    "Dataset":              ds_name,
                    "Classifier":           clf_name,
                    "Accuracy (%)":         fmt(accs, True),
                    "Precision_Macro":      fmt(precs_mac, True),
                    "Recall_Macro":         fmt(recs_mac, True),
                    "F1_Macro":             fmt(f1s_mac, True),
                    "Precision_Weighted":   fmt(precs_w, True),
                    "Recall_Weighted":      fmt(recs_w, True),
                    "F1_Weighted":          fmt(f1s_w, True),
                    "Balanced_Accuracy":    fmt(bal_accs, True),
                    "G-Mean":               fmt(gmeans),
                    "MCC":                  fmt(mccs),
                    "ROC_AUC_Macro":        fmt(roc_mac),
                    "ROC_AUC_Micro":        fmt(roc_mic),
                    "PR_AUC_Macro":         fmt(pr_mac),
                    "PR_AUC_Micro":         fmt(pr_mic),
                    "FS_Time_Avg (s)":      f"{np.mean(fs_times):.3f}",
                    "Total_Time (s)":       f"{np.sum(total_times):.3f}",
                    "Selected_Fold1":       len(sel_fold1),
                    "Initial_Features":     M_raw,
                    "Reduction (%)":        f"{reduction_pct:.2f}",
                    "Selected_Indices_Fold1": ", ".join(map(str, sel_fold1)),
                    "Merit_Mean":           fmt(merits),
                    "Kuncheva_Kappa":       f"{kappa:.4f}" if not np.isnan(kappa) else "N/A",
                    "Mean_Pairwise_Overlap": f"{mpo:.4f}" if not np.isnan(mpo) else "N/A",
                    "Removed_Classes":      removed_summary,
                    "Num_Classes":          n_classes,
                    "Num_Samples":          n_samples,
                }

                save_result_incrementally(result_row, output_file)

        except Exception as e:
            import traceback
            print(f"  ERROR on {ds_name}: {e}")
            traceback.print_exc()

    print(f"\n{'='*65}")
    print(f"  DONE — {method_name}")
    print(f"  Output: {output_file}")
    print(f"{'='*65}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

ALL_METHODS = [
    "MIM", "MIFS", "mRMR", "JMI", "CMIM",
    "FH", "FJMI", "FJMI-IV", "FJMI-UR",
    "CS-FJMI-DRV",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CS-FJMI-DRV experiments.")
    parser.add_argument("--config",  default="experiments/config.yaml", help="Path to config.yaml")
    parser.add_argument("--method",  default=None, choices=ALL_METHODS + ["all"], help="Method to run")
    parser.add_argument("--dataset", default=None, help="Single dataset name to run")
    parser.add_argument("--data_dir", default="data", help="Directory containing dataset files")
    parser.add_argument("--out_dir",  default="results/tables", help="Output directory for Excel results")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Filter datasets if requested
    if args.dataset:
        all_ds = cfg["datasets"]
        if args.dataset not in all_ds:
            print(f"Dataset '{args.dataset}' not found in config. Available: {list(all_ds.keys())}")
            sys.exit(1)
        cfg["datasets"] = {args.dataset: all_ds[args.dataset]}

    methods_to_run = ALL_METHODS if (args.method is None or args.method == "all") else [args.method]

    for method in methods_to_run:
        run_experiment(method, cfg, data_dir=args.data_dir, out_dir=args.out_dir)
