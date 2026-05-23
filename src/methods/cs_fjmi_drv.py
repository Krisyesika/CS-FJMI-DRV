"""
CS-FJMI-DRV v2.1 — Class-Specific Fuzzy Joint Mutual Information
with Dynamic Representative Vectors.

Key innovations:
  - DRV: Dynamic Representative Vectors (O(N×K) vs O(N²))
  - Average RM accumulation (stable, non-monotone-aware)
  - Adaptive stopping rule (relative improvement threshold)

Reference:
  [Author(s)]. "CS-FJMI-DRV: ..." Knowledge-Based Systems, 2025.
"""
import numpy as np
from collections import Counter
from itertools import combinations
from sklearn.cluster import MiniBatchKMeans

_EPS = 1e-12
RANDOM_STATE = 42

# DRV parameters (override via cs_fjmi_drv_select arguments)
DRV_RATIO = 0.05
DRV_MIN_K = 5
DRV_MAX_K = 50

def calculate_fuzzy_relation(feature_vec, reps_vec, gamma=1.0):
    """
    Gaussian fuzzy similarity: μ(x_i, r_j) = exp(-γ ‖x_i - r_j‖²)
    Complexity: O(N × K)
    """
    diff = feature_vec[:, np.newaxis] - reps_vec[np.newaxis, :]
    return np.exp(-gamma * (diff ** 2))


def fuzzy_entropy(relation_matrix):
    """
    Fuzzy entropy: H(R) = (1/N) Σᵢ log₂(K / |[xᵢ]_R|)
    Normalisasi K karena equivalence class didefinisikan terhadap K prototypes.
    """
    N, K = relation_matrix.shape
    equiv_sizes = np.sum(relation_matrix, axis=1)
    equiv_sizes = np.clip(equiv_sizes, _EPS, None)
    entropy = (1.0 / N) * np.sum(np.log2(K / equiv_sizes))
    return max(0.0, float(entropy))


def fuzzy_mutual_information(rel_f, rel_c):
    """
    Fuzzy Mutual Information: I(F; C) = H(F) + H(C) - H(F,C)
    Joint relation via minimum t-norm (Gödel).
    """
    H_f  = fuzzy_entropy(rel_f)
    H_c  = fuzzy_entropy(rel_c)
    H_fc = fuzzy_entropy(np.minimum(rel_f, rel_c))
    return max(0.0, H_f + H_c - H_fc)


# UPDATE RM — v2.0 FIX

def update_rm(RM_current, rel_new, iteration, mode='average'):
    """
    Update Representative Matrix berdasarkan mode.
    'average' (DIREKOMENDASIKAN): running average — stabil, proporsional.
    'minimum' (v1.0 LEGACY): monotone decreasing — bermasalah.
    'product': gradual decay, intermediate.
    """
    if mode == 'minimum':
        return np.minimum(RM_current, rel_new)
    elif mode == 'average':
        return (RM_current * (iteration - 1) + rel_new) / iteration
    elif mode == 'product':
        return RM_current * rel_new
    else:
        raise ValueError(f"RM_MODE tidak dikenal: '{mode}'")


# [v2.1] STOPPING RULE FUNCTION

def should_stop(best_score, prev_score, stop_mode='adaptive', delta=0.01):
    """
    Evaluasi apakah seleksi fitur harus berhenti.

    PERBANDINGAN MODE:
    ┌──────────────┬──────────────────────────────────────────────────────┐
    │ stop_mode    │ Kondisi berhenti                                     │
    ├──────────────┼──────────────────────────────────────────────────────┤
    │ 'adaptive'   │ best_score < prev_score * (1 - delta)               │
    │              │ Mengizinkan fluktuasi kecil, cocok untuk average RM  │
    ├──────────────┼──────────────────────────────────────────────────────┤
    │ 'classic'    │ best_score <= prev_score                            │
    │              │ FJMI klasik — terlalu agresif untuk average RM       │
    ├──────────────┼──────────────────────────────────────────────────────┤
    │ 'none'       │ Tidak pernah berhenti (return False)                │
    │              │ Selalu pilih T_max fitur                            │
    └──────────────┴──────────────────────────────────────────────────────┘

    Args:
        best_score : float — skor terbaik iterasi saat ini
        prev_score : float — skor terbaik iterasi sebelumnya
        stop_mode  : str
        delta      : float — relative tolerance (hanya untuk 'adaptive')

    Returns:
        bool — True jika harus berhenti
    """
    if stop_mode == 'none':
        return False
    elif stop_mode == 'classic':
        return best_score <= prev_score
    elif stop_mode == 'adaptive':
        # Relative improvement threshold:
        # Berhenti jika penurunan relatif melebihi delta.
        # Jika prev_score ≈ 0, gunakan absolute check untuk safety.
        if prev_score < _EPS:
            return best_score < _EPS
        return best_score < prev_score * (1.0 - delta)
    else:
        raise ValueError(f"STOP_MODE tidak dikenal: '{stop_mode}'")


# DYNAMIC REPRESENTATIVE VECTORS (DRV)

def generate_dynamic_representatives(X, y, target_class,
                                     ratio=0.05, min_k=5, max_k=50):
    """
    DRV: Class-conditional clustering → contrastive prototypes.
    Mengurangi kompleksitas dari O(N²) → O(N×K).
    """
    X_pos = X[y == target_class]
    X_neg = X[y != target_class]

    def get_reps(X_sub, max_k_sub):
        n = len(X_sub)
        k = max(min_k, min(int(n * ratio), max_k_sub))
        if n <= k:
            return X_sub
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE,
                             n_init='auto', batch_size=min(256, n))
        km.fit(X_sub)
        return km.cluster_centers_

    reps_pos = get_reps(X_pos, max_k)
    reps_neg = get_reps(X_neg, max_k)

    R   = np.vstack([reps_pos, reps_neg])
    y_R = np.concatenate([np.ones(len(reps_pos)), np.zeros(len(reps_neg))])

    return R, y_R


# CS-FJMI-DRV SELECTOR — v2.1 WITH ADAPTIVE STOPPING

def cs_fjmi_drv_select(X_train, y_train,
                        n_feats_per_class=10,
                        gamma=1.0,
                        rm_mode='average',
                        stop_mode='adaptive',
                        delta=0.01,
                        verbose=False):
    """
    CS-FJMI-DRV v2.1: Class-Specific Feature Selection dengan:
      - DRV (complexity reduction)
      - Average RM accumulation (stability)
      - Adaptive stopping rule (relative improvement threshold)

    [v2.1 PERUBAHAN UTAMA — STOPPING RULE]:

    FJMI klasik: if best_score <= prev_score → break
      Ini bekerja karena minimum t-norm → skor monotone decreasing.

    CS-FJMI-DRV (average RM): skor bersifat NON-MONOTONE.
      Classical stopping terlalu agresif — satu fluktuasi kecil
      bisa menghentikan seleksi prematur.

    Solusi: Relative improvement threshold
      if best_score < prev_score * (1 - delta) → break
      Default delta=0.01 (1%) mengizinkan fluktuasi minor
      sambil mendeteksi genuine diminishing returns.

    Args:
        X_train          : (N, d)
        y_train          : (N,)
        n_feats_per_class: int — T_max
        gamma            : float (FIXED = 1.0)
        rm_mode          : str — 'average'/'minimum'/'product'
        stop_mode        : str — 'adaptive'/'classic'/'none'   [v2.1]
        delta            : float — relative tolerance           [v2.1]
        verbose          : bool

    Returns:
        selected_union     : list[int]
        selected_per_class : dict{int: list[int]}
        stop_info          : dict — per-class stopping diagnostics [v2.1]
    """
    X_calc = np.nan_to_num(np.asarray(X_train, dtype=float))
    y      = np.asarray(y_train, dtype=int)

    # Hapus fitur konstan
    valid_mask    = np.var(X_calc, axis=0) > 1e-9
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        if verbose: print("[WARN] Semua fitur konstan!")
        return [0], {c: [0] for c in np.unique(y)}, {}

    X_clean    = X_calc[:, valid_indices]
    n_features = X_clean.shape[1]
    classes    = np.unique(y)

    if verbose:
        print(f"Fitur valid : {n_features}/{X_calc.shape[1]}")
        print(f"Kelas       : {classes}")
        print(f"RM Mode     : {rm_mode} | Stop: {stop_mode} | Delta: {delta}")

    selected_per_class = {}
    stop_info          = {}   # [v2.1] Diagnostics per kelas

    for c in classes:
        if verbose: print(f"\n  [Kelas {c}] Memulai seleksi...")

        n_c = np.sum(y == c)
        if n_c < DRV_MIN_K:
            if verbose: print(f"    [WARN] Hanya {n_c} sampel, fallback ke top-3 variance.")
            top3 = np.argsort(np.var(X_clean, axis=0))[-min(3, n_features):]
            selected_per_class[int(c)] = [valid_indices[i] for i in top3]
            stop_info[int(c)] = {'reason': 'rare_class_fallback', 'iterations': 0}
            continue

        # 1. Generate DRV
        R, y_R = generate_dynamic_representatives(
            X_clean, y, c, ratio=DRV_RATIO, min_k=DRV_MIN_K, max_k=DRV_MAX_K
        )
        K = R.shape[0]

        if verbose:
            print(f"    DRV: K={K} (pos={int(np.sum(y_R==1))}, neg={int(np.sum(y_R==0))})")

        # 2. Label-Induced Class Relation
        y_binary = (y == c).astype(float)
        Rel_C    = np.zeros((len(X_clean), K))
        for j, lj in enumerate(y_R):
            Rel_C[:, j] = (y_binary == lj).astype(float)

        # 3. Precompute Feature Relations
        Rel_F = [
            calculate_fuzzy_relation(X_clean[:, i], R[:, i], gamma=gamma)
            for i in range(n_features)
        ]

        # 4. Pilih Fitur Pertama: argmax I(F; C)
        first_scores = np.array([fuzzy_mutual_information(Rel_F[i], Rel_C)
                                  for i in range(n_features)])
        best_idx  = int(np.argmax(first_scores))
        S         = [best_idx]
        S_mask    = np.zeros(n_features, dtype=bool)
        S_mask[best_idx] = True

        # Inisialisasi RM
        RM = Rel_F[best_idx].copy()

        # [v2.1] Inisialisasi stopping rule
        
        prev_score   = first_scores[best_idx]  # Skor fitur pertama
        stop_reason  = 'budget_reached'         # Default: T_max tercapai
        actual_iters = 1

        if verbose:
            print(f"    Fitur ke-1: idx={best_idx} | score={prev_score:.6f}")

        # 5. Iterasi Seleksi
        target_k = min(n_feats_per_class, n_features)

        for iteration in range(2, target_k + 1):
            best_score = -np.inf
            best_cand  = -1

            for i in range(n_features):
                if S_mask[i]:
                    continue
                Rel_joint = np.minimum(Rel_F[i], RM)
                score     = fuzzy_mutual_information(Rel_joint, Rel_C)

                if score > best_score:
                    best_score = score
                    best_cand  = i

            if best_cand == -1:
                stop_reason = 'no_candidate'
                if verbose: print(f"    [STOP] Tidak ada kandidat di iterasi {iteration}")
                break

            # [v2.1] ADAPTIVE STOPPING RULE
            # 
            # Evaluasi apakah improvement masih signifikan.
            # Ini menggantikan kondisi best_idx == -1 yang tidak pernah
            # terpicu di v2.0.
            #
            # PENTING: Cek stopping SEBELUM menambahkan fitur ke S.
            # Jika stopping terpicu, fitur saat ini TIDAK ditambahkan.
            # 
            if should_stop(best_score, prev_score,
                           stop_mode=stop_mode, delta=delta):
                stop_reason = f'adaptive_stop(delta={delta})' if stop_mode == 'adaptive' \
                              else 'classic_stop'
                if verbose:
                    print(f"    [STOP-{stop_mode.upper()}] Iterasi {iteration}: "
                          f"score={best_score:.6f} < threshold "
                          f"(prev={prev_score:.6f}, "
                          f"thr={prev_score*(1-delta):.6f})")
                break

            # Fitur lolos stopping check → tambahkan
            S.append(best_cand)
            S_mask[best_cand] = True
            actual_iters = iteration

            # Update RM (average accumulation)
            RM = update_rm(RM, Rel_F[best_cand], iteration=iteration, mode=rm_mode)

            # Update prev_score untuk iterasi berikutnya
            prev_score = best_score

            if verbose:
                print(f"    Iter {iteration}: idx={best_cand} | "
                      f"score={best_score:.6f} | RM_mean={RM.mean():.4f}")

        # Map ke indeks original
        selected_per_class[int(c)] = [valid_indices[i] for i in S]

        # [v2.1] Simpan diagnostics
        stop_info[int(c)] = {
            'reason': stop_reason,
            'iterations': actual_iters,
            'features_selected': len(S),
            'budget': target_k,
            'final_score': float(prev_score)
        }

        if verbose:
            print(f"    Terpilih: {len(S)}/{target_k} fitur | Stop: {stop_reason}")

    # Union semua fitur per kelas
    selected_union = sorted(set(
        f for fs in selected_per_class.values() for f in fs
    ))
    if not selected_union:
        selected_union = [int(valid_indices[0])]

    if verbose:
        print(f"\n  [UNION] Total fitur unik: {len(selected_union)}")
        print(f"  [STOP SUMMARY]")
        for c_key, info in stop_info.items():
            print(f"    Kelas {c_key}: {info['features_selected']}/{info['budget']} "
                  f"fitur | {info['reason']}")

    return selected_union, selected_per_class, stop_info


# UTILITY FUNCTIONS (tidak berubah dari v2.0)
