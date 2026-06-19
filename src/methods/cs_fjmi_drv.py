"""
CS-FJMI-DRV - Class-Specific Fuzzy Joint Mutual Information
with Dynamic Representative Vectors.

Key innovations:
  - DRV: Dynamic Representative Vectors (O(N*K) vs O(N^2))
  - Average RM accumulation (stable, non-monotone-aware)
  - Adaptive stopping rule (relative improvement threshold)

Reference:
  Krisyesika, Joko Lianto Buliali, Ahmad Saikhu. "CS-FJMI-DRV: Class-Specific Fuzzy Joint Mutual Information with Dynamic Representative Vectors", 2025.
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
    Gaussian fuzzy similarity: mu(x_i, r_j) = exp(-gamma * ||x_i - r_j||^2)
    Complexity: O(N x K)
    """
    diff = feature_vec[:, np.newaxis] - reps_vec[np.newaxis, :]
    return np.exp(-gamma * (diff ** 2))


def fuzzy_entropy(relation_matrix):
    """
    Fuzzy entropy: H(R) = (1/N) sum_i log2(K / |[x_i]_R|)
    K is used as the normalising constant since equivalence classes are
    defined relative to the K prototypes rather than the full sample.
    """
    N, K = relation_matrix.shape
    equiv_sizes = np.sum(relation_matrix, axis=1)
    equiv_sizes = np.clip(equiv_sizes, _EPS, None)
    entropy = (1.0 / N) * np.sum(np.log2(K / equiv_sizes))
    return max(0.0, float(entropy))


def fuzzy_mutual_information(rel_f, rel_c):
    """
    Fuzzy Mutual Information: I(F; C) = H(F) + H(C) - H(F,C)
    Joint relation via minimum t-norm (Goedel).
    """
    H_f  = fuzzy_entropy(rel_f)
    H_c  = fuzzy_entropy(rel_c)
    H_fc = fuzzy_entropy(np.minimum(rel_f, rel_c))
    return max(0.0, H_f + H_c - H_fc)


def update_rm(RM_current, rel_new, iteration, mode='average'):
    """
    Update the representative matrix according to the chosen mode.

    'average' (recommended) keeps a running average and stays stable as
    features are added. 'minimum' is the original FJMI rule and is
    monotone decreasing, which tends to over-penalise later features.
    'product' decays gradually and sits between the two.
    """
    if mode == 'minimum':
        return np.minimum(RM_current, rel_new)
    elif mode == 'average':
        return (RM_current * (iteration - 1) + rel_new) / iteration
    elif mode == 'product':
        return RM_current * rel_new
    else:
        raise ValueError(f"Unknown RM mode: '{mode}'")


def should_stop(best_score, prev_score, stop_mode='adaptive', delta=0.01):
    """
    Decide whether feature selection should stop for the current class.

    'classic' stops as soon as the score fails to improve, which is the
    rule used by standard FJMI. It assumes scores are monotone decreasing,
    which only holds for the 'minimum' RM mode. With 'average' RM the
    score can fluctuate slightly between iterations, so 'classic' tends
    to stop too early.

    'adaptive' instead stops only once the score drops by more than a
    relative tolerance `delta`, which tolerates small fluctuations while
    still catching genuine diminishing returns. This is the default for
    CS-FJMI-DRV since it always uses average RM accumulation.

    'none' disables early stopping and always selects up to the feature
    budget T_max.

    Args:
        best_score: best candidate score at the current iteration.
        prev_score: best score from the previous iteration.
        stop_mode: 'adaptive', 'classic', or 'none'.
        delta: relative tolerance used only by 'adaptive'.

    Returns:
        True if selection should stop.
    """
    if stop_mode == 'none':
        return False
    elif stop_mode == 'classic':
        return best_score <= prev_score
    elif stop_mode == 'adaptive':
        # If prev_score is ~0 there is nothing meaningful to compare
        # against, so fall back to an absolute check.
        if prev_score < _EPS:
            return best_score < _EPS
        return best_score < prev_score * (1.0 - delta)
    else:
        raise ValueError(f"Unknown stop mode: '{stop_mode}'")


def generate_dynamic_representatives(X, y, target_class,
                                     ratio=0.05, min_k=5, max_k=50):
    """
    Build a small set of class-conditional prototypes via clustering,
    contrasting the target class against the rest. This is what brings
    the complexity down from O(N^2) to O(N x K).
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


def cs_fjmi_drv_select(X_train, y_train,
                        n_feats_per_class=10,
                        gamma=1.0,
                        rm_mode='average',
                        stop_mode='adaptive',
                        delta=0.01,
                        verbose=False):
    """
    Class-specific feature selection combining DRV-based complexity
    reduction, average RM accumulation, and an adaptive stopping rule.

    For each class, features are selected greedily to maximise fuzzy
    joint mutual information with the class label, using a small set of
    dynamic representative vectors in place of the full pairwise
    similarity matrix.

    Args:
        X_train: (N, d) feature matrix.
        y_train: (N,) integer class labels.
        n_feats_per_class: feature budget T_max per class.
        gamma: Gaussian kernel width (fixed at 1.0 in the paper).
        rm_mode: representative matrix update rule, 'average'/'minimum'/'product'.
        stop_mode: stopping rule, 'adaptive'/'classic'/'none'.
        delta: relative tolerance used by the adaptive stopping rule.
        verbose: print per-class selection progress.

    Returns:
        selected_union: sorted list of unique selected feature indices.
        selected_per_class: dict mapping class label to its selected feature indices.
        stop_info: dict mapping class label to stopping diagnostics.
    """
    X_calc = np.nan_to_num(np.asarray(X_train, dtype=float))
    y      = np.asarray(y_train, dtype=int)

    # Drop constant features before scoring anything.
    valid_mask    = np.var(X_calc, axis=0) > 1e-9
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        if verbose: print("All features are constant, nothing to select.")
        return [0], {c: [0] for c in np.unique(y)}, {}

    X_clean    = X_calc[:, valid_indices]
    n_features = X_clean.shape[1]
    classes    = np.unique(y)

    if verbose:
        print(f"Valid features: {n_features}/{X_calc.shape[1]}")
        print(f"Classes: {classes}")
        print(f"RM mode: {rm_mode} | Stop mode: {stop_mode} | Delta: {delta}")

    selected_per_class = {}
    stop_info          = {}

    for c in classes:
        if verbose: print(f"\nClass {c}: starting selection")

        n_c = np.sum(y == c)
        if n_c < DRV_MIN_K:
            if verbose: print(f"  Only {n_c} samples, falling back to top-3 variance features.")
            top3 = np.argsort(np.var(X_clean, axis=0))[-min(3, n_features):]
            selected_per_class[int(c)] = [valid_indices[i] for i in top3]
            stop_info[int(c)] = {'reason': 'rare_class_fallback', 'iterations': 0}
            continue

        # Build dynamic representative vectors for this class.
        R, y_R = generate_dynamic_representatives(
            X_clean, y, c, ratio=DRV_RATIO, min_k=DRV_MIN_K, max_k=DRV_MAX_K
        )
        K = R.shape[0]

        if verbose:
            print(f"  DRV: K={K} (pos={int(np.sum(y_R==1))}, neg={int(np.sum(y_R==0))})")

        # Label-induced class relation against the representatives.
        y_binary = (y == c).astype(float)
        Rel_C    = np.zeros((len(X_clean), K))
        for j, lj in enumerate(y_R):
            Rel_C[:, j] = (y_binary == lj).astype(float)

        # Precompute the fuzzy relation for every feature once.
        Rel_F = [
            calculate_fuzzy_relation(X_clean[:, i], R[:, i], gamma=gamma)
            for i in range(n_features)
        ]

        # First feature: the one with the highest standalone MI score.
        first_scores = np.array([fuzzy_mutual_information(Rel_F[i], Rel_C)
                                  for i in range(n_features)])
        best_idx  = int(np.argmax(first_scores))
        S         = [best_idx]
        S_mask    = np.zeros(n_features, dtype=bool)
        S_mask[best_idx] = True

        RM = Rel_F[best_idx].copy()

        prev_score   = first_scores[best_idx]
        stop_reason  = 'budget_reached'
        actual_iters = 1

        if verbose:
            print(f"  Feature 1: idx={best_idx} | score={prev_score:.6f}")

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
                if verbose: print(f"  Stopping at iteration {iteration}: no candidate features left.")
                break

            if should_stop(best_score, prev_score,
                           stop_mode=stop_mode, delta=delta):
                stop_reason = f'adaptive_stop(delta={delta})' if stop_mode == 'adaptive' \
                              else 'classic_stop'
                if verbose:
                    print(f"  Stopping ({stop_mode}) at iteration {iteration}: "
                          f"score={best_score:.6f} below threshold "
                          f"(prev={prev_score:.6f}, "
                          f"threshold={prev_score*(1-delta):.6f})")
                break

            S.append(best_cand)
            S_mask[best_cand] = True
            actual_iters = iteration

            RM = update_rm(RM, Rel_F[best_cand], iteration=iteration, mode=rm_mode)
            prev_score = best_score

            if verbose:
                print(f"  Iter {iteration}: idx={best_cand} | "
                      f"score={best_score:.6f} | RM_mean={RM.mean():.4f}")

        selected_per_class[int(c)] = [valid_indices[i] for i in S]

        stop_info[int(c)] = {
            'reason': stop_reason,
            'iterations': actual_iters,
            'features_selected': len(S),
            'budget': target_k,
            'final_score': float(prev_score)
        }

        if verbose:
            print(f"  Selected {len(S)}/{target_k} features | stop reason: {stop_reason}")

    selected_union = sorted(set(
        f for fs in selected_per_class.values() for f in fs
    ))
    if not selected_union:
        selected_union = [int(valid_indices[0])]

    if verbose:
        print(f"\nTotal unique features selected: {len(selected_union)}")
        print("Stop summary:")
        for c_key, info in stop_info.items():
            print(f"  Class {c_key}: {info['features_selected']}/{info['budget']} "
                  f"features | {info['reason']}")

    return selected_union, selected_per_class, stop_info
