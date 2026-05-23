"""
Shared utilities for data loading, preprocessing, and evaluation.
"""
from .data_utils import (
    load_mat_data,
    calculate_merit,
    kuncheva_index,
    mean_pairwise_overlap,
    save_result_incrementally,
    prune_rare_classes,
    detect_column_roles,
)

__all__ = [
    "load_mat_data", "calculate_merit", "kuncheva_index",
    "mean_pairwise_overlap", "save_result_incrementally",
    "prune_rare_classes", "detect_column_roles",
]
