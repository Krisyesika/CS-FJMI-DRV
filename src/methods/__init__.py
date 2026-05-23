"""
Feature selection methods for CS-FJMI-DRV benchmark.

Methods:
    Baselines (classical MI):
        MIMFeatureSelector    - Mutual Information Maximization
        MIFSFeatureSelector   - Mutual Information Feature Selection (Battiti 1994)
        MRMRFeatureSelector   - Minimum Redundancy Maximum Relevance (Peng 2005)
        JMIFeatureSelector    - Joint Mutual Information (Yang 1999)
        CMIMFeatureSelector   - Conditional MI Maximization (Fleuret 2004)

    Baselines (fuzzy):
        FHFeatureSelector     - Fuzzy Entropy (Luukka 2011)
        FJMIFeatureSelector   - Fuzzy JMI (Salem 2021)
        FJMIIVFeatureSelector - Fuzzy JMI with Ideal Vectors (Salem 2022)
        FJMIURFeatureSelector - Fuzzy JMI with Uncertainty Region (Salem 2022)

    Proposed:
        cs_fjmi_drv_select    - CS-FJMI-DRV v2.1 (proposed method)
"""
from .cmim import CMIMFeatureSelector
from .fh import FHFeatureSelector
from .fjmi import FJMIFeatureSelector
from .fjmi_iv import FJMIIVFeatureSelector
from .fjmi_ur import FJMIURFeatureSelector
from .jmi import JMIFeatureSelector
from .mifs import MIFSFeatureSelector
from .mim import MIMFeatureSelector
from .mrmr import MRMRFeatureSelector
from .cs_fjmi_drv import cs_fjmi_drv_select

__all__ = [
    "CMIMFeatureSelector", "FHFeatureSelector",
    "FJMIFeatureSelector", "FJMIIVFeatureSelector", "FJMIURFeatureSelector",
    "JMIFeatureSelector", "MIFSFeatureSelector", "MIMFeatureSelector",
    "MRMRFeatureSelector", "cs_fjmi_drv_select",
]
