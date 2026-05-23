# CS-FJMI-DRV

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

**CS-FJMI-DRV**: Class-Specific Fuzzy Joint Mutual Information with Dynamic Representative Vectors.

> [Author(s)]. *"CS-FJMI-DRV: ..."*  
> DOI: `[to be added upon acceptance]`

---

## Overview

CS-FJMI-DRV is a scalable, class-specific fuzzy feature selection method that reduces computational complexity from O(N²) to O(N×K) via prototype-induced granulation (Dynamic Representative Vectors, DRV), while preserving the theoretical properties of fuzzy rough set-based mutual information.

**Key innovations:**
- **DRV**: Class-conditional MiniBatch K-Means generates K ≪ N prototypes per class, reducing pairwise relation computation from O(N²) to O(N×K).
- **Average RM accumulation**: Running-average update of the representative matrix — stable and non-monotone-aware.
- **Adaptive stopping rule** (v2.1): Relative improvement threshold `δ = 0.01` replaces the classical hard stop, accommodating the non-monotone score trajectory of average accumulation.

---

## Repository Structure

```
CS-FJMI-DRV/
├── src/
│   ├── methods/
│   │   ├── cs_fjmi_drv.py   ← Proposed method (CS-FJMI-DRV v2.1)
│   │   ├── cmim.py          ← CMIM baseline
│   │   ├── fh.py            ← FH baseline
│   │   ├── fjmi.py          ← FJMI baseline
│   │   ├── fjmi_iv.py       ← FJMI-IV baseline
│   │   ├── fjmi_ur.py       ← FJMI-UR baseline
│   │   ├── jmi.py           ← JMI baseline
│   │   ├── mifs.py          ← MIFS baseline
│   │   ├── mim.py           ← MIM baseline
│   │   └── mrmr.py          ← mRMR baseline
│   └── utils/
│       └── data_utils.py    ← Shared data loading & evaluation utilities
├── experiments/
│   ├── run_all.py           ← Main experiment runner
│   └── config.yaml          ← All hyperparameters (fixed for reproducibility)
├── data/
│   └── README.md            ← Dataset download instructions
├── results/
│   ├── tables/              ← Excel output per method
│   └── figures/             ← Generated figures
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/[your-handle]/CS-FJMI-DRV.git
cd CS-FJMI-DRV
pip install -r requirements.txt
```

**Python 3.9 or higher** is required.

---

## Reproducing Results

### Step 1 — Download datasets

Follow the instructions in [`data/README.md`](data/README.md) to download all 17 datasets and place them under `data/`.

### Step 2 — Run all experiments

```bash
python experiments/run_all.py
```

Results are saved to `results/tables/Result_<METHOD>_<TIMESTAMP>.xlsx`.

### Step 3 — Run a single method

```bash
python experiments/run_all.py --method CS-FJMI-DRV
python experiments/run_all.py --method FJMI
```

### Step 4 — Run on a single dataset

```bash
python experiments/run_all.py --method CS-FJMI-DRV --dataset Arrhythmia
```

### All CLI options

```
python experiments/run_all.py --help

  --config    Path to config YAML (default: experiments/config.yaml)
  --method    One of: MIM MIFS mRMR JMI CMIM FH FJMI FJMI-IV FJMI-UR CS-FJMI-DRV all
  --dataset   Single dataset name (from config.yaml)
  --data_dir  Path to data directory (default: data)
  --out_dir   Path to results directory (default: results/tables)
```

---

## Key Hyperparameters

All hyperparameters are defined in [`experiments/config.yaml`](experiments/config.yaml) and fixed for reproducibility.

| Parameter | Value | Description |
|---|---|---|
| `random_seed` | 42 | Global random seed |
| `n_outer_folds` | 5 | Outer cross-validation folds |
| `n_inner_folds` | 3 | Inner CV folds (hyperparameter tuning) |
| `gamma` | 1.0 | Gaussian kernel width (fixed, fair comparison) |
| `drv_ratio` | 0.05 | Prototype fraction per class (5%) |
| `drv_k_min` | 5 | Minimum prototypes per class-side |
| `drv_k_max` | 50 | Maximum prototypes per class-side |
| `n_feats_per_class` | 10 | Maximum features selected per class (T_max) |
| `rm_mode` | `"average"` | RM accumulation strategy |
| `stop_mode` | `"none"` | Stopping rule |
| `selection_ratio` | 0.5 | Feature budget for baselines (50%) |

---

## Ablation Study

To replicate the ablation study on RM mode and stopping rule (see paper Section 5.x):

```bash
# Ablation 1: RM Mode
python experiments/run_all.py --method CS-FJMI-DRV   # edit rm_mode in config.yaml

# Ablation 2: Stopping Rule
# Set stop_mode = 'none' / 'classic' / 'adaptive' in config.yaml
python experiments/run_all.py --method CS-FJMI-DRV
```

---

## Methods

| Method | Type | Reference |
|---|---|---|
| MIM | Classical MI | Lewis (1992) |
| MIFS | Classical MI | Battiti (1994) |
| mRMR | Classical MI | Peng et al. (2005) |
| JMI | Classical MI | Yang & Moody (1999) |
| CMIM | Classical MI | Fleuret (2004) |
| FH | Fuzzy Entropy | Luukka (2011) |
| FJMI | Fuzzy MI | Salem et al. (2021) |
| FJMI-IV | Fuzzy MI | Salem et al. (2022) |
| FJMI-UR | Fuzzy MI | Salem et al. (2022) |
| **CS-FJMI-DRV** | **Proposed** | **This work** |

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this code, please cite:

```bibtex
@article{cs_fjmi_drv_2025,
  author    = {[Author(s)]},
  title     = {{CS-FJMI-DRV}: Class-Specific Fuzzy Joint Mutual Information
               with Dynamic Representative Vectors},
  journal   = {Knowledge-Based Systems},
  year      = {2025},
  doi       = {[to be added]}
}
```

**Code archive (Zenodo):**

```bibtex
@software{cs_fjmi_drv_code,
  author    = {[Author(s)]},
  title     = {{CS-FJMI-DRV}: Source Code and Experiments},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {[to be added upon acceptance]}
}
```
