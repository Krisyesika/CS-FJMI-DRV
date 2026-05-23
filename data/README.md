# Datasets

Place all dataset files in this directory (`data/`).  
The experiment runner reads file paths from `experiments/config.yaml`.

## Required Files

### UCI / Standard (Excel `.xlsx`)

| Dataset | File | Source |
|---|---|---|
| Arrhythmia | `Arrhythmia2.xlsx` | [UCI ML Repository](https://archive.ics.uci.edu/dataset/5) |
| Abalone | `abalone.xlsx` | [UCI ML Repository](https://archive.ics.uci.edu/dataset/1) |
| Dermatology | `dermatology.xlsx` | [UCI ML Repository](https://archive.ics.uci.edu/dataset/33) |
| Ecoli | `ecoli.xlsx` | [UCI ML Repository](https://archive.ics.uci.edu/dataset/39) |
| Hayes-Roth | `hayes.xlsx` | [UCI ML Repository](https://archive.ics.uci.edu/dataset/44) |
| Leaf | `leaf.xlsx` | [UCI ML Repository](https://archive.ics.uci.edu/dataset/288) |
| Libras Movement | `LibrasMovement.xlsx` | [UCI ML Repository](https://archive.ics.uci.edu/dataset/181) |
| Soybean (Large) | `soybeanlarge.xlsx` | [UCI ML Repository](https://archive.ics.uci.edu/dataset/90) |
| Statlog (Vehicle) | `statlog.xlsx` | [UCI ML Repository](https://archive.ics.uci.edu/dataset/149) |
| Heart Disease | `heart_disease.xlsx` | [UCI ML Repository](https://archive.ics.uci.edu/dataset/45) |
| Wine Quality (White) | `winequality_white.xlsx` | [UCI ML Repository](https://archive.ics.uci.edu/dataset/186) |
| Landsat Satellite | `landsatsatellite.xlsx` | [UCI ML Repository](https://archive.ics.uci.edu/dataset/146) |
| Dry Bean | `Dry_Bean_Dataset.xlsx` | [UCI ML Repository](https://archive.ics.uci.edu/dataset/602) |
| Optical Recognition | `optical_recognition.xlsx` | [UCI ML Repository](https://archive.ics.uci.edu/dataset/80) |
| Epileptic Seizure | `Epileptic_Seizure_Recognition.xlsx` | [UCI ML Repository](https://archive.ics.uci.edu/dataset/388) |

### Gene Expression (MATLAB `.mat`)

| Dataset | File | Source |
|---|---|---|
| CLL-SUB-111 | `CLL-SUB-111.mat` | [Arizona Feature Selection Repository](http://featureselection.asu.edu/datasets.php) |
| Lymphoma | `lymphoma.mat` | [Arizona Feature Selection Repository](http://featureselection.asu.edu/datasets.php) |

## Expected Format

**Excel files**: last column or a column named `class` is the target label.  
**MAT files**: variables named `X` (features matrix) and `Y` (labels vector).

## Column Naming Convention

All Excel datasets must have a column named `class` as the target.  
The target column name per dataset is configurable in `experiments/config.yaml`.
