
# sklearn_brane — scikit-learn wrapper for Brane

## Overview

**`sklearn_brane`** is a Brane package that wraps core scikit-learn functionality into 8 BraneScript-callable actions. It enables distributed, reproducible ML pipelines on real healthcare data. The demo pipeline trains two classifiers on the Breast Cancer dataset and saves CSV, JSON, and PNG outputs with metrics and visualizations.

## Dataset

- **Breast Cancer Wisconsin (Diagnostic)**[1]:  
  - Binary classification (malignant vs benign)
  - 569 samples, 30 numeric features
  - CSV: `data/breast_cancer/data/dataset.csv` 

## Core Functions

All 8 functions are implemented in `sklearn_brane` and exposed as Brane actions:

|   | Function            | Description / Output                                 |
|---|---------------------|------------------------------------------------------|
| 1 | `load_and_split`    | Train/test split, returns IntermediateResult         |
| 2 | `scale_features`    | Standard/MinMax scaling, returns IntermediateResult  |
| 3 | `fit_model`         | Train model (LogReg, RF, DT, SVC), returns model     |
| 4 | `predict`           | Predict on test set, returns predictions             |
| 5 | `evaluate`          | Accuracy + classification report, prints string      |
| 6 | `feature_importance`| Ranked feature importances, prints string            |
| 7 | `cross_validate`    | Cross-val accuracy, prints mean +/- std              |
| 8 | `plot_results`      | Confusion matrix + feature importance PNG outputs    |

Each function reads from an IntermediateResult path, performs one sklearn operation, saves to `/result/`, and returns a path or prints a string.

## Project Structure

```  
sklearn-brane/
├── README.md
├── build.sh
├── pipeline.bs
├── data/
│   └── breast_cancer/
│       ├── data.yml
│       └── data/
│           └── dataset.csv
├── packages/
│   └── sklearn_brane/
│       ├── container.yml
│       ├── run.py
│       ├── sklearn_brane.py
│       ├── requirements.txt
│       └── __init__.py
└── scripts/
	 ├── test_preprocess.bs
	 ├── test_model.bs
	 └── test_viz.bs
```

## How to Run

1. **Build and push the package:**
	```bash
	VERSION=$(awk '/^version:/ { print $2; exit }' packages/sklearn_brane/container.yml)
	brane build ./packages/sklearn_brane/container.yml --init ~/branelet
	docker load -i ~/.local/share/brane/packages/sklearn_brane/$VERSION/image.tar
	brane push sklearn_brane
	```

2. **Run the pipeline:**
	```bash
	brane run pipeline.bs --remote
	```

	This will:
	- Load and split the Breast Cancer data
	- Scale features
	- Train and evaluate Random Forest and Logistic Regression models
	- Print metrics and feature importances
	- Generate and commit PNG/CSV/JSON result artifacts

3. **Inspect the committed results:**
	```bash
	brane data path sklearn_report
	# Inspect the returned directory for confusion_matrix.png, feature_importance.png,
	# plot_manifest.json, and other saved outputs
	```

## Visualization

- The `plot_results` function generates `confusion_matrix.png` and `feature_importance.png`.
- The pipeline commits the output as `sklearn_report`.

## Requirements

- Python 3.9+ recommended
- scikit-learn==1.2.2
- pandas==2.0.1
- numpy==1.23.5
- joblib==1.2.0
- matplotlib==3.7.1
- seaborn==0.12.2

## References
[1] https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
