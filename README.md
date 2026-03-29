# scikit-learn wrapper for Brane

`sklearn-for-brane` is a Brane project that wraps core scikit-learn functionality into Brane-callable actions and runs them across a small distributed setup. Brane is a workflow orchestration framework for packaged tasks, so this repo treats model branches as package actions that run on workers and then get merged into final result datasets.

The project is split into two packages:

- `sklearn_brane`
  The main machine-learning package.
- `sklearn_viz`
  The visualization and result-bundling package.

Together, they support:
- a **core** two-worker flow on the Breast Cancer dataset
- an **extended** two-worker flow on the Heart Disease dataset

  <img width="744" height="360" alt="brane-overview" src="https://github.com/user-attachments/assets/ce67ab17-f727-4c90-a875-3c7d16e4087e" />


## Package 1: `sklearn_brane`

`sklearn_brane` contains the main ML and preprocessing actions.

### Core actions

- `load_and_split`
- `scale_features`
- `fit_model`
- `predict`
- `evaluate`
- `feature_importance`
- `cross_validate`

These are the main scikit-learn style wrapped actions in the project. They cover the standard classification flow:
- split data
- scale features
- train models
- predict
- evaluate
- inspect feature importance
- measure cross-validation performance

### Extended preprocessing actions

- `impute_missing`
- `encode_labels`

These are used in the Heart Disease workflow, where the dataset has missing values and categorical columns.

## Package 2: `sklearn_viz`

`sklearn_viz` is the visualization and bundling package.

Its actions are:
- `plot_confusion_matrix`
- `plot_feature_importance`
- `bundle_model_results`
- `make_empty_bundle`
- `bundle_results`

In this package:
- `plot_confusion_matrix` and `plot_feature_importance` are visualization-oriented wrappers
- `bundle_model_results` packs one model branch into a single bundle
- `make_empty_bundle` creates an empty intermediate bundle placeholder for the core merge path
- `bundle_results` is the only final bundler and handles both `core` and `extended` via a workflow argument

This package is what makes the project a real two-package Brane workflow instead of just one sklearn wrapper package.

## Core vs Extended

### Core workflow

The core flow uses the `breast_cancer` dataset and is split across two workers:

- `worker1`: Random Forest branch
- `worker2`: Logistic Regression branch

The branch outputs are then merged by [pipeline.bs], which:
- loads the branch result datasets
- calls `sklearn_viz.bundle_results` with workflow `core`
- commits the final output as `core_results`

### Extended workflow

The extended flow uses the `heart_disease` dataset and is also split across two workers:

- `worker1`: Random Forest and Decision Tree branches
- `worker2`: Logistic Regression branch

The extended flow adds:
- `impute_missing`
- `encode_labels`

Then [pipeline_extended.bs]:
- loads the branch result datasets
- calls `sklearn_viz.bundle_results` with workflow `extended`
- commits the final output as `extended_results`

## Repo structure

- `packages/sklearn_brane`
- `packages/sklearn_viz`
- `pipeline.bs`
- `pipeline_extended.bs`
- `run.sh`
- `scripts/core_*_branch.bs`
- `scripts/extended_*_branch.bs`

## Build

Build and push both packages:

```bash
bash build.sh
```

Dataset handling modes:
- default: build missing datasets, skip existing ones
- `--force-datasets`: rebuild local datasets from the manifests
- `--skip-datasets`: do not touch datasets

## Run

The supported end-to-end entrypoint is:

```bash
bash run.sh core
bash run.sh extended
```

`run.sh`:
- optionally rebuilds packages
- runs the worker-specific branch workflows
- runs the merge workflow
- copies final results into `results/core` or `results/extended`

You can still run the merge workflows directly, but they only work after the branch datasets already exist:

```bash
brane run pipeline.bs --remote
brane run pipeline_extended.bs --remote
```

## References

- Brane user guide: https://wiki.enablingpersonalizedinterventions.nl/user-guide/
- Brane specification: https://wiki.enablingpersonalizedinterventions.nl/specification/
- Breast Cancer Wisconsin (Diagnostic): https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Heart Disease: https://archive.ics.uci.edu/dataset/45/heart+disease
