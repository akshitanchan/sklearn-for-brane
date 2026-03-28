# scikit-learn wrapper for Brane

`sklearn-for-brane` is a Brane project that wraps core scikit-learn functionality into Brane-callable actions and uses them in remote ML workflows. Brane is a workflow orchestration framework that lets packaged tasks run across distributed compute locations while keeping execution reproducible and data location-aware.

The project is split into two packages:

- `sklearn_brane`
  The main machine-learning package.
- `sklearn_viz`
  The visualization and result-bundling package.

Together, they are used in:
- a **core workflow** on the Breast Cancer dataset
- an **extended workflow** on the Heart Disease dataset

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

These cover the main classification pipeline:
- load data
- split data
- scale features
- train a model
- predict
- evaluate results
- inspect feature importance
- measure cross-validation performance

### Extended preprocessing actions

- `impute_missing`
- `encode_labels`

These are used for the Heart Disease extension, where the dataset has missing values and categorical columns.

## Package 2: `sklearn_viz`

`sklearn_viz` is the second Brane package in the project.

It is responsible for visualization and final artifact bundling.

Its actions are:

- `plot_confusion_matrix`
- `plot_feature_importance`
- `bundle_core_results`
- `bundle_results`

This package demonstrates Brane composability:
- `sklearn_brane` produces model outputs
- `sklearn_viz` consumes those outputs and creates the final result datasets

## Core vs Extended

### Core workflow

The core workflow is defined in `pipeline.bs`.

It:
- uses the `breast_cancer` dataset
- trains Random Forest and Logistic Regression
- runs the main sklearn wrapper actions
- uses `sklearn_viz.bundle_core_results`
- commits the final result as `core_results`

### Extended workflow

The extended workflow is defined in `pipeline_extended.bs`.

It:
- uses the `heart_disease` dataset
- adds `impute_missing` and `encode_labels`
- trains Random Forest, Logistic Regression, and Decision Tree
- uses `sklearn_viz.bundle_results`
- commits the final result as `extended_results`

## Build

Build and push the packages with:

```bash
bash build.sh
```

Dataset handling modes:
- default: build missing datasets, skip existing ones
- `--force-datasets`: rebuild local datasets from the manifests
- `--skip-datasets`: do not touch datasets

## Run

Run the helper script:

```bash
bash run.sh
```

Or run the workflows directly:

```bash
brane run pipeline.bs --remote
brane data path core_results

brane run pipeline_extended.bs --remote
brane data path extended_results
```

## Repo structure

- `packages/sklearn_brane`
- `packages/sklearn_viz`
- `pipeline.bs`
- `pipeline_extended.bs`
- `scripts/test_preprocess.bs`
- `scripts/test_model.bs`
- `scripts/test_viz.bs`
- `scripts/test_extended.bs`

## References

- Brane user guide: https://wiki.enablingpersonalizedinterventions.nl/user-guide/
- Brane specification: https://wiki.enablingpersonalizedinterventions.nl/specification/
- Breast Cancer Wisconsin (Diagnostic): https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Heart Disease: https://archive.ics.uci.edu/dataset/45/heart+disease
