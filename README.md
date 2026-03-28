# scikit-learn wrapper for Brane

`sklearn-for-brane` is a Brane project that wraps core scikit-learn functionality into Brane-callable actions and uses them in remote ML workflows. Brane is a workflow orchestration framework that lets packaged tasks run across distributed compute locations while keeping execution reproducible and data location-aware. The repo includes a core workflow on the Breast Cancer dataset and an extended workflow on the Heart Disease dataset, with result artifacts bundled and committed back through Brane.

## What is in this repo

- `packages/sklearn_brane`
  Core ML actions: split, scale, impute, encode, train, predict, evaluate, cross-validate, and basic plotting.
- `packages/sklearn_viz`
  Visualization and bundling actions used to collect model artifacts into a final result dataset.
- `pipeline.bs`
  Core workflow on the Breast Cancer dataset. Commits `core_results`.
- `pipeline_extended.bs`
  Extended workflow on the Heart Disease dataset. Commits `extended_results`.
- `scripts/test_*.bs`
  Remote smoke tests for preprocess, model, visualization, and extension paths.

## Datasets

- `breast_cancer`
  Breast Cancer Wisconsin (Diagnostic), used by the core workflow.
- `heart_disease`
  Heart Disease dataset, used by the extended workflow.

## Workflows

### Core

The core workflow:
- uses the `breast_cancer` dataset
- trains Random Forest and Logistic Regression
- generates plot artifacts for both models
- bundles them into `core_results`

### Extended

The extended workflow:
- uses the `heart_disease` dataset
- applies missing-value imputation and one-hot encoding
- trains Random Forest, Logistic Regression, and Decision Tree
- bundles the final artifacts into `extended_results`

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

Run the core workflow:

```bash
brane run pipeline.bs --remote
brane data path core_results
```

Run the extended workflow:

```bash
brane run pipeline_extended.bs --remote
brane data path extended_results
```

Or use the helper script:

```bash
bash run.sh
```

## References

- Brane user guide: https://wiki.enablingpersonalizedinterventions.nl/user-guide/
- Brane specification: https://wiki.enablingpersonalizedinterventions.nl/specification/
- Breast Cancer Wisconsin (Diagnostic): https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Heart Disease: https://archive.ics.uci.edu/dataset/45/heart+disease
