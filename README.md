# scikit-learn wrapper for Brane

`sklearn-for-brane` is a Brane project that wraps core scikit-learn functionality into Brane-callable actions and runs them across a small distributed setup. Brane is a workflow orchestration framework for packaged tasks, so this repo treats model branches as package actions that run on workers and then get merged into final result datasets.

The project is split into two packages:

- `sklearn_brane`
  Main ML and preprocessing actions.
- `sklearn_viz`
  Visualization and result-bundling actions.

Together, they support:
- a **core** two-worker flow on the Breast Cancer dataset
- an **extended** two-worker flow on the Heart Disease dataset

## What is in this repo

- `packages/sklearn_brane`
  Split, scale, impute, encode, train, predict, evaluate, feature importance, and cross-validation.
- `packages/sklearn_viz`
  Confusion-matrix plotting, feature-importance plotting, single-model bundling, and final result aggregation.
- `scripts/core_*_branch.bs`
  Core branch workflows:
  - `worker1`: Random Forest
  - `worker2`: Logistic Regression
- `scripts/extended_*_branch.bs`
  Extended branch workflows:
  - `worker1`: Random Forest and Decision Tree
  - `worker2`: Logistic Regression
- `pipeline.bs`
  Core merge workflow. Reads `core_rf_branch` and `core_lr_branch`, then commits `core_results`.
- `pipeline_extended.bs`
  Extended merge workflow. Reads `extended_rf_branch`, `extended_lr_branch`, and `extended_dt_branch`, then commits `extended_results`.
- `run.sh`
  Official end-to-end entrypoint. Runs branch workflows first, then the merge workflow.

## Datasets

- `breast_cancer`
  Breast Cancer Wisconsin (Diagnostic), used by the core flow.
- `heart_disease`
  Heart Disease dataset, used by the extended flow.

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
- optionally runs smoke tests
- removes stale branch datasets
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
