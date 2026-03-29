import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

RESULT_ROOT = Path("/result")

# =====================
# Utility Functions
# =====================

def create_result_dir_if_not_exists():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    return RESULT_ROOT

def log_info(message):
    print(message, file=sys.stderr)

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2))

def get_timestamped_path(directory, stem, suffix, stamp):
    return directory / f"{stem}_{stamp}{suffix}"

def get_latest_matching_file(directory, stem, suffix):
    matches = sorted(directory.glob(f"{stem}_*{suffix}"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find any file matching '{stem}_*{suffix}' in '{directory}'."
        )
    return matches[-1]

def save_confusion_matrix_png(confusion_path, cm):
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    Path("/tmp/matplotlib").mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(confusion_path, format="png", bbox_inches="tight")
    plt.close()

def save_feature_importance_png(
    feature_plot_path, features, importances
):
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    Path("/tmp/matplotlib").mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    idx = np.argsort(importances)[::-1]
    plt.bar(np.array(features)[idx], np.array(importances)[idx])
    plt.xticks(rotation=45, ha="right")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(feature_plot_path, format="png", bbox_inches="tight")
    plt.close()

# =====================
# Bundling and Merging Functions
# =====================

def copy_latest_file(source_dir, stem, suffix, destination):
    source_path = get_latest_matching_file(Path(source_dir), stem, suffix)
    shutil.copy2(source_path, destination)
    return destination.name

def bundle_model_artifacts(
    output_dir,
    model_data,
    pred_path,
    confusion_plot,
    feature_plot,
    stamp,
):
    metadata = json.loads(
        get_latest_matching_file(Path(model_data), "metadata", ".json").read_text()
    )
    model_name = metadata.get("model_name", output_dir.name)

    manifest = {
        "model_name": model_name,
        "metadata_json": copy_latest_file(
            model_data,
            "metadata",
            ".json",
            get_timestamped_path(output_dir, f"{model_name}_metadata", ".json", stamp),
        ),
        "predictions_csv": copy_latest_file(
            pred_path,
            "predictions",
            ".csv",
            get_timestamped_path(output_dir, f"{model_name}_predictions", ".csv", stamp),
        ),
        "confusion_matrix_png": copy_latest_file(
            confusion_plot,
            "confusion_matrix",
            ".png",
            get_timestamped_path(output_dir, f"{model_name}_confusion_matrix", ".png", stamp),
        ),
        "feature_importance_png": copy_latest_file(
            feature_plot,
            "feature_importance",
            ".png",
            get_timestamped_path(output_dir, f"{model_name}_feature_importance", ".png", stamp),
        ),
        "feature_importance_csv": copy_latest_file(
            feature_plot,
            "feature_importance",
            ".csv",
            get_timestamped_path(output_dir, f"{model_name}_feature_importance", ".csv", stamp),
        ),
        "top_features_json": copy_latest_file(
            feature_plot,
            "top_features",
            ".json",
            get_timestamped_path(output_dir, f"{model_name}_top_features", ".json", stamp),
        ),
    }

    bundle_manifest_path = get_timestamped_path(
        output_dir,
        f"{model_name}_bundle_manifest",
        ".json",
        stamp,
    )
    save_json(bundle_manifest_path, manifest)
    manifest["bundle_manifest_json"] = bundle_manifest_path.name
    return manifest

def copy_bundle_file(source_dir, source_name, destination):
    shutil.copy2(source_dir / source_name, destination)
    return destination.name

def load_bundle_manifest(bundle_dir):
    bundle_path = Path(bundle_dir)
    manifest_path = bundle_path / "bundle_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Could not find 'bundle_manifest.json' in bundled model directory '{bundle_dir}'."
        )
    return bundle_path, json.loads(manifest_path.read_text())

def merge_model_bundle(output_dir, bundle_dir, stamp):
    source_dir, manifest = load_bundle_manifest(bundle_dir)
    model_name = manifest.get("model_name", output_dir.name)

    merged = {
        "model_name": model_name,
        "metadata_json": copy_bundle_file(
            source_dir,
            manifest["metadata_json"],
            get_timestamped_path(output_dir, f"{model_name}_metadata", ".json", stamp),
        ),
        "predictions_csv": copy_bundle_file(
            source_dir,
            manifest["predictions_csv"],
            get_timestamped_path(output_dir, f"{model_name}_predictions", ".csv", stamp),
        ),
        "confusion_matrix_png": copy_bundle_file(
            source_dir,
            manifest["confusion_matrix_png"],
            get_timestamped_path(output_dir, f"{model_name}_confusion_matrix", ".png", stamp),
        ),
        "feature_importance_png": copy_bundle_file(
            source_dir,
            manifest["feature_importance_png"],
            get_timestamped_path(output_dir, f"{model_name}_feature_importance", ".png", stamp),
        ),
        "feature_importance_csv": copy_bundle_file(
            source_dir,
            manifest["feature_importance_csv"],
            get_timestamped_path(output_dir, f"{model_name}_feature_importance", ".csv", stamp),
        ),
        "top_features_json": copy_bundle_file(
            source_dir,
            manifest["top_features_json"],
            get_timestamped_path(output_dir, f"{model_name}_top_features", ".json", stamp),
        ),
    }

    bundle_manifest_path = get_timestamped_path(
        output_dir,
        f"{model_name}_bundle_manifest",
        ".json",
        stamp,
    )
    save_json(bundle_manifest_path, merged)
    merged["bundle_manifest_json"] = bundle_manifest_path.name
    return merged

def bundle_models(
    root_dir, model_specs, stamp
):
    rows = []
    for label, model_data, predictions, confusion_plot, feature_plot in model_specs:
        model_output_dir = root_dir / label
        model_output_dir.mkdir(parents=True, exist_ok=True)
        manifest = bundle_model_artifacts(
            model_output_dir,
            model_data,
            predictions,
            confusion_plot,
            feature_plot,
            stamp,
        )
        manifest["directory"] = label
        rows.append(manifest)
    return rows

def merge_model_bundles(
    root_dir, bundle_dirs, stamp
):
    rows = []
    for label, bundle_dir in bundle_dirs:
        model_output_dir = root_dir / label
        model_output_dir.mkdir(parents=True, exist_ok=True)
        manifest = merge_model_bundle(model_output_dir, bundle_dir, stamp)
        manifest["directory"] = label
        rows.append(manifest)
    return rows

def bundle_model_results(
    predictions,
    model_data,
    confusion_plot,
    feature_plot,
):
    log_info("Bundling model branch results...")
    stamp = timestamp()
    output_dir = create_result_dir_if_not_exists()
    manifest = bundle_model_artifacts(
        output_dir,
        model_data,
        predictions,
        confusion_plot,
        feature_plot,
        stamp,
    )
    save_json(output_dir / "bundle_manifest.json", manifest)
    return None

def make_empty_bundle():
    log_info("Creating empty bundle placeholder...")
    output_dir = create_result_dir_if_not_exists()
    save_json(output_dir / "empty_bundle.json", {"empty": True})
    return None

def bundle_results(
    rf_bundle,
    lr_bundle,
    dt_bundle,
    target_col,
    workflow,
):
    workflow_name = workflow.strip().lower()
    if workflow_name not in {"core", "extended"}:
        raise ValueError(f"Unknown workflow '{workflow}'. Expected 'core' or 'extended'.")

    log_info(f"Bundling {workflow_name} results...")
    stamp = timestamp()
    output_dir = create_result_dir_if_not_exists()

    bundle_dirs = [
        ("random_forest", rf_bundle),
        ("logistic_regression", lr_bundle),
    ]
    if workflow_name == "extended":
        bundle_dirs.append(("decision_tree", dt_bundle))

    rows = merge_model_bundles(output_dir, bundle_dirs, stamp)

    save_json(
        output_dir / "bundle_manifest.json",
        {
            "workflow": workflow_name,
            "target_col": target_col,
            "models": [row["model_name"] for row in rows],
            "directories": [row["directory"] for row in rows],
        },
    )
    return None

# =====================
# Core Visualization Functions
# =====================

def plot_confusion_matrix(pred_path, split_data, target_col):
    log_info("Generating confusion matrix plot...")
    stamp = timestamp()
    y_test = pd.read_csv(get_latest_matching_file(Path(split_data), "y_test", ".csv")).iloc[:, 0]
    y_pred = pd.read_csv(get_latest_matching_file(Path(pred_path), "predictions", ".csv")).iloc[:, 0]

    output_dir = create_result_dir_if_not_exists()
    confusion_path = get_timestamped_path(output_dir, "confusion_matrix", ".png", stamp)
    save_confusion_matrix_png(confusion_path, confusion_matrix(y_test, y_pred))
    save_json(
        get_timestamped_path(output_dir, "confusion_manifest", ".json", stamp),
        {"confusion_matrix_png": confusion_path.name},
    )
    return None

def plot_feature_importance(model_data):
    log_info("Generating feature importance plot...")
    stamp = timestamp()
    model_root = Path(model_data)
    model = joblib.load(get_latest_matching_file(model_root, "model", ".joblib"))
    metadata = json.loads(get_latest_matching_file(model_root, "metadata", ".json").read_text())
    features = metadata.get("feature_columns")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        raise ValueError("Model does not support feature importance plotting.")

    ranking = sorted(zip(features, importances), key=lambda item: -item[1])
    output_dir = create_result_dir_if_not_exists()
    csv_path = get_timestamped_path(output_dir, "feature_importance", ".csv", stamp)
    top_features_path = get_timestamped_path(output_dir, "top_features", ".json", stamp)
    plot_path = get_timestamped_path(output_dir, "feature_importance", ".png", stamp)

    pd.DataFrame(ranking, columns=["feature", "importance"]).to_csv(csv_path, index=False)
    save_json(
        top_features_path,
        [
            {"feature": name, "importance": round(float(score), 6)}
            for name, score in ranking[:5]
        ],
    )
    save_feature_importance_png(plot_path, features, importances)
    return None

