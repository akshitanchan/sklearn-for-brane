from __future__ import annotations

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


def _ensure_result_dir(name: str) -> Path:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    return RESULT_ROOT


def _emit_result(path: Path) -> None:
    return None


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _timestamped_path(directory: Path, stem: str, suffix: str, stamp: str) -> Path:
    return directory / f"{stem}_{stamp}{suffix}"


def _latest_matching_file(directory: Path, stem: str, suffix: str) -> Path:
    matches = sorted(directory.glob(f"{stem}_*{suffix}"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find any file matching '{stem}_*{suffix}' in '{directory}'."
        )
    return matches[-1]


def _save_confusion_matrix_png(confusion_path: Path, cm: np.ndarray) -> None:
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


def _save_feature_importance_png(
    feature_plot_path: Path, features: list[str], importances: np.ndarray
) -> None:
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


def _copy_latest_file(source_dir: str, stem: str, suffix: str, destination: Path) -> str:
    source_path = _latest_matching_file(Path(source_dir), stem, suffix)
    shutil.copy2(source_path, destination)
    return destination.name


def _copy_model_bundle(
    output_dir: Path,
    model_data: str,
    pred_path: str,
    confusion_plot: str,
    feature_plot: str,
    stamp: str,
) -> dict:
    metadata_source = _latest_matching_file(Path(model_data), "metadata", ".json")
    metadata = json.loads(metadata_source.read_text())
    model_name = metadata.get("model_name", output_dir.name)

    confusion_name = _copy_latest_file(
        confusion_plot,
        "confusion_matrix",
        ".png",
        _timestamped_path(output_dir, f"{model_name}_confusion_matrix", ".png", stamp),
    )
    feature_png_name = _copy_latest_file(
        feature_plot,
        "feature_importance",
        ".png",
        _timestamped_path(output_dir, f"{model_name}_feature_importance", ".png", stamp),
    )
    feature_csv_name = _copy_latest_file(
        feature_plot,
        "feature_importance",
        ".csv",
        _timestamped_path(output_dir, f"{model_name}_feature_importance", ".csv", stamp),
    )
    top_features_name = _copy_latest_file(
        feature_plot,
        "top_features",
        ".json",
        _timestamped_path(output_dir, f"{model_name}_top_features", ".json", stamp),
    )
    predictions_name = _copy_latest_file(
        pred_path,
        "predictions",
        ".csv",
        _timestamped_path(output_dir, f"{model_name}_predictions", ".csv", stamp),
    )
    metadata_name = _copy_latest_file(
        model_data,
        "metadata",
        ".json",
        _timestamped_path(output_dir, f"{model_name}_metadata", ".json", stamp),
    )

    manifest = {
        "model_name": model_name,
        "metadata_json": metadata_name,
        "predictions_csv": predictions_name,
        "confusion_matrix_png": confusion_name,
        "feature_importance_png": feature_png_name,
        "feature_importance_csv": feature_csv_name,
        "top_features_json": top_features_name,
    }
    manifest_path = _timestamped_path(
        output_dir,
        f"{model_name}_bundle_manifest",
        ".json",
        stamp,
    )
    _save_json(manifest_path, manifest)
    manifest["bundle_manifest_json"] = manifest_path.name
    return manifest


def plot_confusion_matrix(pred_path: str, split_data: str, target_col: str) -> None:
    _log("Generating confusion matrix plot...")
    stamp = _timestamp()
    y_test = pd.read_csv(_latest_matching_file(Path(split_data), "y_test", ".csv")).iloc[:, 0]
    y_pred = pd.read_csv(_latest_matching_file(Path(pred_path), "predictions", ".csv")).iloc[:, 0]

    output_dir = _ensure_result_dir("confusion_matrix")
    confusion_path = _timestamped_path(output_dir, "confusion_matrix", ".png", stamp)
    _save_confusion_matrix_png(confusion_path, confusion_matrix(y_test, y_pred))
    _save_json(
        _timestamped_path(output_dir, "confusion_manifest", ".json", stamp),
        {"confusion_matrix_png": confusion_path.name},
    )
    _emit_result(output_dir)


def plot_feature_importance(model_data: str) -> None:
    _log("Generating feature importance plot...")
    stamp = _timestamp()
    model_root = Path(model_data)
    model = joblib.load(_latest_matching_file(model_root, "model", ".joblib"))
    metadata = json.loads(_latest_matching_file(model_root, "metadata", ".json").read_text())
    features = metadata.get("feature_columns")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        raise ValueError("Model does not support feature importance plotting.")

    ranking = sorted(zip(features, importances), key=lambda item: -item[1])
    output_dir = _ensure_result_dir("feature_importance")
    csv_path = _timestamped_path(output_dir, "feature_importance", ".csv", stamp)
    top_features_path = _timestamped_path(output_dir, "top_features", ".json", stamp)
    plot_path = _timestamped_path(output_dir, "feature_importance", ".png", stamp)

    pd.DataFrame(ranking, columns=["feature", "importance"]).to_csv(csv_path, index=False)
    _save_json(
        top_features_path,
        [
            {"feature": name, "importance": round(float(score), 6)}
            for name, score in ranking[:5]
        ],
    )
    _save_feature_importance_png(plot_path, features, importances)
    _emit_result(output_dir)


def bundle_results(
    rf_predictions: str,
    rf_model_data: str,
    rf_confusion_plot: str,
    rf_feature_plot: str,
    lr_predictions: str,
    lr_model_data: str,
    lr_confusion_plot: str,
    lr_feature_plot: str,
    dt_predictions: str,
    dt_model_data: str,
    dt_confusion_plot: str,
    dt_feature_plot: str,
    split_data: str,
    target_col: str,
) -> None:
    _log("Bundling final results...")
    stamp = _timestamp()
    output_dir = _ensure_result_dir("heart_results")

    model_specs = [
        ("random_forest", rf_model_data, rf_predictions, rf_confusion_plot, rf_feature_plot),
        ("logistic_regression", lr_model_data, lr_predictions, lr_confusion_plot, lr_feature_plot),
        ("decision_tree", dt_model_data, dt_predictions, dt_confusion_plot, dt_feature_plot),
    ]

    rows = []
    for label, model_data, predictions, confusion_plot, feature_plot in model_specs:
        model_output_dir = output_dir / label
        model_output_dir.mkdir(parents=True, exist_ok=True)
        manifest = _copy_model_bundle(
            model_output_dir,
            model_data,
            predictions,
            confusion_plot,
            feature_plot,
            stamp,
        )
        manifest["directory"] = label
        rows.append(manifest)

    comparison_csv_path = _timestamped_path(output_dir, "model_comparison", ".csv", stamp)
    comparison_json_path = _timestamped_path(output_dir, "model_comparison", ".json", stamp)
    pd.DataFrame(rows).to_csv(comparison_csv_path, index=False)
    _save_json(comparison_json_path, rows)
    _save_json(
        output_dir / "bundle_manifest.json",
        {
            "target_col": target_col,
            "comparison_csv": comparison_csv_path.name,
            "comparison_json": comparison_json_path.name,
            "models": [row["model_name"] for row in rows],
        },
    )
    _emit_result(output_dir)
