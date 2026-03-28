from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np

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


def _resolve_csv_path(filepath: str) -> Path:
    path = Path(filepath)
    if path.is_file():
        return path

    candidates = [
        path / "dataset.csv",
        path / "data" / "dataset.csv",
        path / "data.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find a dataset CSV from input path '{filepath}'."
    )


def _load_split_frames(
    data_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    base = Path(data_path)
    x_train = pd.read_csv(_latest_matching_file(base, "X_train", ".csv"))
    x_test = pd.read_csv(_latest_matching_file(base, "X_test", ".csv"))
    y_train = pd.read_csv(_latest_matching_file(base, "y_train", ".csv")).iloc[:, 0]
    y_test = pd.read_csv(_latest_matching_file(base, "y_test", ".csv")).iloc[:, 0]
    return x_train, x_test, y_train, y_test


def _latest_metadata(data_path: str) -> dict:
    metadata_path = _latest_matching_file(Path(data_path), "metadata", ".json")
    return json.loads(metadata_path.read_text())


def _save_split_output(
    output_dir: Path,
    stamp: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    metadata: dict,
) -> None:
    x_train_path = _timestamped_path(output_dir, "X_train", ".csv", stamp)
    x_test_path = _timestamped_path(output_dir, "X_test", ".csv", stamp)
    y_train_path = _timestamped_path(output_dir, "y_train", ".csv", stamp)
    y_test_path = _timestamped_path(output_dir, "y_test", ".csv", stamp)
    metadata_path = _timestamped_path(output_dir, "metadata", ".json", stamp)
    x_train.to_csv(x_train_path, index=False)
    x_test.to_csv(x_test_path, index=False)
    y_train.to_frame(name=y_train.name or "target").to_csv(y_train_path, index=False)
    y_test.to_frame(name=y_test.name or "target").to_csv(y_test_path, index=False)
    _save_json(metadata_path, metadata)


def _build_model(model_name: str):
    normalized = model_name.strip().lower()
    if normalized == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    if normalized == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=42)
    if normalized == "decision_tree":
        return DecisionTreeClassifier(random_state=42)
    if normalized == "svc":
        return SVC()
    raise ValueError(f"Unsupported model_name '{model_name}'.")


def _model_dir_name(model_name: str) -> str:
    return model_name.strip().lower().replace(" ", "_")


def _load_model_bundle(model_data: str, pred_path: str, split_data: str):
    model_root = Path(model_data)
    pred_root = Path(pred_path)
    split_root = Path(split_data)
    model = joblib.load(_latest_matching_file(model_root, "model", ".joblib"))
    meta = json.loads(_latest_matching_file(model_root, "metadata", ".json").read_text())
    features = meta.get("feature_columns")
    model_name = meta.get("model_name", "model")
    y_test = pd.read_csv(_latest_matching_file(split_root, "y_test", ".csv")).iloc[:, 0]
    y_pred = pd.read_csv(_latest_matching_file(pred_root, "predictions", ".csv")).iloc[:, 0]
    x_train, _, y_train, _ = _load_split_frames(split_data)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        importances = None
    return model, meta, features, model_name, y_test, y_pred, x_train, y_train, importances


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


def _save_feature_importance_png(feature_plot_path: Path, features: list[str], importances: np.ndarray) -> None:
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


def _save_model_comparison_png(comparison_plot_path: Path, rows: list[dict]) -> None:
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    Path("/tmp/matplotlib").mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ordered_rows = sorted(rows, key=lambda row: row["accuracy"], reverse=True)
    labels = [row["model_name"].replace("_", " ").title() for row in ordered_rows]
    test_acc = np.array([row["accuracy"] for row in ordered_rows], dtype=float)
    cv_mean = np.array([row["cross_validation_mean_accuracy"] for row in ordered_rows], dtype=float)
    cv_std = np.array([row["cross_validation_std_accuracy"] for row in ordered_rows], dtype=float)

    n = len(labels)
    best_idx = int(np.argmax(test_acc))
    most_stable_idx = int(np.argmin(cv_std))

    x = np.arange(n)
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.set_facecolor("white")
    bars1 = ax.bar(
        x - width / 2,
        test_acc,
        width,
        label="Test Accuracy",
        color="#2196a6",
        edgecolor="white",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        cv_mean,
        width,
        label="CV Mean Accuracy",
        color="#90caf9",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.errorbar(
        x + width / 2,
        cv_mean,
        yerr=cv_std,
        fmt="none",
        ecolor="#555",
        elinewidth=1.2,
        capsize=4,
    )

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    y_min = max(0, min(test_acc.min(), (cv_mean - cv_std).min()) - 0.04)
    ax.set_ylim(y_min, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for tick_idx, tick in enumerate(ax.get_xticklabels()):
        if tick_idx == best_idx:
            tick.set_fontweight("bold")
            tick.set_color("#2196a6")

    ax2 = axes[1]
    ax2.set_facecolor("white")
    stability_colors = [
        "#2196a6" if i == most_stable_idx else "#90caf9"
        for i in range(n)
    ]
    bars3 = ax2.bar(x, cv_std, color=stability_colors, edgecolor="white", linewidth=0.5)

    for bar in bars3:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0005,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("Std Dev")
    ax2.set_title("CV Stability (lower = better)")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for tick_idx, tick in enumerate(ax2.get_xticklabels()):
        if tick_idx == most_stable_idx:
            tick.set_fontweight("bold")
            tick.set_color("#2196a6")

    plt.suptitle("Model Comparison", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(
        comparison_plot_path,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def _write_model_bundle(output_dir: Path, model_data: str, pred_path: str, split_data: str, cv: int = 5) -> dict:
    stamp = _timestamp()
    model, _, features, model_name, y_test, y_pred, x_train, y_train, importances = _load_model_bundle(
        model_data, pred_path, split_data
    )
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    confusion_path = _timestamped_path(output_dir, f"{model_name}_confusion_matrix", ".png", stamp)
    _save_confusion_matrix_png(confusion_path, cm)

    classification_report_path = _timestamped_path(
        output_dir, f"{model_name}_classification_report", ".csv", stamp
    )
    pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().to_csv(
        classification_report_path
    )

    cv_scores = cross_val_score(_build_model(model_name), x_train, y_train, cv=cv, scoring="accuracy")
    cv_scores_path = _timestamped_path(
        output_dir, f"{model_name}_cross_validation_scores", ".csv", stamp
    )
    pd.DataFrame({"fold": list(range(1, len(cv_scores) + 1)), "accuracy": cv_scores}).to_csv(
        cv_scores_path, index=False
    )

    summary_payload = {
        "model_name": model_name,
        "accuracy": round(float(acc), 6),
        "classification_report_csv": classification_report_path.name,
        "cross_validation_scores_csv": cv_scores_path.name,
        "cross_validation_mean_accuracy": round(float(cv_scores.mean()), 6),
        "cross_validation_std_accuracy": round(float(cv_scores.std()), 6),
    }

    if importances is not None and features is not None:
        ranking = sorted(zip(features, importances), key=lambda item: -item[1])
        feature_csv_path = _timestamped_path(output_dir, f"{model_name}_feature_importance", ".csv", stamp)
        top_features_path = _timestamped_path(output_dir, f"{model_name}_top_features", ".json", stamp)
        pd.DataFrame(ranking, columns=["feature", "importance"]).to_csv(feature_csv_path, index=False)
        _save_json(
            top_features_path,
            [{"feature": name, "importance": round(float(score), 6)} for name, score in ranking[:5]],
        )
        summary_payload["feature_importance_csv"] = feature_csv_path.name
        summary_payload["top_features_json"] = top_features_path.name
        feature_plot_path = _timestamped_path(output_dir, f"{model_name}_feature_importance", ".png", stamp)
        _save_feature_importance_png(feature_plot_path, features, importances)
        summary_payload["feature_importance_png"] = feature_plot_path.name

    summary_path = _timestamped_path(output_dir, f"{model_name}_results_summary", ".json", stamp)
    _save_json(summary_path, summary_payload)
    summary_payload["results_summary_json"] = summary_path.name
    return summary_payload


def load_and_split(filepath: str, target_col: str, test_size: float) -> None:
    _log("Loading dataset...")
    stamp = _timestamp()
    dataset_path = _resolve_csv_path(filepath)
    frame = pd.read_csv(dataset_path)

    if target_col not in frame.columns:
        raise KeyError(f"Column '{target_col}' was not found in {dataset_path}.")

    x = frame.drop(columns=[target_col])
    y = frame[target_col]

    _log("Splitting data...")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    output_dir = _ensure_result_dir("split_data")
    x_train_path = _timestamped_path(output_dir, "X_train", ".csv", stamp)
    x_test_path = _timestamped_path(output_dir, "X_test", ".csv", stamp)
    y_train_path = _timestamped_path(output_dir, "y_train", ".csv", stamp)
    y_test_path = _timestamped_path(output_dir, "y_test", ".csv", stamp)
    x_train.to_csv(x_train_path, index=False)
    x_test.to_csv(x_test_path, index=False)
    y_train.to_frame(name=target_col).to_csv(y_train_path, index=False)
    y_test.to_frame(name=target_col).to_csv(y_test_path, index=False)

    metadata = {
        "target_col": target_col,
        "test_size": test_size,
        "feature_columns": list(x.columns),
        "dataset_path": str(dataset_path),
    }
    metadata_path = _timestamped_path(output_dir, "metadata", ".json", stamp)
    _save_json(metadata_path, metadata)
    _log(f"Saved split files to {output_dir}")
    _emit_result(output_dir)


def scale_features(data_path: str, method: str) -> None:
    _log(f"Scaling features with {method} scaler...")
    stamp = _timestamp()
    x_train, x_test, y_train, y_test = _load_split_frames(data_path)
    normalized = method.strip().lower()

    if normalized == "standard":
        scaler = StandardScaler()
    elif normalized == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaling method '{method}'.")

    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train),
        columns=x_train.columns,
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        columns=x_test.columns,
    )

    output_dir = _ensure_result_dir("scaled_data")
    x_train_path = _timestamped_path(output_dir, "X_train", ".csv", stamp)
    x_test_path = _timestamped_path(output_dir, "X_test", ".csv", stamp)
    y_train_path = _timestamped_path(output_dir, "y_train", ".csv", stamp)
    y_test_path = _timestamped_path(output_dir, "y_test", ".csv", stamp)
    scaler_path = _timestamped_path(output_dir, "scaler", ".joblib", stamp)
    x_train_scaled.to_csv(x_train_path, index=False)
    x_test_scaled.to_csv(x_test_path, index=False)
    y_train.to_frame(name=y_train.name or "target").to_csv(y_train_path, index=False)
    y_test.to_frame(name=y_test.name or "target").to_csv(y_test_path, index=False)

    metadata_path = _latest_matching_file(Path(data_path), "metadata", ".json")
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    metadata["scaler"] = normalized
    metadata_path_out = _timestamped_path(output_dir, "metadata", ".json", stamp)
    _save_json(metadata_path_out, metadata)
    joblib.dump(scaler, scaler_path)
    _log(f"Saved scaled data to {output_dir}")
    _emit_result(output_dir)


def impute_missing(data_path: str, strategy: str = "most_frequent") -> None:
    _log(f"Imputing missing values with {strategy} strategy...")
    stamp = _timestamp()
    x_train, x_test, y_train, y_test = _load_split_frames(data_path)
    metadata = _latest_metadata(data_path)

    imputer = SimpleImputer(strategy=strategy)
    x_train_imputed = pd.DataFrame(
        imputer.fit_transform(x_train),
        columns=x_train.columns,
    )
    x_test_imputed = pd.DataFrame(
        imputer.transform(x_test),
        columns=x_test.columns,
    )

    metadata["imputer_strategy"] = strategy
    metadata["missing_values_imputed"] = True

    output_dir = _ensure_result_dir("imputed_data")
    _save_split_output(output_dir, stamp, x_train_imputed, x_test_imputed, y_train, y_test, metadata)
    _log(f"Saved imputed data to {output_dir}")
    _emit_result(output_dir)


def encode_labels(data_path: str, columns: str = "") -> None:
    _log("Encoding categorical features...")
    stamp = _timestamp()
    x_train, x_test, y_train, y_test = _load_split_frames(data_path)
    metadata = _latest_metadata(data_path)

    if columns.strip():
        candidate_cols = [col.strip() for col in columns.split(",") if col.strip()]
    else:
        heart_defaults = ["cp", "restecg", "thal", "slope"]
        candidate_cols = [
            col for col in x_train.columns
            if col in heart_defaults or x_train[col].dtype == object
        ]

    categorical_cols = [col for col in candidate_cols if col in x_train.columns]
    if not categorical_cols:
        metadata["encoded_columns"] = []
        output_dir = _ensure_result_dir("encoded_data")
        _save_split_output(output_dir, stamp, x_train, x_test, y_train, y_test, metadata)
        _log(f"Saved encoded data to {output_dir}")
        _emit_result(output_dir)
        return

    remaining_cols = [col for col in x_train.columns if col not in categorical_cols]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    train_encoded = encoder.fit_transform(x_train[categorical_cols].astype(str))
    test_encoded = encoder.transform(x_test[categorical_cols].astype(str))
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    x_train_encoded = pd.concat(
        [
            x_train[remaining_cols].reset_index(drop=True),
            pd.DataFrame(train_encoded, columns=encoded_cols),
        ],
        axis=1,
    )
    x_test_encoded = pd.concat(
        [
            x_test[remaining_cols].reset_index(drop=True),
            pd.DataFrame(test_encoded, columns=encoded_cols),
        ],
        axis=1,
    )

    metadata["encoded_columns"] = categorical_cols
    metadata["feature_columns"] = list(x_train_encoded.columns)

    output_dir = _ensure_result_dir("encoded_data")
    _save_split_output(output_dir, stamp, x_train_encoded, x_test_encoded, y_train, y_test, metadata)
    _log(f"Saved encoded data to {output_dir}")
    _emit_result(output_dir)


def fit_model(data_path: str, target_col: str, model_name: str) -> None:
    _log(f"Training {model_name} model...")
    stamp = _timestamp()
    x_train, _, y_train, _ = _load_split_frames(data_path)
    model = _build_model(model_name)
    model.fit(x_train, y_train)

    output_dir = _ensure_result_dir("model_data")
    model_path = _timestamped_path(output_dir, "model", ".joblib", stamp)
    metadata_path = _timestamped_path(output_dir, "metadata", ".json", stamp)
    joblib.dump(model, model_path)

    metadata = {
        "target_col": target_col,
        "model_name": model_name,
        "feature_columns": list(x_train.columns),
        "training_rows": len(x_train),
    }
    _save_json(metadata_path, metadata)
    _log(f"Saved model files to {output_dir}")
    _emit_result(output_dir)


def predict(model_data: str, split_data: str) -> None:
    _log("Running predictions...")
    stamp = _timestamp()
    model = joblib.load(_latest_matching_file(Path(model_data), "model", ".joblib"))
    _, x_test, _, y_test = _load_split_frames(split_data)
    predictions = pd.Series(model.predict(x_test), name="prediction")

    output_dir = _ensure_result_dir("predictions")
    predictions_path = _timestamped_path(output_dir, "predictions", ".csv", stamp)
    y_test_path = _timestamped_path(output_dir, "y_test", ".csv", stamp)
    x_test_path = _timestamped_path(output_dir, "X_test", ".csv", stamp)
    predictions.to_frame().to_csv(predictions_path, index=False)
    y_test.to_frame(name=y_test.name or "target").to_csv(y_test_path, index=False)
    x_test.to_csv(x_test_path, index=False)
    _log(f"Saved predictions to {predictions_path}")
    _emit_result(output_dir)

def evaluate(pred_path: str, split_data: str, target_col: str) -> None:
    _log("Evaluating predictions...")
    stamp = _timestamp()
    y_test = pd.read_csv(_latest_matching_file(Path(split_data), "y_test", ".csv")).iloc[:, 0]
    y_pred = pd.read_csv(_latest_matching_file(Path(pred_path), "predictions", ".csv")).iloc[:, 0]
    acc = accuracy_score(y_test, y_pred)
    output_dir = _ensure_result_dir("evaluation")
    report_path = _timestamped_path(output_dir, "classification_report", ".csv", stamp)
    summary_path = _timestamped_path(output_dir, "results_summary", ".json", stamp)
    pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().to_csv(report_path)
    summary = {"accuracy": round(float(acc), 6), "classification_report_csv": str(report_path)}
    _save_json(summary_path, summary)
    result = f"Accuracy: {acc:.4f}"
    _log(f"Saved classification report to {report_path}")
    _log(f"Saved summary to {summary_path}")
    print(f'output: "{result}"')

def feature_importance(model_data: str) -> None:
    _log("Calculating feature importance...")
    stamp = _timestamp()
    model_root = Path(model_data)
    model = joblib.load(_latest_matching_file(model_root, "model", ".joblib"))
    meta = json.loads(_latest_matching_file(model_root, "metadata", ".json").read_text())
    features = meta.get("feature_columns")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        print('output: "Model does not support feature importances."')
        return
    ranking = sorted(zip(features, importances), key=lambda x: -x[1])
    output_dir = _ensure_result_dir("feature_importance")
    csv_path = _timestamped_path(output_dir, "feature_importance", ".csv", stamp)
    summary_path = _timestamped_path(output_dir, "top_features", ".json", stamp)
    pd.DataFrame(ranking, columns=["feature", "importance"]).to_csv(csv_path, index=False)
    top_five = ranking[:5]
    _save_json(
        summary_path,
        [{"feature": name, "importance": round(float(score), 6)} for name, score in top_five],
    )
    lines = ["Top 5 features:"]
    for name, score in top_five:
        lines.append(f"  {name}: {score:.4f}")
    result = "\\n".join(lines)
    _log(f"Saved full feature ranking to {csv_path}")
    _log(f"Saved top 5 feature summary to {summary_path}")
    print(f'output: "{result}"')

def cross_validate(split_data: str, target_col: str, model_name: str, cv: int = 5) -> None:
    _log(f"Running {cv}-fold cross-validation...")
    stamp = _timestamp()
    x_train, _, y_train, _ = _load_split_frames(split_data)
    model = _build_model(model_name)
    scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy")
    output_dir = _ensure_result_dir("cross_validation")
    scores_path = _timestamped_path(output_dir, "cross_validation_scores", ".csv", stamp)
    pd.DataFrame(
        {"fold": list(range(1, len(scores) + 1)), "accuracy": scores}
    ).to_csv(scores_path, index=False)
    result = f"Cross-validated accuracy: {scores.mean():.4f} +/- {scores.std():.4f} (n={cv})"
    _log(f"Saved cross-validation scores to {scores_path}")
    print(f'output: "{result}"')

def plot_results(pred_path: str, model_data: str, split_data: str, target_col: str) -> None:
    _log("Generating plots...")
    stamp = _timestamp()
    _, meta, features, _, y_test, y_pred, _, _, importances = _load_model_bundle(
        model_data, pred_path, split_data
    )
    output_dir = _ensure_result_dir("plots")
    confusion_path = _timestamped_path(output_dir, "confusion_matrix", ".png", stamp)
    _save_confusion_matrix_png(confusion_path, confusion_matrix(y_test, y_pred))
    if importances is not None and features is not None:
        feature_plot_path = _timestamped_path(output_dir, "feature_importance", ".png", stamp)
        _save_feature_importance_png(feature_plot_path, features, importances)
    _emit_result(output_dir)


def bundle_results(
    rf_predictions: str,
    rf_model_data: str,
    lr_predictions: str,
    lr_model_data: str,
    dt_predictions: str,
    dt_model_data: str,
    split_data: str,
    target_col: str,
) -> None:
    _log("Bundling final results...")
    stamp = _timestamp()
    output_dir = _ensure_result_dir("sklearn_results")
    rf_output_dir = output_dir / "random_forest"
    lr_output_dir = output_dir / "logistic_regression"
    dt_output_dir = output_dir / "decision_tree"
    rf_output_dir.mkdir(parents=True, exist_ok=True)
    lr_output_dir.mkdir(parents=True, exist_ok=True)
    dt_output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _write_model_bundle(rf_output_dir, rf_model_data, rf_predictions, split_data),
        _write_model_bundle(lr_output_dir, lr_model_data, lr_predictions, split_data),
        _write_model_bundle(dt_output_dir, dt_model_data, dt_predictions, split_data),
    ]

    comparison_csv_path = _timestamped_path(output_dir, "model_comparison", ".csv", stamp)
    comparison_json_path = _timestamped_path(output_dir, "model_comparison", ".json", stamp)
    comparison_plot_path = _timestamped_path(output_dir, "model_comparison", ".png", stamp)
    pd.DataFrame(
        [
            {
                "model_name": row["model_name"],
                "accuracy": row["accuracy"],
                "cross_validation_mean_accuracy": row["cross_validation_mean_accuracy"],
                "cross_validation_std_accuracy": row["cross_validation_std_accuracy"],
            }
            for row in rows
        ]
    ).sort_values(by=["accuracy", "cross_validation_mean_accuracy"], ascending=False).to_csv(
        comparison_csv_path, index=False
    )
    best_model = max(rows, key=lambda row: (row["accuracy"], row["cross_validation_mean_accuracy"]))
    _save_json(
        comparison_json_path,
        {
            "models": rows,
            "best_model": best_model["model_name"],
            "comparison_csv": comparison_csv_path.name,
            "comparison_plot_png": comparison_plot_path.name,
        },
    )
    _save_model_comparison_png(comparison_plot_path, rows)
    _emit_result(output_dir)
