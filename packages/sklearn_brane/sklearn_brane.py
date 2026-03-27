from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np

RESULT_ROOT = Path("/result")


def _ensure_result_dir(name: str) -> Path:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    return RESULT_ROOT


def _emit_result(path: Path) -> None:
    print(f'result: "{path}"')


def _log(message: str) -> None:
    print(message, file=sys.stderr)


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
    x_train = pd.read_csv(base / "X_train.csv")
    x_test = pd.read_csv(base / "X_test.csv")
    y_train = pd.read_csv(base / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(base / "y_test.csv").iloc[:, 0]
    return x_train, x_test, y_train, y_test


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


def load_and_split(filepath: str, target_col: str, test_size: float) -> None:
    _log("Loading dataset...")
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
    x_train.to_csv(output_dir / "X_train.csv", index=False)
    x_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_frame(name=target_col).to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_frame(name=target_col).to_csv(output_dir / "y_test.csv", index=False)

    metadata = {
        "target_col": target_col,
        "test_size": test_size,
        "feature_columns": list(x.columns),
        "dataset_path": str(dataset_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    _log(f"Saved split files to {output_dir}")
    _emit_result(output_dir)


def scale_features(data_path: str, method: str) -> None:
    _log(f"Scaling features with {method} scaler...")
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
    x_train_scaled.to_csv(output_dir / "X_train.csv", index=False)
    x_test_scaled.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_frame(name=y_train.name or "target").to_csv(
        output_dir / "y_train.csv", index=False
    )
    y_test.to_frame(name=y_test.name or "target").to_csv(
        output_dir / "y_test.csv", index=False
    )

    metadata_path = Path(data_path) / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    metadata["scaler"] = normalized
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    joblib.dump(scaler, output_dir / "scaler.joblib")
    _log(f"Saved scaled data to {output_dir}")
    _emit_result(output_dir)


def fit_model(data_path: str, target_col: str, model_name: str) -> None:
    _log(f"Training {model_name} model...")
    x_train, _, y_train, _ = _load_split_frames(data_path)
    model = _build_model(model_name)
    model.fit(x_train, y_train)

    output_dir = _ensure_result_dir("model_data")
    joblib.dump(model, output_dir / "model.joblib")

    metadata = {
        "target_col": target_col,
        "model_name": model_name,
        "feature_columns": list(x_train.columns),
        "training_rows": len(x_train),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    _log(f"Saved model files to {output_dir}")
    _emit_result(output_dir)


def predict(model_data: str, split_data: str) -> None:
    _log("Running predictions...")
    model = joblib.load(Path(model_data) / "model.joblib")
    _, x_test, _, y_test = _load_split_frames(split_data)
    predictions = pd.Series(model.predict(x_test), name="prediction")

    output_dir = _ensure_result_dir("predictions")
    predictions.to_frame().to_csv(output_dir / "predictions.csv", index=False)
    y_test.to_frame(name=y_test.name or "target").to_csv(
        output_dir / "y_test.csv", index=False
    )
    x_test.to_csv(output_dir / "X_test.csv", index=False)
    _log(f"Saved predictions to {output_dir / 'predictions.csv'}")
    _emit_result(output_dir)

def evaluate(pred_path: str, split_data: str, target_col: str) -> None:
    _log("Evaluating predictions...")
    y_test = pd.read_csv(Path(split_data) / "y_test.csv").iloc[:, 0]
    y_pred = pd.read_csv(Path(pred_path) / "predictions.csv").iloc[:, 0]
    acc = accuracy_score(y_test, y_pred)
    output_dir = _ensure_result_dir("evaluation")
    report_path = output_dir / "classification_report.csv"
    summary_path = output_dir / "results_summary.json"
    pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True)
    ).transpose().to_csv(report_path)
    summary = {"accuracy": round(float(acc), 6), "classification_report_csv": str(report_path)}
    summary_path.write_text(json.dumps(summary, indent=2))
    result = f"Accuracy: {acc:.4f}"
    _log(f"Saved classification report to {report_path}")
    _log(f"Saved summary to {summary_path}")
    print(f'output: "{result}"')

def feature_importance(model_data: str) -> None:
    _log("Calculating feature importance...")
    model = joblib.load(Path(model_data) / "model.joblib")
    meta = json.loads((Path(model_data) / "metadata.json").read_text())
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
    csv_path = output_dir / "feature_importance.csv"
    summary_path = output_dir / "top_features.json"
    pd.DataFrame(ranking, columns=["feature", "importance"]).to_csv(csv_path, index=False)
    top_five = ranking[:5]
    summary_path.write_text(
        json.dumps(
            [{"feature": name, "importance": round(float(score), 6)} for name, score in top_five],
            indent=2,
        )
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
    x_train, _, y_train, _ = _load_split_frames(split_data)
    model = _build_model(model_name)
    scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy")
    output_dir = _ensure_result_dir("cross_validation")
    scores_path = output_dir / "cross_validation_scores.csv"
    pd.DataFrame(
        {"fold": list(range(1, len(scores) + 1)), "accuracy": scores}
    ).to_csv(scores_path, index=False)
    result = f"Cross-validated accuracy: {scores.mean():.4f} +/- {scores.std():.4f} (n={cv})"
    _log(f"Saved cross-validation scores to {scores_path}")
    print(f'output: "{result}"')

def plot_results(pred_path: str, model_data: str, split_data: str, target_col: str) -> None:
    _log("Generating plots...")
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    Path("/tmp/matplotlib").mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    model = joblib.load(Path(model_data) / "model.joblib")
    meta = json.loads((Path(model_data) / "metadata.json").read_text())
    features = meta.get("feature_columns")
    y_test = pd.read_csv(Path(split_data) / "y_test.csv").iloc[:, 0]
    y_pred = pd.read_csv(Path(pred_path) / "predictions.csv").iloc[:, 0]

    cm = confusion_matrix(y_test, y_pred)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        importances = None

    output_dir = _ensure_result_dir("sklearn_report")
    confusion_path = output_dir / "confusion_matrix.png"
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(confusion_path, format="png", bbox_inches="tight")
    plt.close()
    if importances is not None and features is not None:
        feature_plot_path = output_dir / "feature_importance.png"
        plt.figure(figsize=(6, 4))
        idx = np.argsort(importances)[::-1]
        plt.bar(np.array(features)[idx], np.array(importances)[idx])
        plt.xticks(rotation=45, ha="right")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(feature_plot_path, format="png", bbox_inches="tight")
        plt.close()
        _log(f"Saved feature importance plot to {feature_plot_path}")
    _log(f"Saved confusion matrix plot to {confusion_path}")
    _emit_result(output_dir)
