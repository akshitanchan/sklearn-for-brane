from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import base64
from io import BytesIO

RESULT_ROOT = Path("/result")


def _ensure_result_dir(name: str) -> Path:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    return RESULT_ROOT


def _emit_result(path: Path) -> None:
    return None


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
    dataset_path = _resolve_csv_path(filepath)
    frame = pd.read_csv(dataset_path)

    if target_col not in frame.columns:
        raise KeyError(f"Column '{target_col}' was not found in {dataset_path}.")

    x = frame.drop(columns=[target_col])
    y = frame[target_col]

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
    _emit_result(output_dir)


def scale_features(data_path: str, method: str) -> None:
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
    _emit_result(output_dir)


def fit_model(data_path: str, target_col: str, model_name: str) -> None:
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
    _emit_result(output_dir)


def predict(model_data: str, split_data: str) -> None:
    model = joblib.load(Path(model_data) / "model.joblib")
    _, x_test, _, y_test = _load_split_frames(split_data)
    predictions = pd.Series(model.predict(x_test), name="prediction")

    output_dir = _ensure_result_dir("predictions")
    predictions.to_frame().to_csv(output_dir / "predictions.csv", index=False)
    y_test.to_frame(name=y_test.name or "target").to_csv(
        output_dir / "y_test.csv", index=False
    )
    x_test.to_csv(output_dir / "X_test.csv", index=False)
    _emit_result(output_dir)

def evaluate(pred_path: str, split_data: str, target_col: str) -> None:
    y_test = pd.read_csv(Path(split_data) / "y_test.csv").iloc[:, 0]
    y_pred = pd.read_csv(Path(pred_path) / "predictions.csv").iloc[:, 0]
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    report_escaped = report.replace('"', '\\"').replace('\n', '\\n')
    result = f"Accuracy: {acc:.4f}\\n{report_escaped}"
    print(f'output: "{result}"')

def feature_importance(model_data: str) -> None:
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
    lines = ["Feature Importances:"]
    for name, score in ranking:
        lines.append(f"  {name}: {score:.4f}")
    result = "\\n".join(lines)
    print(f'output: "{result}"')

def cross_validate(split_data: str, target_col: str, model_name: str, cv: int = 5) -> None:
    x_train, _, y_train, _ = _load_split_frames(split_data)
    model = _build_model(model_name)
    scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy")
    result = f"Cross-validated accuracy: {scores.mean():.4f} +/- {scores.std():.4f} (n={cv})"
    print(f'output: "{result}"')

def plot_results(pred_path: str, model_data: str, split_data: str, target_col: str) -> None:
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

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    buf1 = BytesIO()
    plt.savefig(buf1, format="png", bbox_inches="tight")
    plt.close()
    buf1.seek(0)
    cm_b64 = base64.b64encode(buf1.read()).decode()

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        importances = None

    if importances is not None and features is not None:
        plt.figure(figsize=(6,4))
        idx = np.argsort(importances)[::-1]
        plt.bar(np.array(features)[idx], np.array(importances)[idx])
        plt.xticks(rotation=45, ha="right")
        plt.title("Feature Importance")
        plt.tight_layout()
        buf2 = BytesIO()
        plt.savefig(buf2, format="png", bbox_inches="tight")
        plt.close()
        buf2.seek(0)
        fi_b64 = base64.b64encode(buf2.read()).decode()
    else:
        fi_b64 = ""

    # Write HTML report
    output_dir = _ensure_result_dir("sklearn_report")
    html_path = output_dir / "report.html"
    template_path = Path(__file__).parent / "report_template.html"
    template = template_path.read_text()
    html = template.replace("{{CONFUSION_MATRIX}}", cm_b64).replace("{{FEATURE_IMPORTANCE}}", fi_b64)
    html_path.write_text(html)
    _emit_result(output_dir)
