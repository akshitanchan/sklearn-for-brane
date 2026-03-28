from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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


def _parse_columns(columns: str, available_columns: list[str]) -> list[str]:
    parsed = [column.strip() for column in columns.split(",") if column.strip()]
    if not parsed:
        raise ValueError("Expected at least one column name.")

    unknown = [column for column in parsed if column not in available_columns]
    if unknown:
        raise KeyError(
            f"Columns {unknown} are not present in available feature columns {available_columns}."
        )
    return parsed


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


def _load_model_bundle(model_data: str, pred_path: str, split_data: str):
    model_root = Path(model_data)
    pred_root = Path(pred_path)
    split_root = Path(split_data)
    model = joblib.load(_latest_matching_file(model_root, "model", ".joblib"))
    meta = json.loads(_latest_matching_file(model_root, "metadata", ".json").read_text())
    features = meta.get("feature_columns")
    y_test = pd.read_csv(_latest_matching_file(split_root, "y_test", ".csv")).iloc[:, 0]
    y_pred = pd.read_csv(_latest_matching_file(pred_root, "predictions", ".csv")).iloc[:, 0]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        importances = None
    return meta, features, y_test, y_pred, importances


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
    metadata = {
        "target_col": target_col,
        "test_size": test_size,
        "feature_columns": list(x.columns),
        "dataset_path": str(dataset_path),
    }
    _save_split_output(output_dir, stamp, x_train, x_test, y_train, y_test, metadata)
    _log(f"Saved split files to {output_dir}")
    _emit_result(output_dir)


def scale_features(data_path: str, method: str) -> None:
    _log(f"Scaling features with {method} scaler...")
    stamp = _timestamp()
    x_train, x_test, y_train, y_test = _load_split_frames(data_path)
    metadata = _latest_metadata(data_path)
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

    metadata["scaler"] = normalized

    output_dir = _ensure_result_dir("scaled_data")
    _save_split_output(
        output_dir,
        stamp,
        x_train_scaled,
        x_test_scaled,
        y_train,
        y_test,
        metadata,
    )
    joblib.dump(scaler, _timestamped_path(output_dir, "scaler", ".joblib", stamp))
    _log(f"Saved scaled data to {output_dir}")
    _emit_result(output_dir)


def impute_missing(data_path: str, columns: str, strategy: str = "most_frequent") -> None:
    _log(f"Imputing missing values with {strategy} strategy...")
    stamp = _timestamp()
    x_train, x_test, y_train, y_test = _load_split_frames(data_path)
    metadata = _latest_metadata(data_path)
    selected_columns = _parse_columns(columns, list(x_train.columns))

    imputer = SimpleImputer(strategy=strategy)
    x_train_imputed = x_train.copy()
    x_test_imputed = x_test.copy()
    x_train_imputed[selected_columns] = imputer.fit_transform(x_train[selected_columns])
    x_test_imputed[selected_columns] = imputer.transform(x_test[selected_columns])

    metadata["imputer_strategy"] = strategy
    metadata["imputed_columns"] = selected_columns

    output_dir = _ensure_result_dir("imputed_data")
    _save_split_output(
        output_dir,
        stamp,
        x_train_imputed,
        x_test_imputed,
        y_train,
        y_test,
        metadata,
    )
    _log(f"Saved imputed data to {output_dir}")
    _emit_result(output_dir)


def encode_labels(data_path: str, columns: str, method: str) -> None:
    normalized_method = method.strip().lower()
    if normalized_method != "onehot":
        raise ValueError(
            f"Unsupported encoding method '{method}'. Only 'onehot' is supported."
        )

    _log("Encoding categorical features...")
    stamp = _timestamp()
    x_train, x_test, y_train, y_test = _load_split_frames(data_path)
    metadata = _latest_metadata(data_path)
    categorical_columns = _parse_columns(columns, list(x_train.columns))

    remaining_columns = [
        column for column in x_train.columns if column not in categorical_columns
    ]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    train_encoded = encoder.fit_transform(x_train[categorical_columns].astype(str))
    test_encoded = encoder.transform(x_test[categorical_columns].astype(str))
    encoded_columns = list(encoder.get_feature_names_out(categorical_columns))

    x_train_encoded = pd.concat(
        [
            x_train[remaining_columns].reset_index(drop=True),
            pd.DataFrame(train_encoded, columns=encoded_columns),
        ],
        axis=1,
    )
    x_test_encoded = pd.concat(
        [
            x_test[remaining_columns].reset_index(drop=True),
            pd.DataFrame(test_encoded, columns=encoded_columns),
        ],
        axis=1,
    )

    metadata["encoding_method"] = normalized_method
    metadata["encoded_columns"] = categorical_columns
    metadata["feature_columns"] = list(x_train_encoded.columns)

    output_dir = _ensure_result_dir("encoded_data")
    _save_split_output(
        output_dir,
        stamp,
        x_train_encoded,
        x_test_encoded,
        y_train,
        y_test,
        metadata,
    )
    _log(f"Saved encoded data to {output_dir}")
    _emit_result(output_dir)


def fit_model(data_path: str, target_col: str, model_name: str) -> None:
    _log(f"Training {model_name} model...")
    stamp = _timestamp()
    x_train, _, y_train, _ = _load_split_frames(data_path)
    model = _build_model(model_name)
    model.fit(x_train, y_train)

    output_dir = _ensure_result_dir("model_data")
    joblib.dump(model, _timestamped_path(output_dir, "model", ".joblib", stamp))

    metadata = {
        "target_col": target_col,
        "model_name": model_name,
        "feature_columns": list(x_train.columns),
        "training_rows": len(x_train),
    }
    _save_json(_timestamped_path(output_dir, "metadata", ".json", stamp), metadata)
    _log(f"Saved model files to {output_dir}")
    _emit_result(output_dir)


def predict(model_data: str, split_data: str) -> None:
    _log("Running predictions...")
    stamp = _timestamp()
    model = joblib.load(_latest_matching_file(Path(model_data), "model", ".joblib"))
    _, x_test, _, y_test = _load_split_frames(split_data)
    predictions = pd.Series(model.predict(x_test), name="prediction")

    output_dir = _ensure_result_dir("predictions")
    predictions.to_frame().to_csv(
        _timestamped_path(output_dir, "predictions", ".csv", stamp),
        index=False,
    )
    y_test.to_frame(name=y_test.name or "target").to_csv(
        _timestamped_path(output_dir, "y_test", ".csv", stamp),
        index=False,
    )
    x_test.to_csv(_timestamped_path(output_dir, "X_test", ".csv", stamp), index=False)
    _log(f"Saved predictions to {output_dir}")
    _emit_result(output_dir)


def evaluate(pred_path: str, split_data: str, target_col: str) -> None:
    _log("Evaluating predictions...")
    stamp = _timestamp()
    y_test = pd.read_csv(_latest_matching_file(Path(split_data), "y_test", ".csv")).iloc[:, 0]
    y_pred = pd.read_csv(_latest_matching_file(Path(pred_path), "predictions", ".csv")).iloc[:, 0]
    accuracy = accuracy_score(y_test, y_pred)

    output_dir = _ensure_result_dir("evaluation")
    report_path = _timestamped_path(output_dir, "classification_report", ".csv", stamp)
    summary_path = _timestamped_path(output_dir, "results_summary", ".json", stamp)
    pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).transpose().to_csv(report_path)
    _save_json(
        summary_path,
        {
            "accuracy": round(float(accuracy), 6),
            "classification_report_csv": report_path.name,
        },
    )
    _log(f"Saved classification report to {report_path}")
    _log(f"Saved summary to {summary_path}")
    print(f'output: "Accuracy: {accuracy:.4f}"')


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

    ranking = sorted(zip(features, importances), key=lambda item: -item[1])
    output_dir = _ensure_result_dir("feature_importance")
    csv_path = _timestamped_path(output_dir, "feature_importance", ".csv", stamp)
    summary_path = _timestamped_path(output_dir, "top_features", ".json", stamp)
    pd.DataFrame(ranking, columns=["feature", "importance"]).to_csv(csv_path, index=False)
    _save_json(
        summary_path,
        [
            {"feature": name, "importance": round(float(score), 6)}
            for name, score in ranking[:5]
        ],
    )

    lines = ["Top 5 features:"]
    for name, score in ranking[:5]:
        lines.append(f"  {name}: {score:.4f}")
    _log(f"Saved full feature ranking to {csv_path}")
    _log(f"Saved top 5 feature summary to {summary_path}")
    result = "\\n".join(lines)
    print(f'output: "{result}"')


def cross_validate(split_data: str, target_col: str, model_name: str, cv: int = 5) -> None:
    _log(f"Running {cv}-fold cross-validation...")
    stamp = _timestamp()
    x_train, _, y_train, _ = _load_split_frames(split_data)
    model = _build_model(model_name)
    scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy")

    output_dir = _ensure_result_dir("cross_validation")
    pd.DataFrame(
        {"fold": list(range(1, len(scores) + 1)), "accuracy": scores}
    ).to_csv(
        _timestamped_path(output_dir, "cross_validation_scores", ".csv", stamp),
        index=False,
    )
    print(
        f'output: "Cross-validated accuracy: {scores.mean():.4f} +/- {scores.std():.4f} (n={cv})"'
    )


def plot_results(pred_path: str, model_data: str, split_data: str, target_col: str) -> None:
    _log("Generating plots...")
    stamp = _timestamp()
    _, features, y_test, y_pred, importances = _load_model_bundle(
        model_data, pred_path, split_data
    )

    output_dir = _ensure_result_dir("plots")
    confusion_path = _timestamped_path(output_dir, "confusion_matrix", ".png", stamp)
    _save_confusion_matrix_png(confusion_path, confusion_matrix(y_test, y_pred))

    manifest = {"confusion_matrix_png": confusion_path.name}
    if importances is not None and features is not None:
        feature_plot_path = _timestamped_path(
            output_dir,
            "feature_importance",
            ".png",
            stamp,
        )
        _save_feature_importance_png(feature_plot_path, features, importances)
        manifest["feature_importance_png"] = feature_plot_path.name

    _save_json(_timestamped_path(output_dir, "plot_manifest", ".json", stamp), manifest)
    _emit_result(output_dir)
