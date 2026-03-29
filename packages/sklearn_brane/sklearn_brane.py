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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RESULT_ROOT = Path("/result")

# =====================
# Utility Functions
# =====================

def create_result_dir_if_not_exists() -> Path:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    return RESULT_ROOT

def log_info(message):
    print(message, file=sys.stderr)

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2))

def timestamped_path(directory, stem, suffix, stamp):
    return directory / f"{stem}_{stamp}{suffix}"

def get_latest_matching_file(directory, stem, suffix):
    matches = sorted(directory.glob(f"{stem}_*{suffix}"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find any file matching '{stem}_*{suffix}' in '{directory}'."
        )
    return matches[-1]

def resolve_csv_path(filepath):
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
        f"Could not find a dataset CSV at path '{filepath}'."
    )

def load_split_frames(data_path):
    base = Path(data_path)
    x_train = pd.read_csv(get_latest_matching_file(base, "X_train", ".csv"))
    x_test = pd.read_csv(get_latest_matching_file(base, "X_test", ".csv"))
    y_train = pd.read_csv(get_latest_matching_file(base, "y_train", ".csv")).iloc[:, 0]
    y_test = pd.read_csv(get_latest_matching_file(base, "y_test", ".csv")).iloc[:, 0]
    return x_train, x_test, y_train, y_test

def get_latest_metadata(data_path):
    metadata_path = get_latest_matching_file(Path(data_path), "metadata", ".json")
    return json.loads(metadata_path.read_text())

def save_split_output(
    output_dir,
    stamp,
    x_train,
    x_test,
    y_train,
    y_test,
    metadata,
):
    x_train_path = timestamped_path(output_dir, "X_train", ".csv", stamp)
    x_test_path = timestamped_path(output_dir, "X_test", ".csv", stamp)
    y_train_path = timestamped_path(output_dir, "y_train", ".csv", stamp)
    y_test_path = timestamped_path(output_dir, "y_test", ".csv", stamp)
    metadata_path = timestamped_path(output_dir, "metadata", ".json", stamp)
    x_train.to_csv(x_train_path, index=False)
    x_test.to_csv(x_test_path, index=False)
    y_train.to_frame(name=y_train.name or "target").to_csv(y_train_path, index=False)
    y_test.to_frame(name=y_test.name or "target").to_csv(y_test_path, index=False)
    save_json(metadata_path, metadata)

def parse_columns(columns, available_columns):
    parsed = [column.strip() for column in columns.split(",") if column.strip()]
    if not parsed:
        raise ValueError("Expected at least one column name.")

    unknown = [column for column in parsed if column not in available_columns]
    if unknown:
        raise KeyError(
            f"Columns {unknown} are not present in feature columns {available_columns}."
        )
    return parsed

def build_model(model_name):
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

# =====================
# Main Core Functions
# =====================

def load_and_split(filepath, target_col, test_size):
    log_info("Loading dataset...")
    stamp = timestamp()
    dataset_path = resolve_csv_path(filepath)
    frame = pd.read_csv(dataset_path)

    if target_col not in frame.columns:
        raise KeyError(f"Column '{target_col}' was not found in {dataset_path}.")

    x = frame.drop(columns=[target_col])
    y = frame[target_col]

    log_info("Splitting data...")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    output_dir = create_result_dir_if_not_exists()
    metadata = {
        "target_col": target_col,
        "test_size": test_size,
        "feature_columns": list(x.columns),
        "dataset_path": str(dataset_path),
    }
    save_split_output(output_dir, stamp, x_train, x_test, y_train, y_test, metadata)
    log_info(f"Saved split files to {output_dir}")
    return None

def scale_features(data_path, method):
    log_info(f"Scaling features with {method} scaler...")
    stamp = timestamp()
    x_train, x_test, y_train, y_test = load_split_frames(data_path)
    metadata = get_latest_metadata(data_path)
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

    output_dir = create_result_dir_if_not_exists()
    save_split_output(
        output_dir,
        stamp,
        x_train_scaled,
        x_test_scaled,
        y_train,
        y_test,
        metadata,
    )
    joblib.dump(scaler, timestamped_path(output_dir, "scaler", ".joblib", stamp))
    log_info(f"Saved scaled data to {output_dir}")
    return None

def fit_model(data_path, target_col, model_name):
    log_info(f"Training {model_name} model...")
    stamp = timestamp()
    x_train, _, y_train, _ = load_split_frames(data_path)
    model = build_model(model_name)
    model.fit(x_train, y_train)

    output_dir = create_result_dir_if_not_exists()
    joblib.dump(model, timestamped_path(output_dir, "model", ".joblib", stamp))

    metadata = {
        "target_col": target_col,
        "model_name": model_name,
        "feature_columns": list(x_train.columns),
        "training_rows": len(x_train),
    }
    save_json(timestamped_path(output_dir, "metadata", ".json", stamp), metadata)
    log_info(f"Saved model files to {output_dir}")
    return None

def predict(model_data, split_data):
    log_info("Running predictions...")
    stamp = timestamp()
    model = joblib.load(get_latest_matching_file(Path(model_data), "model", ".joblib"))
    _, x_test, _, y_test = load_split_frames(split_data)
    predictions = pd.Series(model.predict(x_test), name="prediction")

    output_dir = create_result_dir_if_not_exists()
    predictions.to_frame().to_csv(
        timestamped_path(output_dir, "predictions", ".csv", stamp),
        index=False,
    )
    y_test.to_frame(name=y_test.name or "target").to_csv(
        timestamped_path(output_dir, "y_test", ".csv", stamp),
        index=False,
    )
    x_test.to_csv(timestamped_path(output_dir, "X_test", ".csv", stamp), index=False)
    log_info(f"Saved predictions to {output_dir}")
    return None

def evaluate(pred_path, split_data, target_col):
    log_info("Evaluating predictions...")
    stamp = timestamp()
    y_test = pd.read_csv(get_latest_matching_file(Path(split_data), "y_test", ".csv")).iloc[:, 0]
    y_pred = pd.read_csv(get_latest_matching_file(Path(pred_path), "predictions", ".csv")).iloc[:, 0]
    accuracy = accuracy_score(y_test, y_pred)

    output_dir = create_result_dir_if_not_exists()
    report_path = timestamped_path(output_dir, "classification_report", ".csv", stamp)
    summary_path = timestamped_path(output_dir, "results_summary", ".json", stamp)
    pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).transpose().to_csv(report_path)
    save_json(
        summary_path,
        {
            "accuracy": round(float(accuracy), 6),
            "classification_report_csv": report_path.name,
        },
    )
    log_info(f"Saved classification report to {report_path}")
    log_info(f"Saved summary to {summary_path}")
    print(f'output: "Accuracy: {accuracy:.4f}"')

def feature_importance(model_data):
    log_info("Calculating feature importance...")
    stamp = timestamp()
    model_root = Path(model_data)
    model = joblib.load(get_latest_matching_file(model_root, "model", ".joblib"))
    meta = json.loads(get_latest_matching_file(model_root, "metadata", ".json").read_text())
    features = meta.get("feature_columns")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        print('output: "Model does not support feature importances."')
        return

    ranking = sorted(zip(features, importances), key=lambda item: -item[1])
    output_dir = create_result_dir_if_not_exists()
    csv_path = timestamped_path(output_dir, "feature_importance", ".csv", stamp)
    summary_path = timestamped_path(output_dir, "top_features", ".json", stamp)
    pd.DataFrame(ranking, columns=["feature", "importance"]).to_csv(csv_path, index=False)
    save_json(
        summary_path,
        [
            {"feature": name, "importance": round(float(score), 6)}
            for name, score in ranking[:5]
        ],
    )

    lines = ["Top 5 features:"]
    for name, score in ranking[:5]:
        lines.append(f"  {name}: {score:.4f}")
    log_info(f"Saved full feature ranking to {csv_path}")
    log_info(f"Saved top 5 feature summary to {summary_path}")
    result = "\\n".join(lines)
    print(f'output: "{result}"')

def cross_validate(split_data, target_col, model_name, cv = 5):
    log_info(f"Running {cv}-fold cross-validation...")
    stamp = timestamp()
    x_train, _, y_train, _ = load_split_frames(split_data)
    model = build_model(model_name)
    scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy")

    output_dir = create_result_dir_if_not_exists()
    pd.DataFrame(
        {"fold": list(range(1, len(scores) + 1)), "accuracy": scores}
    ).to_csv(
        timestamped_path(output_dir, "cross_validation_scores", ".csv", stamp),
        index=False,
    )
    print(
        f'output: "Cross-validated accuracy: {scores.mean():.4f} +/- {scores.std():.4f} (n={cv})"'
    )

# =====================
# Additional Preprocessing Functions - Extended Workflow
# =====================

def impute_missing(data_path, columns, strategy = "most_frequent"):
    log_info(f"Imputing missing values with {strategy} strategy...")
    stamp = timestamp()
    x_train, x_test, y_train, y_test = load_split_frames(data_path)
    metadata = get_latest_metadata(data_path)
    selected_columns = parse_columns(columns, list(x_train.columns))

    imputer = SimpleImputer(strategy=strategy)
    x_train_imputed = x_train.copy()
    x_test_imputed = x_test.copy()
    x_train_imputed[selected_columns] = imputer.fit_transform(x_train[selected_columns])
    x_test_imputed[selected_columns] = imputer.transform(x_test[selected_columns])

    metadata["imputer_strategy"] = strategy
    metadata["imputed_columns"] = selected_columns

    output_dir = create_result_dir_if_not_exists()
    save_split_output(
        output_dir,
        stamp,
        x_train_imputed,
        x_test_imputed,
        y_train,
        y_test,
        metadata,
    )
    log_info(f"Saved imputed data to {output_dir}")
    return None

def encode_labels(data_path, columns, method):
    normalized_method = method.strip().lower()
    if normalized_method != "onehot":
        raise ValueError(
            f"Unsupported encoding method '{method}'. Only 'onehot' is supported."
        )

    log_info("Encoding categorical features...")
    stamp = timestamp()
    x_train, x_test, y_train, y_test = load_split_frames(data_path)
    metadata = get_latest_metadata(data_path)
    categorical_columns = parse_columns(columns, list(x_train.columns))

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

    output_dir = create_result_dir_if_not_exists()
    save_split_output(
        output_dir,
        stamp,
        x_train_encoded,
        x_test_encoded,
        y_train,
        y_test,
        metadata,
    )
    log_info(f"Saved encoded data to {output_dir}")
    return None