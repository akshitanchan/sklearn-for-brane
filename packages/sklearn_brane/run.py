#!/usr/bin/env python3
import json
import os
import sys

import sklearn_brane


def _env(name: str):
    return json.loads(os.environ[name])


if __name__ == "__main__":
    command = sys.argv[1]

    if command == "load_and_split":
        sklearn_brane.load_and_split(
            _env("FILEPATH"),
            _env("TARGET_COL"),
            float(_env("TEST_SIZE")),
        )
    elif command == "scale_features":
        sklearn_brane.scale_features(
            _env("FILEPATH"),
            _env("METHOD"),
        )
    elif command == "impute_missing":
        sklearn_brane.impute_missing(
            _env("FILEPATH"),
            _env("COLUMNS"),
            _env("STRATEGY"),
        )
    elif command == "encode_labels":
        sklearn_brane.encode_labels(
            _env("FILEPATH"),
            _env("COLUMNS"),
            _env("METHOD"),
        )
    elif command == "fit_model":
        sklearn_brane.fit_model(
            _env("FILEPATH"),
            _env("TARGET_COL"),
            _env("MODEL_NAME"),
        )
    elif command == "predict":
        sklearn_brane.predict(
            _env("MODEL_DATA"),
            _env("SPLIT_DATA"),
        )
    elif command == "evaluate":
        sklearn_brane.evaluate(
            _env("PREDICTIONS"),
            _env("SPLIT_DATA"),
            _env("TARGET_COL"),
        )
    elif command == "feature_importance":
        sklearn_brane.feature_importance(
            _env("MODEL_DATA"),
        )
    elif command == "cross_validate":
        sklearn_brane.cross_validate(
            _env("SPLIT_DATA"),
            _env("TARGET_COL"),
            _env("MODEL_NAME"),
            int(_env("N_FOLDS")),
        )
    else:
        raise ValueError(f"Unknown command '{command}'.")
