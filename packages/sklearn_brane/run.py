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
    else:
        raise ValueError(f"Unknown command '{command}'.")
