#!/usr/bin/env python3
import json
import os
import sys

import sklearn_viz


def _env(name: str):
    return json.loads(os.environ[name])


if __name__ == "__main__":
    command = sys.argv[1]

    if command == "plot_confusion_matrix":
        sklearn_viz.plot_confusion_matrix(
            _env("PREDICTIONS"),
            _env("SPLIT_DATA"),
            _env("TARGET_COL"),
        )
    elif command == "plot_feature_importance":
        sklearn_viz.plot_feature_importance(
            _env("MODEL_DATA"),
        )
    elif command == "bundle_model_results":
        sklearn_viz.bundle_model_results(
            _env("PREDICTIONS"),
            _env("MODEL_DATA"),
            _env("CONFUSION_PLOT"),
            _env("FEATURE_PLOT"),
        )
    elif command == "bundle_core_results":
        sklearn_viz.bundle_core_results(
            _env("RF_BUNDLE"),
            _env("LR_BUNDLE"),
            _env("TARGET_COL"),
        )
    elif command == "bundle_results":
        sklearn_viz.bundle_results(
            _env("RF_BUNDLE"),
            _env("LR_BUNDLE"),
            _env("DT_BUNDLE"),
            _env("TARGET_COL"),
        )
    else:
        raise ValueError(f"Unknown command '{command}'.")
