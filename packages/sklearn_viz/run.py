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
    elif command == "bundle_core_results":
        sklearn_viz.bundle_core_results(
            _env("RF_PREDICTIONS"),
            _env("RF_MODEL_DATA"),
            _env("RF_PLOT_RESULTS"),
            _env("LR_PREDICTIONS"),
            _env("LR_MODEL_DATA"),
            _env("LR_PLOT_RESULTS"),
            _env("TARGET_COL"),
        )
    elif command == "bundle_results":
        sklearn_viz.bundle_results(
            _env("RF_PREDICTIONS"),
            _env("RF_MODEL_DATA"),
            _env("RF_CONFUSION_PLOT"),
            _env("RF_FEATURE_PLOT"),
            _env("LR_PREDICTIONS"),
            _env("LR_MODEL_DATA"),
            _env("LR_CONFUSION_PLOT"),
            _env("LR_FEATURE_PLOT"),
            _env("DT_PREDICTIONS"),
            _env("DT_MODEL_DATA"),
            _env("DT_CONFUSION_PLOT"),
            _env("DT_FEATURE_PLOT"),
            _env("SPLIT_DATA"),
            _env("TARGET_COL"),
        )
    else:
        raise ValueError(f"Unknown command '{command}'.")
