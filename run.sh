#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$ROOT_DIR/packages/sklearn_brane"

VERSION="$(awk '/^version:/ { print $2; exit }' "$PKG_DIR/container.yml")"
IMAGE_TAR="$HOME/.local/share/brane/packages/sklearn_brane/$VERSION/image.tar"

prompt_yes_no() {
    local message="$1"
    local reply
    read -r -p "$message [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]]
}

prompt_pipeline() {
    local reply
    read -r -p "Which pipeline do you want to run? [breast/heart] " reply
    case "${reply,,}" in
        heart)
            PIPELINE_PATH="$ROOT_DIR/pipeline_heart.bs"
            RESULT_NAME="heart_results"
            RESULT_DEST="$ROOT_DIR/results/heart_disease"
            PIPELINE_LABEL="heart disease"
            ;;
        *)
            PIPELINE_PATH="$ROOT_DIR/pipeline.bs"
            RESULT_NAME="sklearn_results"
            RESULT_DEST="$ROOT_DIR/results/breast_cancer"
            PIPELINE_LABEL="breast cancer"
            ;;
    esac
}

rebuild_package() {
    echo "Building sklearn_brane version $VERSION..."
    (
        cd "$PKG_DIR"
        brane build ./container.yml --init ~/branelet
    )

    echo "Loading Docker image from $IMAGE_TAR..."
    docker load -i "$IMAGE_TAR"

    echo "Pushing sklearn_brane..."
    brane push sklearn_brane
}

run_tests() {
    echo "Running preprocess smoke test..."
    brane run "$ROOT_DIR/scripts/test_preprocess.bs" --remote

    echo "Running model smoke test..."
    brane run "$ROOT_DIR/scripts/test_model.bs" --remote

    echo "Running visualization smoke test..."
    brane run "$ROOT_DIR/scripts/test_viz.bs" --remote
}

copy_results() {
    mkdir -p "$HOME/.local/share/brane/data/$RESULT_NAME/data"
    local source_path
    source_path="$(brane data path "$RESULT_NAME")"

    rm -rf "$RESULT_DEST"
    mkdir -p "$RESULT_DEST"
    cp -R "$source_path"/. "$RESULT_DEST"/

    echo "Copied committed results to $RESULT_DEST"
}

if prompt_yes_no "Do you want to rebuild the package first?"; then
    rebuild_package
else
    echo "Skipping rebuild."
fi

prompt_pipeline
echo "Selected $PIPELINE_LABEL pipeline."

if [[ "$RESULT_NAME" == "sklearn_results" ]] && prompt_yes_no "Do you want to run the smoke tests first?"; then
    run_tests
else
    echo "Skipping smoke tests."
fi

mkdir -p "$HOME/.local/share/brane/data/$RESULT_NAME/data"

echo "Running $PIPELINE_LABEL pipeline..."
brane run "$PIPELINE_PATH" --remote

copy_results

echo "Done."
