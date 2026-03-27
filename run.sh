#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$ROOT_DIR/packages/sklearn_brane"
RESULT_NAME="sklearn_results"
RESULT_DEST="$ROOT_DIR/results/$RESULT_NAME"

VERSION="$(awk '/^version:/ { print $2; exit }' "$PKG_DIR/container.yml")"
IMAGE_TAR="$HOME/.local/share/brane/packages/sklearn_brane/$VERSION/image.tar"

prompt_yes_no() {
    local message="$1"
    local reply
    read -r -p "$message [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]]
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

if prompt_yes_no "Do you want to run the smoke tests first?"; then
    run_tests
else
    echo "Skipping smoke tests."
fi

mkdir -p "$HOME/.local/share/brane/data/$RESULT_NAME/data"

echo "Running full pipeline..."
brane run "$ROOT_DIR/pipeline.bs" --remote

copy_results

echo "Done."
