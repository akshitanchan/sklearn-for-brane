#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

prompt_yes_no() {
    local message="$1"
    local reply
    read -r -p "$message [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]]
}

prompt_mode() {
    local reply
    read -r -p "Which mode do you want to run? [core/extended] " reply
    case "${reply,,}" in
        extended)
            PIPELINE_PATH="$ROOT_DIR/pipeline_extended.bs"
            RESULT_NAME="extended_results"
            RESULT_DEST="$ROOT_DIR/results/extended"
            PIPELINE_LABEL="extended"
            ;;
        *)
            PIPELINE_PATH="$ROOT_DIR/pipeline.bs"
            RESULT_NAME="core_results"
            RESULT_DEST="$ROOT_DIR/results/core"
            PIPELINE_LABEL="core"
            ;;
    esac
}

build_package() {
    local package_name="$1"
    local package_dir="$ROOT_DIR/packages/$package_name"
    local version
    local image_tar

    version="$(awk '/^version:/ { print $2; exit }' "$package_dir/container.yml")"
    image_tar="$HOME/.local/share/brane/packages/$package_name/$version/image.tar"

    echo "Building $package_name version $version..."
    (
        cd "$package_dir"
        brane build ./container.yml --init ~/branelet
    )

    echo "Loading Docker image from $image_tar..."
    docker load -i "$image_tar"

    echo "Pushing $package_name..."
    brane push "$package_name"
}

rebuild_packages() {
    build_package "sklearn_brane"
    build_package "sklearn_viz"
}

run_core_tests() {
    echo "Running preprocess smoke test..."
    brane run "$ROOT_DIR/scripts/test_preprocess.bs" --remote

    echo "Running model smoke test..."
    brane run "$ROOT_DIR/scripts/test_model.bs" --remote

    echo "Running visualization smoke test..."
    brane run "$ROOT_DIR/scripts/test_viz.bs" --remote
}

run_extension_test() {
    echo "Running extension smoke test..."
    brane run "$ROOT_DIR/scripts/test_extended.bs" --remote
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

prompt_mode
echo "Selected $PIPELINE_LABEL mode."

if prompt_yes_no "Do you want to rebuild the required packages first?"; then
    rebuild_packages
else
    echo "Skipping rebuild."
fi

if [[ "$PIPELINE_LABEL" == "core" ]]; then
    if prompt_yes_no "Do you want to run the core smoke tests first?"; then
        run_core_tests
    else
        echo "Skipping core smoke tests."
    fi
else
    if prompt_yes_no "Do you want to run the extension smoke test first?"; then
        run_extension_test
    else
        echo "Skipping extension smoke test."
    fi
fi

mkdir -p "$HOME/.local/share/brane/data/$RESULT_NAME/data"

echo "Running $PIPELINE_LABEL pipeline..."
brane run "$PIPELINE_PATH" --remote

copy_results

echo "Done."
