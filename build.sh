#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_MODE="${DATASET_MODE:-auto}"

usage() {
    cat <<'EOF'
Usage: bash build.sh [--force-datasets|--skip-datasets]

Dataset handling:
  default / auto      Build datasets that are missing locally and skip ones that already exist.
  --force-datasets    Rebuild local datasets from their manifests.
  --skip-datasets     Do not touch local datasets.

You can also set DATASET_MODE=auto|force|skip in the environment.
EOF
}

dataset_exists() {
    local name="$1"
    brane data path "$name" >/dev/null 2>&1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --force-datasets|--rebuild-datasets)
                DATASET_MODE="force"
                ;;
            --skip-datasets)
                DATASET_MODE="skip"
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown argument: $1" >&2
                usage >&2
                exit 1
                ;;
        esac
        shift
    done

    case "$DATASET_MODE" in
        auto|force|skip) ;;
        *)
            echo "Invalid DATASET_MODE: $DATASET_MODE" >&2
            usage >&2
            exit 1
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

    echo "Building $package_name package version $version..."
    (
        cd "$package_dir"
        brane build ./container.yml --init ~/branelet
    )

    echo "Loading image into docker from $image_tar..."
    docker load -i "$image_tar"

    echo "Pushing $package_name to registry..."
    brane push "$package_name"
}

ensure_result_dirs() {
    mkdir -p "$HOME/.local/share/brane/data"
    mkdir -p "$HOME/.local/share/brane/data/core_results/data"
    mkdir -p "$HOME/.local/share/brane/data/extended_results/data"
    mkdir -p "$HOME/.local/share/brane/data/core_rf_branch/data"
    mkdir -p "$HOME/.local/share/brane/data/core_lr_branch/data"
    mkdir -p "$HOME/.local/share/brane/data/extended_rf_branch/data"
    mkdir -p "$HOME/.local/share/brane/data/extended_lr_branch/data"
    mkdir -p "$HOME/.local/share/brane/data/extended_dt_branch/data"
}

register_dataset() {
    local name="$1"
    local manifest="$2"

    if [[ ! -f "$manifest" ]]; then
        return 0
    fi

    case "$DATASET_MODE" in
        skip)
            echo "Skipping dataset registration for $name."
            return 0
            ;;
        force)
            echo "Rebuilding $name dataset..."
            if dataset_exists "$name"; then
                brane data remove --force "$name"
            fi
            brane data build "$manifest"
            return 0
            ;;
    esac

    if dataset_exists "$name"; then
        echo "Dataset $name already exists locally; skipping."
    else
        echo "Registering $name dataset..."
        brane data build "$manifest"
    fi
}

parse_args "$@"

ensure_result_dirs

register_dataset "breast_cancer" "$ROOT_DIR/data/breast_cancer/data.yml"
register_dataset "heart_disease" "$ROOT_DIR/data/heart_disease/data.yml"

build_package "sklearn_brane"
build_package "sklearn_viz"

echo "Done. Run 'brane search' to verify."
