#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-core}"

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

register_dataset() {
    local name="$1"
    local manifest="$2"
    if [[ -f "$manifest" ]]; then
        echo "Registering $name dataset..."
        brane data remove "$name" 2>/dev/null || true
        brane data build "$manifest"
    fi
}

register_dataset "breast_cancer" "$ROOT_DIR/data/breast_cancer/data.yml"
register_dataset "heart_disease" "$ROOT_DIR/data/heart_disease/data.yml"

build_package "sklearn_brane"

if [[ "$MODE" == "extended" ]]; then
    build_package "sklearn_viz"
fi

echo "Done. Run 'brane search' to verify."
