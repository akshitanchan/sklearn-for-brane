#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$ROOT_DIR/packages/sklearn_brane"
VERSION="$(awk '/^version:/ { print $2; exit }' "$PKG_DIR/container.yml")"
IMAGE_TAR="$HOME/.local/share/brane/packages/sklearn_brane/$VERSION/image.tar"

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

echo "Building sklearn_brane package version $VERSION..."
(
    cd "$PKG_DIR"
    brane build ./container.yml --init ~/branelet
)

echo "Loading image into docker from $IMAGE_TAR..."
docker load -i "$IMAGE_TAR"

echo "Pushing to registry..."
brane push sklearn_brane

echo "Done. Run 'brane search' to verify."
