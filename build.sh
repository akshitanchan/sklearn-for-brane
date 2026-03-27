#!/bin/bash
set -e

echo "Registering breast_cancer dataset..."
brane data remove breast_cancer 2>/dev/null || true
brane data build ./data/breast_cancer/data.yml

echo "Building sklearn_brane package..."
cd packages/sklearn_brane
brane build ./container.yml --init ~/branelet
cd ../..

echo "Loading image into docker..."
docker load -i ~/.local/share/brane/packages/sklearn_brane/1.0.0/image.tar

echo "Pushing to registry..."
brane push sklearn_brane

echo "Done. Run 'brane list' to verify."
