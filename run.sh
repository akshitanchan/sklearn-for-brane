#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-}"

prompt_yes_no() {
    local message="$1"
    local reply
    read -r -p "$message [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]]
}

prompt_mode() {
    local reply="${MODE,,}"

    if [[ -z "$reply" ]]; then
        read -r -p "Which mode do you want to run? [core/extended] " reply
        reply="${reply,,}"
    fi

    case "$reply" in
        extended)
            PIPELINE_PATH="$ROOT_DIR/pipeline_extended.bs"
            RESULT_NAME="extended_results"
            RESULT_DEST="$ROOT_DIR/results/extended"
            PIPELINE_LABEL="extended"
            BRANCH_SCRIPTS=(
                "$ROOT_DIR/scripts/extended_rf_branch.bs"
                "$ROOT_DIR/scripts/extended_lr_branch.bs"
                "$ROOT_DIR/scripts/extended_dt_branch.bs"
            )
            BRANCH_DATASETS=(
                "extended_rf_branch"
                "extended_lr_branch"
                "extended_dt_branch"
            )
            ;;
        core|"")
            PIPELINE_PATH="$ROOT_DIR/pipeline.bs"
            RESULT_NAME="core_results"
            RESULT_DEST="$ROOT_DIR/results/core"
            PIPELINE_LABEL="core"
            BRANCH_SCRIPTS=(
                "$ROOT_DIR/scripts/core_rf_branch.bs"
                "$ROOT_DIR/scripts/core_lr_branch.bs"
            )
            BRANCH_DATASETS=(
                "core_rf_branch"
                "core_lr_branch"
            )
            ;;
        *)
            echo "Unknown mode: $reply" >&2
            echo "Usage: bash run.sh [core|extended]" >&2
            exit 1
            ;;
    esac
}

remove_dataset_if_present() {
    local name="$1"
    brane data remove --force "$name" >/dev/null 2>&1 || true
    rm -rf "$HOME/.local/share/brane/data/$name"
}

ensure_dataset_dir() {
    local name="$1"
    mkdir -p "$HOME/.local/share/brane/data/$name/data"
}

copy_results() {
    mkdir -p "$HOME/.local/share/brane/data/$RESULT_NAME/data"
    local source_path
    source_path="$(brane data path "$RESULT_NAME")"

    mkdir -p "$ROOT_DIR/results"
    rm -rf "$RESULT_DEST"
    mkdir -p "$RESULT_DEST"
    cp -R "$source_path"/. "$RESULT_DEST"/

    echo "Copied committed results to $RESULT_DEST"
}

cleanup_run_datasets() {
    for dataset_name in "${BRANCH_DATASETS[@]}" "$RESULT_NAME"; do
        remove_dataset_if_present "$dataset_name"
    done
}

run_branch_workflows() {
    local branch_script
    local dataset_name

    for dataset_name in "${BRANCH_DATASETS[@]}"; do
        ensure_dataset_dir "$dataset_name"
    done

    for branch_script in "${BRANCH_SCRIPTS[@]}"; do
        echo "Running branch workflow $(basename "$branch_script")..."
        brane run "$branch_script" --remote
    done
}

prompt_mode
echo "Selected $PIPELINE_LABEL mode."

if prompt_yes_no "Do you want to rebuild the required packages first?"; then
    bash "$ROOT_DIR/build.sh"
else
    echo "Skipping rebuild."
fi

echo "Cleaning stale branch and result datasets..."
cleanup_run_datasets

echo "Running $PIPELINE_LABEL branch workflows..."
run_branch_workflows

ensure_dataset_dir "$RESULT_NAME"

echo "Running $PIPELINE_LABEL merge pipeline..."
brane run "$PIPELINE_PATH" --remote

copy_results

echo "Done."
