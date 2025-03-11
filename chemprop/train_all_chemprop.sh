#!/bin/bash

# Ensure minimum required arguments are provided
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 --model_folders <INPUT_FOLDER> --train_fraction <TRAINING_FRACTION> [--seed <SEED>] --gpus <GPU_IDs>"
    echo "Example: $0 --model_folders /path/to/folder --train_fraction 0.833 --gpus 0 1 2 3"
    echo "         $0 --model_folders /path/to/folder --train_fraction 0.833 --gpus 0 1 2 3 --seed 123"
    exit 1
fi

# Initialize variables
FOLDER=""
TRAINING_FRACTION=""
SEED_ARG=""  # Default: empty (no seed)
GPU_IDS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_folders)
            FOLDER="$2"
            shift 2
            ;;
        --train_fraction)
            TRAINING_FRACTION="$2"
            shift 2
            ;;
        --seed)
            SEED_ARG="--seed $2"
            shift 2
            ;;
        --gpus)
            shift 1
            while [[ $# -gt 0 && "$1" != "--"* ]]; do  # Collect all GPU IDs
                GPU_IDS+=("$1")
                shift 1
            done
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ensure required arguments are set
if [[ -z "$FOLDER" || -z "$TRAINING_FRACTION" || ${#GPU_IDS[@]} -eq 0 ]]; then
    echo "Error: --model_folders, --train_fraction, and --gpus must be specified."
    exit 1
fi

GPU_COUNT=${#GPU_IDS[@]}
i=0

# Traverse all the subdirectories in the input to the "leaf" directories where the sampled data is
find "$FOLDER" -type d -links 2| sort | while read -r dir; do
    GPU_INDEX=$((i % GPU_COUNT))  # Assign GPU in round-robin fashion
    GPU=${GPU_IDS[$GPU_INDEX]}
    
    echo "Starting: CUDA_VISIBLE_DEVICES=$GPU python train_chemprop.py \"$dir\""
    CUDA_VISIBLE_DEVICES=$GPU python "$(dirname "$0")/train_chemprop.py" \
    --input_folder "$dir" \
    --train_fraction "$TRAINING_FRACTION" \
    $SEED_ARG &> "$dir/train_log.log" &

    ((i++))
done

wait 