#!/bin/bash

# Ensure minimum required arguments are provided
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 --model_folders <INPUT_FOLDER> --n_to_score <N_TO_SCORE> --hnsw_path <HNSW_PATH> --sql_path <SQL_PATH>"
    exit 1
fi


# Initialize variables
MODEL_FOLDERS=""
N_TO_SCORE=""
HNSW_PATH=""
SQL_PATH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_folders)
            MODEL_FOLDERS="$2"
            shift 2
            ;;
        --n_to_score)
            N_TO_SCORE="$2"
            shift 2
            ;;
        --hnsw_path)
            HNSW_PATH="$2"
            shift 2
            ;;
        --sql_path)
            SQL_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ensure required arguments are set
if [[ -z "$MODEL_FOLDERS" || -z "$N_TO_SCORE" || -z "$HNSW_PATH" || -z "$SQL_PATH" ]]; then
    echo "Error: --model_folders, --n_to_score, --hnsw_path, and --sql_path must be specified."
    exit 1
fi

# Traverse all the subdirectories in the input to the "leaf" directories where the models are
port=6379
find "$MODEL_FOLDERS" -type d -exec test -f "{}/best_checkpoint.ckpt" \; -print | sort | while read -r dir; do
    echo "Starting traversal for:\"$dir\""
    python "$(dirname "$0")/traverse_hnsw_chemprop.py" \
    --model_folder "$dir" \
    --hnsw_path "$HNSW_PATH" \
    --sql_path "$SQL_PATH" \
    --n_to_score "$N_TO_SCORE" \
    --redis_port "$port" &> "$dir/traversal_log.log" &
    ((port++))
done

wait 