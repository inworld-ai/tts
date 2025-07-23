#!/bin/bash

# Define array of model paths
MODEL_PATHS=(
    "/home/feifan/tts_training/checkpoints/1b_sft_v0_i18n_clean_v2_100k"
    "/inworld/tts/checkpoints/rlhf/rlhf_2499/checkpoint-4000"
    "/inworld/tts/checkpoints/finch_production/1b/sft_v0_i18n_clean_v3"
    # Add more model paths as needed
)

# Define available CUDA devices (find which GPU is available)
CUDA_DEVICES=(0 1 2 3 4 7)

# Set base directory for output
EVAL_OUTPUT_DIR="./eval_i18n"

USE_NEW_PROMPT_PATTERN=True
BATCH_SIZE=32

# Function to run evaluation for a single model
run_evaluation() {
    local MODEL_PATH=$1
    local CUDA_DEVICE=$2

    echo "Starting evaluation for model: $MODEL_PATH on CUDA device: $CUDA_DEVICE"

    # Extract model name from path for generations directory (last directory + base name)
    PARENT_DIR=$(basename "$(dirname "$MODEL_PATH")")
    BASE_NAME=$(basename "$MODEL_PATH")
    MODEL_NAME="${PARENT_DIR}_${BASE_NAME}"

    echo "Running evaluation for $MODEL_PATH on GPU $CUDA_DEVICE"

    # Run long evaluation
    echo "Running long evaluation on GPU $CUDA_DEVICE..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE PYTHONPATH=`pwd` python ./scripts/eval/evaluate_i18n.py \
        --samples_jsonl=/inworld/tts/datasets/artificial/long_not_glued/samples.jsonl \
        --generations_dir=${EVAL_OUTPUT_DIR}/${MODEL_NAME}_long \
        --model_path=$MODEL_PATH \
        --batch_size=$BATCH_SIZE \
        --use_new_prompt_pattern=$USE_NEW_PROMPT_PATTERN

    # Run short evaluation
    echo "Running short evaluation on GPU $CUDA_DEVICE..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE PYTHONPATH=`pwd` python ./scripts/eval/evaluate_i18n.py \
        --samples_jsonl=/inworld/tts/datasets/artificial/short/samples.jsonl \
        --generations_dir=${EVAL_OUTPUT_DIR}/${MODEL_NAME}_short \
        --model_path=$MODEL_PATH \
        --batch_size=$BATCH_SIZE \
        --use_new_prompt_pattern=$USE_NEW_PROMPT_PATTERN

    # Run medium evaluation
    echo "Running medium evaluation on GPU $CUDA_DEVICE..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE PYTHONPATH=`pwd` python ./scripts/eval/evaluate_i18n.py \
        --samples_jsonl=/inworld/tts/datasets/artificial/medium/samples.jsonl \
        --generations_dir=${EVAL_OUTPUT_DIR}/${MODEL_NAME}_medium \
        --model_path=$MODEL_PATH \
        --batch_size=$BATCH_SIZE \
        --use_new_prompt_pattern=$USE_NEW_PROMPT_PATTERN
    echo "Completed evaluation for model: $MODEL_PATH on GPU $CUDA_DEVICE"
}

# Launch evaluations in parallel
pids=()
for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$i]}"
    # Cycle through available CUDA devices
    CUDA_DEVICE="${CUDA_DEVICES[$((i % ${#CUDA_DEVICES[@]}))]}"

    # Run evaluation in background
    run_evaluation "$MODEL_PATH" "$CUDA_DEVICE" &
    pids+=($!)

    echo "Launched evaluation for $MODEL_PATH on GPU $CUDA_DEVICE (PID: ${pids[-1]})"
done

# Wait for all background processes to complete
echo "Waiting for all evaluations to complete..."
for pid in "${pids[@]}"; do
    wait $pid
    echo "Process $pid completed"
done

echo "All evaluations completed"
