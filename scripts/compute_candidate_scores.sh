#!/usr/bin/env bash
set -e

INPUT_FILE=${1:-/path/to/retrieved_or_train.json}
OUTPUT_DIR=${2:-outputs/scores/nq}
MODEL_PATH=${3:-/path/to/llm}
DATASET_TYPE=${4:-default}
QUANTIZATION_MODE=${5:-none}

python src/features/score_candidate_contexts.py \
  --input_file "$INPUT_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --model_path "$MODEL_PATH" \
  --top_k 10 \
  --batch_size 5 \
  --dataset_type "$DATASET_TYPE" \
  --quantization_mode "$QUANTIZATION_MODE"
