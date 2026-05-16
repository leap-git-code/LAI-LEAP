#!/usr/bin/env bash
set -e

DATA_DIR=${1:-outputs/scores/nq/test}
WEIGHT_PATH=${2:-outputs/leap/nq/leap_predictor.pth}
OUTPUT_PATH=${3:-outputs/rerank/nq_leap_top5.jsonl}
ENCODER_MODEL=${4:-/path/to/bge-large-en-v1.5}

python src/leap/rerank_with_predictor.py \
  --data_dir "$DATA_DIR" \
  --weight_path "$WEIGHT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --encoder_model_path "$ENCODER_MODEL" \
  --top_k_output 5 \
  --batch_size 8
