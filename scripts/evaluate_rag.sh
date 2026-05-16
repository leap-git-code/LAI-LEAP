#!/usr/bin/env bash
set -e

MODEL_PATH=${1:-/path/to/llm}
DATASET_PATH=${2:-outputs/rerank/nq_leap_top5.jsonl}
SAVE_PATH=${3:-outputs/eval/nq_leap_result.json}
DATASET_TYPE=${4:-nq}
QUANTIZATION_MODE=${5:-none}

python src/evaluation/evaluate_rag.py \
  --model_path "$MODEL_PATH" \
  --dataset_path "$DATASET_PATH" \
  --save_path "$SAVE_PATH" \
  --dataset_type "$DATASET_TYPE" \
  --enable_rag \
  --top_k 5 \
  --quantization_mode "$QUANTIZATION_MODE"
