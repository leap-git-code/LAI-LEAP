#!/usr/bin/env bash
set -e

CORPUS_PATH=${1:-/path/to/wiki100w.jsonl}
INDEX_DIR=${2:-outputs/index/wiki100w_e5_ivfpq}
EMBEDDING_MODEL=${3:-/path/to/e5-base-v2}

python src/retriever/build_index.py \
  --corpus_path "$CORPUS_PATH" \
  --index_dir "$INDEX_DIR" \
  --model_path "$EMBEDDING_MODEL"
