#!/usr/bin/env bash
set -e

INDEX_PATH=${1:-outputs/index/wiki100w_e5_ivfpq/faiss.index}
SQLITE_PATH=${2:-outputs/index/wiki100w_e5_ivfpq/contexts.sqlite}
EMBEDDING_MODEL=${3:-/path/to/e5-base-v2}
QUERY_JSONL=${4:-/path/to/query.jsonl}
OUTPUT_JSONL=${5:-outputs/retrieval/retrieved_candidates.jsonl}

python src/retriever/retrieve.py \
  --index_path "$INDEX_PATH" \
  --sqlite_path "$SQLITE_PATH" \
  --model_path "$EMBEDDING_MODEL" \
  --query_jsonl "$QUERY_JSONL" \
  --output_jsonl "$OUTPUT_JSONL" \
  --top_k 10
