#!/usr/bin/env bash
set -e

# Run standard offline reranking baselines.

METHOD=${1:-standard_rag}
INPUT_FILE=${2:-data/rag/nq/test.json}
OUTPUT_DIR=${3:-outputs/baselines/nq}
MODEL_PATH=${4:-}
QUANTIZATION_MODE=${5:-none}

TOP_K_INPUT=${TOP_K_INPUT:-10}
TOP_K_OUTPUT=${TOP_K_OUTPUT:-5}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_LENGTH=${MAX_LENGTH:-512}

case "$METHOD" in
  standard_rag)
    python src/baselines/offline_rerank.py \
      --input "$INPUT_FILE" \
      --output_dir "$OUTPUT_DIR" \
      --method standard_rag \
      --top_k_input "$TOP_K_INPUT" \
      --top_k_output "$TOP_K_OUTPUT"
    ;;

  bge_reranker)
    if [ -z "$MODEL_PATH" ]; then
      echo "Error: MODEL_PATH is required for bge_reranker."
      echo "Example: bash scripts/07_run_baselines.sh bge_reranker data/rag/nq/test.json outputs/baselines/nq /path/to/bge-reranker-large"
      exit 1
    fi

    python src/baselines/offline_rerank.py \
      --input "$INPUT_FILE" \
      --output_dir "$OUTPUT_DIR" \
      --method bge_reranker \
      --model_path "$MODEL_PATH" \
      --top_k_input "$TOP_K_INPUT" \
      --top_k_output "$TOP_K_OUTPUT" \
      --batch_size "$BATCH_SIZE" \
      --max_length "$MAX_LENGTH"
    ;;

  rankgpt)
    if [ -z "$MODEL_PATH" ]; then
      echo "Error: MODEL_PATH is required for rankgpt."
      echo "Example: bash scripts/07_run_baselines.sh rankgpt data/rag/nq/test.json outputs/baselines/nq /path/to/llm bnb4"
      exit 1
    fi

    python src/baselines/offline_rerank.py \
      --input "$INPUT_FILE" \
      --output_dir "$OUTPUT_DIR" \
      --method rankgpt \
      --model_path "$MODEL_PATH" \
      --quantization_mode "$QUANTIZATION_MODE" \
      --top_k_input "$TOP_K_INPUT" \
      --top_k_output "$TOP_K_OUTPUT"
    ;;

  rqrag)
    if [ -z "$MODEL_PATH" ]; then
      echo "Error: MODEL_PATH is required for rqrag."
      echo "Example: BGE_MODEL_PATH=/path/to/bge-reranker-large bash scripts/07_run_baselines.sh rqrag data/rag/nq/test.json outputs/baselines/nq /path/to/llm bnb4"
      exit 1
    fi

    if [ -z "$BGE_MODEL_PATH" ]; then
      echo "Error: BGE_MODEL_PATH environment variable is required for rqrag."
      echo "Example: BGE_MODEL_PATH=/path/to/bge-reranker-large bash scripts/07_run_baselines.sh rqrag data/rag/nq/test.json outputs/baselines/nq /path/to/llm bnb4"
      exit 1
    fi

    python src/baselines/offline_rerank.py \
      --input "$INPUT_FILE" \
      --output_dir "$OUTPUT_DIR" \
      --method rqrag \
      --model_path "$MODEL_PATH" \
      --bge_model_path "$BGE_MODEL_PATH" \
      --quantization_mode "$QUANTIZATION_MODE" \
      --top_k_input "$TOP_K_INPUT" \
      --top_k_output "$TOP_K_OUTPUT" \
      --batch_size "$BATCH_SIZE" \
      --max_length "$MAX_LENGTH"
    ;;

  recomp_ext)
    if [ -z "$MODEL_PATH" ]; then
      echo "Error: MODEL_PATH is required for recomp_ext."
      echo "Example: bash scripts/07_run_baselines.sh recomp_ext data/rag/nq/test.json outputs/baselines/nq /path/to/recomp-ext-model"
      exit 1
    fi

    python src/baselines/offline_rerank.py \
      --input "$INPUT_FILE" \
      --output_dir "$OUTPUT_DIR" \
      --method recomp_ext \
      --model_path "$MODEL_PATH" \
      --top_k_input "$TOP_K_INPUT" \
      --top_k_output "$TOP_K_OUTPUT" \
      --batch_size "$BATCH_SIZE" \
      --max_length "$MAX_LENGTH"
    ;;

  recomp_abs)
    if [ -z "$MODEL_PATH" ]; then
      echo "Error: MODEL_PATH is required for recomp_abs."
      echo "Example: bash scripts/07_run_baselines.sh recomp_abs data/rag/nq/test.json outputs/baselines/nq /path/to/recomp-abs-compressor"
      exit 1
    fi

    python src/baselines/offline_rerank.py \
      --input "$INPUT_FILE" \
      --output_dir "$OUTPUT_DIR" \
      --method recomp_abs \
      --model_path "$MODEL_PATH" \
      --top_k_input "$TOP_K_INPUT" \
      --top_k_output "$TOP_K_OUTPUT" \
      --max_length "$MAX_LENGTH"
    ;;

  *)
    echo "Unknown method: $METHOD"
    echo "Supported methods: standard_rag, bge_reranker, rankgpt, rqrag, recomp_ext, recomp_abs"
    exit 1
    ;;
esac
