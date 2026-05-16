#!/usr/bin/env bash
set -e

# Run standalone baselines that are not handled by offline_rerank.py.

METHOD=${1:-oracle}
INPUT_FILE=${2:-data/rag/nq/test.json}
OUTPUT_PATH=${3:-outputs/baselines/nq/test.oracle_top5.jsonl}
MODEL_PATH=${4:-}
DATASET_TYPE=${5:-nq}

TOP_K_INPUT=${TOP_K_INPUT:-10}
TOP_K_OUTPUT=${TOP_K_OUTPUT:-5}
TOP_K=${TOP_K:-5}
BATCH_SIZE=${BATCH_SIZE:-8}
QUANTIZATION_MODE=${QUANTIZATION_MODE:-none}

case "$METHOD" in
  oracle)
    if [ -z "$MODEL_PATH" ]; then
      echo "Error: MODEL_PATH is required for oracle."
      echo "Example: bash scripts/08_run_standalone_baselines.sh oracle data/rag/nq/test.json outputs/baselines/nq/test.oracle_top5.jsonl /path/to/llm nq"
      exit 1
    fi

    CMD=(python src/baselines/oracle.py
      --input "$INPUT_FILE"
      --output "$OUTPUT_PATH"
      --model_path "$MODEL_PATH"
      --dataset_type "$DATASET_TYPE"
      --quantization_mode "$QUANTIZATION_MODE"
      --top_k_input "$TOP_K_INPUT"
      --top_k_output "$TOP_K_OUTPUT"
    )

    if [ "$DATASET_TYPE" = "truthfulqa" ]; then
      if [ -z "$TRUTHFULQA_DEV_JSONL" ]; then
        echo "Error: TRUTHFULQA_DEV_JSONL is required for oracle on TruthfulQA."
        exit 1
      fi
      CMD+=(--truthfulqa_dev_jsonl "$TRUTHFULQA_DEV_JSONL")
    fi

    "${CMD[@]}"
    ;;

  selfrag)
    if [ -z "$MODEL_PATH" ]; then
      echo "Error: MODEL_PATH is required for selfrag."
      echo "Example: bash scripts/08_run_standalone_baselines.sh selfrag outputs/baselines/nq/test.standard_rag_top5.jsonl outputs/eval/nq_selfrag.json /path/to/llm general"
      exit 1
    fi

    SELF_RAG_TYPE="$DATASET_TYPE"
    if [ "$SELF_RAG_TYPE" != "fever" ] && [ "$SELF_RAG_TYPE" != "truthfulqa" ]; then
      SELF_RAG_TYPE="general"
    fi

    CMD=(python src/baselines/selfrag.py
      --dataset_path "$INPUT_FILE"
      --save_path "$OUTPUT_PATH"
      --model_path "$MODEL_PATH"
      --dataset_type "$SELF_RAG_TYPE"
      --top_k "$TOP_K"
      --batch_size "$BATCH_SIZE"
      --quantization_mode "$QUANTIZATION_MODE"
    )

    if [ "$SELF_RAG_TYPE" = "truthfulqa" ]; then
      if [ -z "$TRUTHFULQA_DEV_JSONL" ]; then
        echo "Error: TRUTHFULQA_DEV_JSONL is required for Self-RAG on TruthfulQA."
        exit 1
      fi
      CMD+=(--truthfulqa_dev_jsonl "$TRUTHFULQA_DEV_JSONL")
    fi

    "${CMD[@]}"
    ;;

  csm_prepare)
    # INPUT_FILE: retrieval train file
    # OUTPUT_PATH: aligned train json
    # MODEL_PATH: oracle score jsonl
    if [ -z "$MODEL_PATH" ]; then
      echo "Error: oracle score JSONL path is required as the 4th argument for csm_prepare."
      echo "Example: bash scripts/08_run_standalone_baselines.sh csm_prepare data/rag/nq/train.json outputs/csm/nq/train_aligned.json outputs/oracle/nq_train_scores.jsonl"
      exit 1
    fi

    python src/baselines/csm/prepare_oracle_scores.py \
      --input_file "$INPUT_FILE" \
      --oracle_file "$MODEL_PATH" \
      --output_file "$OUTPUT_PATH" \
      --top_k "$TOP_K_INPUT" \
      --score_field score
    ;;

  csm_build_data)
    # INPUT_FILE: aligned train json
    # OUTPUT_PATH: csm save dir
    # MODEL_PATH: bert path
    if [ -z "$MODEL_PATH" ]; then
      echo "Error: BERT path is required as the 4th argument for csm_build_data."
      exit 1
    fi

    python src/baselines/csm/build_training_data.py \
      --input_json "$INPUT_FILE" \
      --save_dir "$OUTPUT_PATH" \
      --bert_path "$MODEL_PATH" \
      --top_k "$TOP_K_INPUT" \
      --max_len 512 \
      --score_field score
    ;;

  csm_train)
    # INPUT_FILE is unused in this mode.
    # OUTPUT_PATH: csm save dir
    # MODEL_PATH: bert path
    if [ -z "$MODEL_PATH" ]; then
      echo "Error: BERT path is required as the 4th argument for csm_train."
      exit 1
    fi

    python src/baselines/csm/train_csm.py \
      --bert_path "$MODEL_PATH" \
      --save_dir "$OUTPUT_PATH" \
      --top_k "$TOP_K_INPUT" \
      --num_epochs "${CSM_EPOCHS:-20}" \
      --batch_size "${CSM_BATCH_SIZE:-64}" \
      --lr "${CSM_LR:-1e-4}"
    ;;

  csm_rerank)
    # INPUT_FILE: test retrieval file
    # OUTPUT_PATH: reranked jsonl
    # MODEL_PATH: bert path
    # CSM_CKPT_DIR: checkpoint directory
    if [ -z "$MODEL_PATH" ]; then
      echo "Error: BERT path is required as the 4th argument for csm_rerank."
      exit 1
    fi
    if [ -z "$CSM_CKPT_DIR" ]; then
      echo "Error: CSM_CKPT_DIR environment variable is required for csm_rerank."
      echo "Example: CSM_CKPT_DIR=outputs/csm/nq bash scripts/08_run_standalone_baselines.sh csm_rerank data/rag/nq/test.json outputs/baselines/nq/test.csm_top5.jsonl /path/to/bert-base-uncased"
      exit 1
    fi

    python src/baselines/csm/rerank_with_csm.py \
      --input_json "$INPUT_FILE" \
      --output_jsonl "$OUTPUT_PATH" \
      --ckpt_dir "$CSM_CKPT_DIR" \
      --bert_path "$MODEL_PATH" \
      --top_k "$TOP_K_INPUT" \
      --keep "$TOP_K_OUTPUT" \
      --batch_size "${CSM_PRED_BATCH_SIZE:-64}"
    ;;

  *)
    echo "Unknown method: $METHOD"
    echo "Supported methods: selfrag, oracle, csm_prepare, csm_build_data, csm_train, csm_rerank"
    exit 1
    ;;
esac
