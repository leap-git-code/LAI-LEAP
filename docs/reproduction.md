# Reproduction Guide

This document provides the main commands for reproducing LEAP and baseline results.

Full datasets, Wikipedia corpus files, FAISS indexes, model checkpoints, intermediate score files, and experiment outputs are not included in this repository. Please prepare them separately and pass their paths through command-line arguments.

## 1. Data and Retrieval Index

Build the retrieval index:

```bash
bash scripts/build_index.sh \
  /path/to/wiki100w.jsonl \
  outputs/index/wiki100w_e5_ivfpq \
  /path/to/e5-base-v2

Retrieve candidate contexts:

bash scripts/retrieve.sh \
  outputs/index/wiki100w_e5_ivfpq/faiss.index \
  outputs/index/wiki100w_e5_ivfpq/contexts.sqlite \
  /path/to/e5-base-v2 \
  /path/to/query.jsonl \
  outputs/retrieval/retrieved_candidates.jsonl
2. LEAP Pipeline

Compute candidate scores:

bash scripts/compute_candidate_scores.sh \
  /path/to/train_or_test.json \
  outputs/scores/nq/train \
  /path/to/llm \
  nq \
  none

For TruthfulQA, pass the auxiliary dev.jsonl file:

python src/features/score_candidate_contexts.py \
  --input_file /path/to/truthfulqa/train.json \
  --output_dir outputs/scores/truthfulqa/train \
  --model_path /path/to/llm \
  --dataset_type truthfulqa \
  --truthfulqa_dev_jsonl /path/to/truthfulqa/dev.jsonl \
  --top_k 10 \
  --batch_size 5

Normalize candidate scores:

bash scripts/normalize_scores.sh outputs/scores/nq/train
bash scripts/normalize_scores.sh outputs/scores/nq/test

Train the LEAP predictor:

bash scripts/train_leap.sh \
  /path/to/bge-large-en-v1.5 \
  outputs/scores/nq/train \
  outputs/scores/nq/test \
  outputs/leap/nq/leap_predictor.pth \
  outputs/leap/nq/plots

Rerank with LEAP:

bash scripts/rerank_with_leap.sh \
  outputs/scores/nq/test \
  outputs/leap/nq/leap_predictor.pth \
  outputs/rerank/nq_leap_top5.jsonl \
  /path/to/bge-large-en-v1.5

Evaluate LEAP:

bash scripts/evaluate_rag.sh \
  /path/to/llm \
  outputs/rerank/nq_leap_top5.jsonl \
  outputs/eval/nq_leap_result.json \
  nq \
  none
3. Offline Baselines

The following baselines are handled by scripts/run_baselines.sh:

standard_rag
bge_reranker
rankgpt
rqrag
recomp_ext
recomp_abs

Run Standard RAG:

bash scripts/run_baselines.sh \
  standard_rag \
  data/rag/nq/test.json \
  outputs/baselines/nq

Run BGE reranker:

bash scripts/run_baselines.sh \
  bge_reranker \
  data/rag/nq/test.json \
  outputs/baselines/nq \
  /path/to/bge-reranker-large

Run RankGPT:

bash scripts/run_baselines.sh \
  rankgpt \
  data/rag/nq/test.json \
  outputs/baselines/nq \
  /path/to/llm \
  bnb4

Run RQ-RAG:

BGE_MODEL_PATH=/path/to/bge-reranker-large \
bash scripts/run_baselines.sh \
  rqrag \
  data/rag/nq/test.json \
  outputs/baselines/nq \
  /path/to/llm \
  bnb4

Run ReComp-Ext:

bash scripts/run_baselines.sh \
  recomp_ext \
  data/rag/nq/test.json \
  outputs/baselines/nq \
  /path/to/recomp-ext-model

Run ReComp-Abs:

bash scripts/run_baselines.sh \
  recomp_abs \
  data/rag/nq/test.json \
  outputs/baselines/nq \
  /path/to/recomp-abs-compressor
4. Standalone Baselines

The following baselines are handled by scripts/run_standalone_baselines.sh:

oracle
selfrag
csm

Run Oracle:

bash scripts/run_standalone_baselines.sh \
  oracle \
  data/rag/nq/test.json \
  outputs/baselines/nq/test.oracle_top5.jsonl \
  /path/to/llm \
  nq

Run Self-RAG:

bash scripts/run_standalone_baselines.sh \
  selfrag \
  outputs/baselines/nq/test.standard_rag_top5.jsonl \
  outputs/eval/nq_selfrag.json \
  /path/to/llm \
  general

Run CSM:

bash scripts/run_standalone_baselines.sh \
  csm_prepare \
  data/rag/nq/train.json \
  outputs/csm/nq/train_aligned.json \
  outputs/oracle/nq_train_scores.jsonl

bash scripts/run_standalone_baselines.sh \
  csm_build_data \
  outputs/csm/nq/train_aligned.json \
  outputs/csm/nq \
  /path/to/bert-base-uncased

bash scripts/run_standalone_baselines.sh \
  csm_train \
  none \
  outputs/csm/nq \
  /path/to/bert-base-uncased

CSM_CKPT_DIR=outputs/csm/nq \
bash scripts/run_standalone_baselines.sh \
  csm_rerank \
  data/rag/nq/test.json \
  outputs/baselines/nq/test.csm_top5.jsonl \
  /path/to/bert-base-uncased
5. Evaluation

All reranked JSONL files with selected_contexts can be evaluated with src/evaluation/evaluate_rag.py.

General QA:

python src/evaluation/evaluate_rag.py \
  --model_path /path/to/llm \
  --dataset_path outputs/baselines/nq/test.standard_rag_top5.jsonl \
  --save_path outputs/eval/nq_standard_rag.json \
  --dataset_type nq \
  --enable_rag \
  --top_k 5

FEVER:

python src/evaluation/evaluate_rag.py \
  --model_path /path/to/llm \
  --dataset_path outputs/baselines/fever/test.standard_rag_top5.jsonl \
  --save_path outputs/eval/fever_standard_rag.json \
  --dataset_type fever \
  --enable_rag \
  --top_k 5

TruthfulQA:

python src/evaluation/evaluate_rag.py \
  --model_path /path/to/llm \
  --dataset_path outputs/baselines/truthfulqa/test.standard_rag_top5.jsonl \
  --truthfulqa_dev_jsonl data/rag/truthfulqa/dev.jsonl \
  --save_path outputs/eval/truthfulqa_standard_rag.json \
  --dataset_type truthfulqa \
  --enable_rag \
  --top_k 5

