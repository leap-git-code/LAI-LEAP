<h1 align="center">Counteracting Static Embedding Collapse with Internal Latent Activation</h1>

<p align="center">
Code for LEAP, a RAG reranking framework based on internal latent activation signals.
</p>

## Overview

This repository provides the implementation for LEAP and its RAG reranking experiments. The framework supports candidate retrieval, latent-activation-based candidate scoring, LEAP predictor training, reranking, and final RAG evaluation.

The code also includes representative baseline methods, including Standard RAG, BGE-Reranker, RankGPT, RQ-RAG, ReComp-Ext, ReComp-Abs, Self-RAG, Oracle, and CSM.

## Repository Structure

```text
├── configs/
│   └── default.yaml
├── data_sample/
│   └── *_sample.jsonl
├── docs/
│   ├── data_format.md
│   ├── reproduction.md
│   └── variable_mapping.md
├── scripts/
│   ├── build_index.sh
│   ├── retrieve.sh
│   ├── compute_candidate_scores.sh
│   ├── normalize_scores.sh
│   ├── train_leap.sh
│   ├── rerank_with_leap.sh
│   ├── evaluate_rag.sh
│   ├── run_baselines.sh
│   └── run_standalone_baselines.sh
├── src/
│   ├── retriever/
│   ├── features/
│   ├── leap/
│   ├── evaluation/
│   └── baselines/
│       ├── standard_rag.py
│       ├── bge_reranker.py
│       ├── rankgpt.py
│       ├── rqrag.py
│       ├── recomp_ext.py
│       ├── recomp_abs.py
│       ├── selfrag.py
│       ├── oracle.py
│       └── csm/
├── requirements.txt
└── README.md
```

## Environment

Install the main dependencies with:

```bash
pip install -r requirements.txt
```

FAISS is not pinned in `requirements.txt`. Please install `faiss-gpu` according to your CUDA environment. For CPU-only testing, `faiss-cpu` can also be used.

GPT-OSS-20B may require a separate environment depending on the CUDA, PyTorch, and Transformers versions. Please set up the GPT-OSS-20B environment according to the target model and server CUDA configuration.

For GPT-OSS-style models, use:

```bash
--quantization_mode mxfp4
```

## Data

Full datasets, Wikipedia corpus files, retrieval indexes, model checkpoints, intermediate score files, and experiment outputs are not included in this repository.

Small examples are provided under:

```text
data_sample/
```

Please see:

```text
docs/data_format.md
```

for the expected data format.

TruthfulQA requires an auxiliary `dev.jsonl` file for multiple-choice candidate construction and gold answer mapping.

## Quick Start

The main LEAP pipeline can be reproduced with the provided scripts:

```bash
bash scripts/build_index.sh ...
bash scripts/retrieve.sh ...
bash scripts/compute_candidate_scores.sh ...
bash scripts/normalize_scores.sh ...
bash scripts/train_leap.sh ...
bash scripts/rerank_with_leap.sh ...
bash scripts/evaluate_rag.sh ...
```

Run standard offline baselines:

```bash
bash scripts/run_baselines.sh standard_rag data/rag/nq/test.json outputs/baselines/nq
bash scripts/run_baselines.sh bge_reranker data/rag/nq/test.json outputs/baselines/nq /path/to/bge-reranker-large
```

Run standalone baselines:

```bash
bash scripts/run_standalone_baselines.sh oracle data/rag/nq/test.json outputs/baselines/nq/test.oracle_top5.jsonl /path/to/llm nq
```

Detailed reproduction commands are available in:

```text
docs/reproduction.md
```

## LEAP Pipeline

The main LEAP pipeline consists of:

```text
build retrieval index
→ retrieve candidate contexts
→ compute candidate scores
→ normalize scores
→ train LEAP predictor
→ rerank with LEAP
→ evaluate RAG
```

Candidate scoring uses internal latent activation signals from the language model. The normalized candidate scores are then used to train the LEAP predictor, which reranks retrieved contexts at inference time.

## Baselines

Standard offline reranking baselines can be run with:

```bash
bash scripts/run_baselines.sh <method> <input_file> <output_dir> [model_path] [quantization_mode]
```

Supported methods:

```text
standard_rag
bge_reranker
rankgpt
rqrag
recomp_ext
recomp_abs
```

Standalone baselines can be run with:

```bash
bash scripts/run_standalone_baselines.sh <method> <input_file> <output_path> [model_path] [dataset_type]
```

Supported standalone methods:

```text
oracle
selfrag
csm_prepare
csm_build_data
csm_train
csm_rerank
```

## Evaluation

All reranked files with `selected_contexts` can be evaluated with:

```bash
python src/evaluation/evaluate_rag.py \
  --model_path /path/to/llm \
  --dataset_path /path/to/reranked.jsonl \
  --save_path /path/to/result.json \
  --dataset_type nq \
  --enable_rag \
  --top_k 5
```

For FEVER, use:

```bash
--dataset_type fever
```

For TruthfulQA, additionally pass:

```bash
--dataset_type truthfulqa \
--truthfulqa_dev_jsonl /path/to/truthfulqa/dev.jsonl
```

## Documentation

Additional documentation is provided under `docs/`:

```text
docs/data_format.md       Dataset and sample file format
docs/reproduction.md      Reproduction commands
docs/variable_mapping.md  Naming and variable mapping
```