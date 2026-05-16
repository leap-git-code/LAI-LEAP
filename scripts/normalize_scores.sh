#!/usr/bin/env bash
set -e

INPUT_DIR=${1:-outputs/scores/nq}

python src/features/normalize_candidate_scores.py \
  --input_dir "$INPUT_DIR"
