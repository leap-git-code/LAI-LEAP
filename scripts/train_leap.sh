#!/usr/bin/env bash
set -e

ENCODER_MODEL=${1:-/path/to/bge-large-en-v1.5}
TRAIN_DIR=${2:-outputs/scores/nq/train}
TEST_DIR=${3:-outputs/scores/nq/test}
SAVE_PATH=${4:-outputs/leap/nq/leap_predictor.pth}
PLOT_DIR=${5:-outputs/leap/nq/plots}

python src/leap/train_predictor.py \
  --encoder_model_path "$ENCODER_MODEL" \
  --train_dir "$TRAIN_DIR" \
  --test_dir "$TEST_DIR" \
  --save_path "$SAVE_PATH" \
  --plot_dir "$PLOT_DIR" \
  --batch_size 8 \
  --epochs 10
