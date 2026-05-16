#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train CSM reranker.

This script wraps the original CSM implementation and trains it using the
prepared parquet dataset under the specified save directory.
"""

import argparse
import os

from csm_model import CSM


def build_csm_config(args):
    return {
        "refiner_local_hidden_size": args.local_hidden_size,
        "refiner_global_hidden_size": args.global_hidden_size,
        "refiner_num_heads": args.num_heads,
        "refiner_global_layers": args.global_layers,
        "retrieval_topk": args.top_k,
        "model2path": {
            "bert": args.bert_path,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train CSM reranker.")

    parser.add_argument("--bert_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--local_hidden_size", type=int, default=768)
    parser.add_argument("--global_hidden_size", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--global_layers", type=int, default=2)

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    config = build_csm_config(args)
    model = CSM(config)

    model.fit_st(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
    )

    print(f"CSM training finished. Checkpoints saved to: {args.save_dir}")


if __name__ == "__main__":
    main()