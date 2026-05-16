#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rerank candidate contexts with a trained LEAP predictor.
"""

import os
import json
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from train_predictor import (
    CandidateScoreDataset,
    LEAPPredictor,
    make_collate_fn,
)


def run_inference_and_rerank(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Infer] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_path)

    collate_fn = make_collate_fn(
        topk=args.topk,
        max_steps=args.max_steps,
        trace_feat_dim=args.trace_feat_dim,
    )

    dataset = CandidateScoreDataset(
        args.data_dir,
        tokenizer,
        max_len=args.max_len,
        max_steps=args.max_steps,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = LEAPPredictor(
        encoder_model_path=args.encoder_model_path,
        trace_feat_dim=args.trace_feat_dim,
    )

    print(f"[Infer] Loading predictor checkpoint: {args.weight_path}")
    state_dict = torch.load(args.weight_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    all_scores = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting LEAP scores"):
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            trace_feats = batch.trace_feats.to(device)
            step_mask = batch.step_mask.to(device)
            layer_mask = batch.layer_mask.to(device)

            scores = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                trace_feats=trace_feats,
                step_mask=step_mask,
                layer_mask=layer_mask,
            )

            all_scores.extend(scores.detach().cpu().numpy().tolist())

    file_paths = dataset.file_paths
    assert len(file_paths) == len(all_scores)

    grouped_candidates = {}
    query_metadata = {}

    for file_path, leap_score in zip(file_paths, all_scores):
        with open(file_path, "r", encoding="utf-8") as f:
            item = json.load(f)

        query_id = str(item.get("query_id", "unknown"))
        context_text = item.get("text", "")

        if query_id not in query_metadata:
            query_metadata[query_id] = {
                "query": item.get("question", ""),
                "ground_truth": item.get("ground_truth", []),
            }

        grouped_candidates.setdefault(query_id, []).append(
            {
                "text": context_text,
                "leap_score": float(leap_score),
                "rank_index": item.get("rank_index", None),
                "final_score": item.get("final_score", None),
                "final_score_normal": item.get("final_score_normal", None),
            }
        )

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as fout:
        for query_id, candidates in grouped_candidates.items():
            candidates.sort(key=lambda x: x["leap_score"], reverse=True)

            if args.top_k_output > 0:
                candidates = candidates[: args.top_k_output]

            selected_contexts = [c["text"] for c in candidates]
            leap_scores = [c["leap_score"] for c in candidates]

            metadata = query_metadata.get(query_id, {})

            output_item = {
                "query_id": query_id,
                "query": metadata.get("query", ""),
                "ground_truth": metadata.get("ground_truth", []),
                "selected_contexts": selected_contexts,
                "leap_scores": leap_scores,
                "reranked_candidates": candidates,
            }

            fout.write(json.dumps(output_item, ensure_ascii=False) + "\n")

    print(f"[Infer] Reranked file saved to: {args.output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Rerank candidates with a trained LEAP predictor.")

    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing candidate score JSON files.")
    parser.add_argument("--weight_path", type=str, required=True, help="Path to trained LEAP predictor checkpoint.")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL path.")

    parser.add_argument("--encoder_model_path", type=str, required=True, help="Path or HF name of the semantic encoder.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=128)
    parser.add_argument("--trace_feat_dim", type=int, default=7)

    parser.add_argument("--top_k_output", type=int, default=5, help="Number of reranked contexts to keep. Use <=0 to keep all.")

    return parser.parse_args()


def main():
    args = parse_args()
    run_inference_and_rerank(args)


if __name__ == "__main__":
    main()