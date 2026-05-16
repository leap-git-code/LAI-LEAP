#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rerank candidate contexts with a trained CSM model.
"""

import argparse
import json
import os
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from csm_model import CSM


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]

        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]

    raise ValueError(f"Unsupported input format: {path}")


def extract_query(item: Dict[str, Any]) -> str:
    for key in ["query", "question", "input", "prompt"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_query_id(item: Dict[str, Any]) -> str:
    for key in ["query_id", "qid", "id", "question_id", "uid"]:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def extract_ground_truth(item: Dict[str, Any]) -> Any:
    for key in ["ground_truth", "groud_truth", "answers", "answer", "gold", "golden_answers"]:
        if key in item:
            return item[key]
    return []


def extract_candidates(item: Dict[str, Any]) -> List[Any]:
    for key in [
        "candidates",
        "ctxs",
        "retrieved_results",
        "retrieved_contexts",
        "contexts",
        "docs",
        "documents",
        "passages",
    ]:
        value = item.get(key)
        if isinstance(value, list):
            return value
    return []


def get_candidate_text(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate

    if not isinstance(candidate, dict):
        return str(candidate)

    for key in [
        "contents",
        "text",
        "content",
        "passage",
        "document",
        "ctx",
        "body",
        "context",
        "chunk",
        "paragraph",
        "sentence",
        "fact",
    ]:
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def build_csm_config(args):
    return {
        "retrieval_topk": args.top_k,
        "model2path": {
            "bert": args.bert_path,
        },
        "refiner_local_hidden_size": args.local_hidden_size,
        "refiner_global_hidden_size": args.global_hidden_size,
        "refiner_num_heads": args.num_heads,
        "refiner_global_layers": args.global_layers,
    }


def resolve_checkpoint_dir(ckpt_dir: str, target_file: str = "model.safetensors") -> str:
    if os.path.exists(os.path.join(ckpt_dir, target_file)):
        return ckpt_dir

    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")

    for name in os.listdir(ckpt_dir):
        subdir = os.path.join(ckpt_dir, name)
        if os.path.isdir(subdir) and os.path.exists(os.path.join(subdir, target_file)):
            print(f"Found CSM checkpoint in subdirectory: {subdir}")
            return subdir

    raise FileNotFoundError(f"Could not find {target_file} under: {ckpt_dir}")


def collect_valid_examples(data: List[Dict[str, Any]], top_k: int):
    questions = []
    context_lists = []
    valid_indices = []
    candidate_texts_by_index = {}

    skipped = 0

    for index, item in enumerate(data):
        query = extract_query(item)
        candidates = extract_candidates(item)

        if not query or len(candidates) < top_k:
            skipped += 1
            continue

        contexts = []
        valid = True

        for candidate in candidates[:top_k]:
            text = get_candidate_text(candidate)
            if not text:
                valid = False
                break
            contexts.append(text)

        if not valid:
            skipped += 1
            continue

        questions.append(query)
        context_lists.append(contexts)
        valid_indices.append(index)
        candidate_texts_by_index[index] = contexts

    print(f"Valid CSM inference examples: {len(valid_indices)}")
    print(f"Skipped examples: {skipped}")

    return questions, context_lists, valid_indices, candidate_texts_by_index


def parse_args():
    parser = argparse.ArgumentParser(description="Rerank candidates with a trained CSM model.")

    parser.add_argument("--input_json", type=str, required=True, help="Input retrieval results in JSON or JSONL format.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output reranked JSONL file.")

    parser.add_argument("--ckpt_dir", type=str, required=True, help="CSM checkpoint directory.")
    parser.add_argument("--bert_path", type=str, required=True)

    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--keep", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--local_hidden_size", type=int, default=768)
    parser.add_argument("--global_hidden_size", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--global_layers", type=int, default=2)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    data = read_json_or_jsonl(args.input_json)

    config = build_csm_config(args)
    model = CSM(config)

    checkpoint_dir = resolve_checkpoint_dir(args.ckpt_dir)
    model.load_model(checkpoint_dir)

    questions, context_lists, valid_indices, candidate_texts_by_index = collect_valid_examples(
        data=data,
        top_k=args.top_k,
    )

    scores = model.predict(
        questions=questions,
        context_lists=context_lists,
        batch_size=args.batch_size,
        device=device,
    )

    score_map = {}

    for item_scores, data_index in zip(scores, valid_indices):
        context_texts = candidate_texts_by_index[data_index]
        candidates = []

        for context_text, score in zip(context_texts, item_scores):
            candidates.append(
                {
                    "text": context_text,
                    "csm_score": float(score),
                }
            )

        candidates.sort(key=lambda x: x["csm_score"], reverse=True)
        score_map[data_index] = candidates

    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for index, item in enumerate(tqdm(data, desc="Writing CSM reranked output")):
            query_id = extract_query_id(item)
            query = extract_query(item)
            ground_truth = extract_ground_truth(item)

            if index in score_map:
                ranked_candidates = score_map[index]
                selected_contexts = [candidate["text"] for candidate in ranked_candidates[:args.keep]]
                csm_scores = [candidate["csm_score"] for candidate in ranked_candidates[:args.keep]]
            else:
                candidates = extract_candidates(item)
                fallback_contexts = [get_candidate_text(candidate) for candidate in candidates[:args.keep]]
                selected_contexts = fallback_contexts
                csm_scores = []

            if len(selected_contexts) < args.keep:
                selected_contexts += [""] * (args.keep - len(selected_contexts))

            output_item = {
                "query_id": query_id,
                "query": query,
                "ground_truth": ground_truth,
                "selected_contexts": selected_contexts[:args.keep],
                "csm_scores": csm_scores,
                "score_method": "csm",
            }

            fout.write(json.dumps(output_item, ensure_ascii=False) + "\n")

    print(f"Saved CSM reranked output to: {args.output_jsonl}")


if __name__ == "__main__":
    main()