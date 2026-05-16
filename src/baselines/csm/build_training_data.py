#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build CSM training data from aligned retrieval candidates.

The output is a parquet dataset. Each group contains:
- one query item
- top-k candidate contexts

The query label is a placeholder and context labels are normalized from
candidate scores within the same query group.
"""

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig


def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]

    raise ValueError(f"Unsupported JSON structure in: {path}")


def extract_query(item: Dict[str, Any]) -> str:
    for key in ["query", "question", "input", "prompt"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_candidates(item: Dict[str, Any]) -> List[Dict[str, Any]]:
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


def ensure_k_candidates(item: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
    candidates = extract_candidates(item)
    if not isinstance(candidates, list):
        return []
    return candidates[:k]


def make_labels_from_scores(scores: List[float]) -> List[float]:
    values = np.array(scores, dtype=np.float32)

    if len(values) == 0:
        return []

    values = values - float(values.mean())
    max_abs = float(np.max(np.abs(values))) if np.max(np.abs(values)) > 1e-8 else 0.0

    if max_abs < 1e-8:
        return [0.0 for _ in scores]

    labels = values / max_abs
    labels = np.clip(labels, -1.0, 1.0)

    return labels.astype(np.float32).tolist()


def build_rows(
    data: List[Dict[str, Any]],
    tokenizer,
    use_token_type: bool,
    top_k: int,
    max_len: int,
    score_field: str,
) -> List[Dict[str, Any]]:
    rows = []
    group_size = top_k + 1
    skipped = 0

    for item in tqdm(data, desc="Building CSM training groups"):
        query = extract_query(item)
        if not query:
            skipped += 1
            continue

        candidates = ensure_k_candidates(item, top_k)
        if len(candidates) < top_k:
            skipped += 1
            continue

        context_texts = []
        context_scores = []
        valid = True

        for candidate in candidates:
            text = get_candidate_text(candidate)

            if not text:
                valid = False
                break

            if not isinstance(candidate, dict):
                valid = False
                break

            score = candidate.get(score_field, None)
            try:
                score = float(score)
            except Exception:
                valid = False
                break

            context_texts.append(text)
            context_scores.append(score)

        if not valid:
            skipped += 1
            continue

        context_labels = make_labels_from_scores(context_scores)

        texts = [query] + context_texts
        labels = [0.0] + context_labels

        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )

        for index in range(group_size):
            row = {
                "input_ids": encoded["input_ids"][index],
                "attention_mask": encoded["attention_mask"][index],
                "labels": labels[index],
            }

            if use_token_type and "token_type_ids" in encoded:
                row["token_type_ids"] = encoded["token_type_ids"][index]

            rows.append(row)

    if len(rows) % group_size != 0:
        cut = len(rows) - (len(rows) % group_size)
        rows = rows[:cut]

    print(f"Built rows: {len(rows)}")
    print(f"Built groups: {len(rows) // group_size}")
    print(f"Skipped examples: {skipped}")

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Build CSM parquet training data.")

    parser.add_argument("--input_json", type=str, required=True, help="Aligned training JSON file.")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory for train_ds.parquet.")
    parser.add_argument("--bert_path", type=str, required=True, help="Path or HF name of the CSM encoder.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of candidate contexts per query.")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--score_field", type=str, default="score")

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_path, use_fast=True)
    bert_config = AutoConfig.from_pretrained(args.bert_path)
    use_token_type = bool(getattr(bert_config, "type_vocab_size", 0) and bert_config.type_vocab_size > 1)

    data = read_json(args.input_json)

    rows = build_rows(
        data=data,
        tokenizer=tokenizer,
        use_token_type=use_token_type,
        top_k=args.top_k,
        max_len=args.max_len,
        score_field=args.score_field,
    )

    if len(rows) == 0:
        raise RuntimeError("No CSM training samples were generated. Please check input_json, top_k, and score_field.")

    dataset = Dataset.from_list(rows)
    output_path = os.path.join(args.save_dir, "train_ds.parquet")
    dataset.to_parquet(output_path)

    print(f"Saved CSM training data to: {output_path}")


if __name__ == "__main__":
    main()