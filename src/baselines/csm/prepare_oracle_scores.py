#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, Iterable, List


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


def load_oracle_scores(path: str) -> Dict[str, List[float]]:
    score_map: Dict[str, List[float]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            query_id = str(item.get("query_id", "")).strip()

            scores = item.get("deltasim_scores", None)
            if scores is None:
                scores = item.get("oracle_scores", None)
            if scores is None:
                scores = item.get("scores", None)

            if not query_id or not isinstance(scores, list):
                continue

            parsed_scores = []
            for score in scores:
                try:
                    parsed_scores.append(float(score))
                except Exception:
                    parsed_scores.append(0.0)

            score_map[query_id] = parsed_scores

    return score_map


def extract_query_id(item: Dict[str, Any]) -> str:
    for key in ["query_id", "qid", "id", "question_id", "uid"]:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
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


def align_scores(
    retrieval_data: List[Dict[str, Any]],
    score_map: Dict[str, List[float]],
    top_k: int,
    score_field: str,
) -> List[Dict[str, Any]]:
    aligned_data = []
    skipped = 0

    for item in retrieval_data:
        query_id = extract_query_id(item)
        if query_id not in score_map:
            skipped += 1
            continue

        candidates = extract_candidates(item)
        if not candidates:
            skipped += 1
            continue

        scores = score_map[query_id]

        for index, score in enumerate(scores[:top_k]):
            if index < len(candidates) and isinstance(candidates[index], dict):
                candidates[index][score_field] = float(score)

        aligned_data.append(item)

    print(f"Aligned examples: {len(aligned_data)}")
    print(f"Skipped examples: {skipped}")

    return aligned_data


def parse_args():
    parser = argparse.ArgumentParser(description="Align oracle scores with retrieval candidates for CSM training.")

    parser.add_argument("--input_file", type=str, required=True, help="Retrieval data in JSON or JSONL format.")
    parser.add_argument("--oracle_file", type=str, required=True, help="Oracle score JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Output aligned JSON file.")

    parser.add_argument("--top_k", type=int, default=10, help="Number of candidates to align.")
    parser.add_argument("--score_field", type=str, default="score", help="Candidate score field name.")

    return parser.parse_args()


def main():
    args = parse_args()

    retrieval_data = read_json_or_jsonl(args.input_file)
    score_map = load_oracle_scores(args.oracle_file)

    aligned_data = align_scores(
        retrieval_data=retrieval_data,
        score_map=score_map,
        top_k=args.top_k,
        score_field=args.score_field,
    )

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(aligned_data, f, ensure_ascii=False, indent=2)

    print(f"Saved aligned data to: {args.output_file}")


if __name__ == "__main__":
    main()