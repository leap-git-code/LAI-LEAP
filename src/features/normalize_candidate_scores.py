#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_scores(input_dir: str, score_key: str = "final_score", output_key: str = "final_score_normal"):
    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".json")
    ]

    by_query_id = defaultdict(list)

    for file_path in files:
        obj = load_json(file_path)
        query_id = obj.get("query_id")
        if query_id is None:
            continue
        by_query_id[str(query_id)].append(file_path)

    for query_id, file_paths in tqdm(by_query_id.items(), desc="Normalizing"):
        objects = []
        scores = []

        for file_path in file_paths:
            obj = load_json(file_path)
            score = float(obj.get(score_key, 0.0))
            objects.append((file_path, obj))
            scores.append(score)

        max_score = max(scores) if scores else 0.0

        for file_path, obj in objects:
            score = float(obj.get(score_key, 0.0))
            obj[output_key] = score / max_score if max_score > 0 else 0.0
            save_json(obj, file_path)

    print(f"Normalization finished. Processed {len(by_query_id)} queries.")


def main():
    parser = argparse.ArgumentParser(description="Normalize candidate scores within each query group.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing candidate score JSON files.")
    parser.add_argument("--score_key", type=str, default="final_score", help="Raw score field name.")
    parser.add_argument("--output_key", type=str, default="final_score_normal", help="Normalized score field name.")

    args = parser.parse_args()

    normalize_scores(
        input_dir=args.input_dir,
        score_key=args.score_key,
        output_key=args.output_key,
    )


if __name__ == "__main__":
    main()