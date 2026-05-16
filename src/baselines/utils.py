#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for offline baseline reranking.
"""

import json
import os
import re
from typing import Any, Dict, Iterable, List, TextIO


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


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


def normalize_ground_truth(ground_truth: Any) -> List[str]:
    if ground_truth is None:
        return []

    if isinstance(ground_truth, list):
        output: List[str] = []
        for item in ground_truth:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                output.append(text)
        return output

    if isinstance(ground_truth, str):
        text = ground_truth.strip()
        if not text:
            return []

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return normalize_ground_truth(parsed)
            except Exception:
                pass

        if "|" in text:
            return [part.strip() for part in text.split("|") if part.strip()]

        return [text]

    if isinstance(ground_truth, (int, float, bool)):
        return [str(ground_truth)]

    if isinstance(ground_truth, dict):
        for key in ["answers", "gold_answers", "golden_answers", "ground_truth", "answer", "text"]:
            if key in ground_truth:
                return normalize_ground_truth(ground_truth[key])
        return [json.dumps(ground_truth, ensure_ascii=False)]

    return [str(ground_truth)]


def extract_ground_truth(item: Dict[str, Any]) -> List[str]:
    for key in ["ground_truth", "groud_truth", "answers", "answer", "gold", "golden_answers"]:
        if key in item:
            return normalize_ground_truth(item[key])
    return []


def get_candidate_text(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate

    if not isinstance(candidate, dict):
        return str(candidate)

    text_keys = [
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
    ]

    for key in text_keys:
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return ""


def extract_candidates(item: Dict[str, Any]) -> List[Any]:
    candidate_keys = [
        "candidates",
        "ctxs",
        "retrieved_results",
        "retrieved_contexts",
        "selected_contexts",
        "contexts",
        "docs",
        "documents",
        "passages",
    ]

    for key in candidate_keys:
        value = item.get(key)
        if isinstance(value, list):
            return value

    return []


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def strip_document_text(text: str) -> str:
    """
    Strip leading and trailing spaces while preserving internal structure.

    This is safer for LLM-based rerankers, because tables, bullet lists,
    code blocks, and line breaks may carry useful structure.
    """
    return str(text).strip()


def read_json_or_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
            return

        data = json.load(f)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
    elif isinstance(data, dict):
        yield data
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")


def write_jsonl_line(file_obj: TextIO, item: Dict[str, Any]) -> None:
    file_obj.write(json.dumps(item, ensure_ascii=False) + "\n")