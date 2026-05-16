#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Retrieve candidate contexts from a FAISS index and a SQLite context store.

Input:
- Query file in JSON or JSONL format.

Output:
- JSON or JSONL file containing:
  query_id
  query
  ground_truth
  candidates
"""

import argparse
import json
import os
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


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


def write_json_or_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)


def extract_query(item: Dict[str, Any]) -> str:
    for key in ["query", "question", "input", "prompt"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_query_id(item: Dict[str, Any], fallback_index: int) -> str:
    for key in ["query_id", "qid", "id", "question_id", "uid"]:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(fallback_index)


def extract_ground_truth(item: Dict[str, Any]) -> Any:
    for key in ["ground_truth", "groud_truth", "answers", "answer", "gold", "golden_answers"]:
        if key in item:
            return item[key]
    return []


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def list_sqlite_tables(conn: sqlite3.Connection) -> List[str]:
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [row[0] for row in cursor.fetchall()]


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]


def choose_context_table(conn: sqlite3.Connection) -> Tuple[str, List[str]]:
    candidate_text_columns = [
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
    ]

    tables = list_sqlite_tables(conn)

    for table in tables:
        columns = get_table_columns(conn, table)
        if any(col in columns for col in candidate_text_columns):
            return table, columns

    if tables:
        table = tables[0]
        return table, get_table_columns(conn, table)

    raise ValueError("No table was found in the SQLite database.")


def choose_id_column(columns: List[str]) -> Optional[str]:
    for key in ["id", "doc_id", "context_id", "idx", "faiss_id"]:
        if key in columns:
            return key
    return None


def choose_text_column(columns: List[str]) -> Optional[str]:
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
    ]:
        if key in columns:
            return key
    return None


def choose_title_column(columns: List[str]) -> Optional[str]:
    for key in ["title", "doc_title", "name"]:
        if key in columns:
            return key
    return None


class SQLiteContextStore:
    def __init__(self, sqlite_path: str):
        self.conn = sqlite3.connect(sqlite_path)
        self.table_name, self.columns = choose_context_table(self.conn)
        self.id_column = choose_id_column(self.columns)
        self.text_column = choose_text_column(self.columns)
        self.title_column = choose_title_column(self.columns)

        if self.text_column is None:
            raise ValueError(
                f"Could not find a text column in table '{self.table_name}'. "
                f"Available columns: {self.columns}"
            )

    def get(self, faiss_id: int) -> Dict[str, Any]:
        if self.id_column is not None:
            query = (
                f"SELECT * FROM {self.table_name} "
                f"WHERE {self.id_column} = ? "
                f"LIMIT 1"
            )
            cursor = self.conn.execute(query, (int(faiss_id),))
        else:
            query = (
                f"SELECT * FROM {self.table_name} "
                f"WHERE rowid = ? "
                f"LIMIT 1"
            )
            cursor = self.conn.execute(query, (int(faiss_id) + 1,))

        row = cursor.fetchone()

        if row is None:
            return {
                "id": int(faiss_id),
                "title": "",
                "contents": "",
            }

        obj = dict(zip(self.columns, row))
        text = obj.get(self.text_column, "")
        title = obj.get(self.title_column, "") if self.title_column else ""

        return {
            "id": int(faiss_id),
            "title": str(title) if title is not None else "",
            "contents": str(text) if text is not None else "",
        }

    def close(self) -> None:
        self.conn.close()


def encode_queries(
    model: SentenceTransformer,
    queries: List[str],
    batch_size: int,
    normalize: bool,
    query_prefix: str,
) -> np.ndarray:
    formatted_queries = [
        query_prefix + query if query_prefix else query
        for query in queries
    ]

    embeddings = model.encode(
        formatted_queries,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )

    embeddings = embeddings.astype("float32")

    if normalize:
        embeddings = normalize_embeddings(embeddings).astype("float32")

    return embeddings


def retrieve(
    index,
    store: SQLiteContextStore,
    query_embeddings: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    scores, indices = index.search(query_embeddings, top_k)
    return scores, indices


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieve candidate contexts with FAISS.")

    parser.add_argument("--index_path", type=str, required=True, help="Path to FAISS index.")
    parser.add_argument("--sqlite_path", type=str, required=True, help="Path to SQLite context store.")
    parser.add_argument("--model_path", type=str, required=True, help="SentenceTransformer retriever path.")

    parser.add_argument("--query_jsonl", type=str, required=True, help="Input query file in JSON or JSONL format.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output file in JSON or JSONL format.")

    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--normalize_embeddings", action="store_true")
    parser.add_argument("--query_prefix", type=str, default="query: ")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading queries from: {args.query_jsonl}")
    data = read_json_or_jsonl(args.query_jsonl)

    queries = []
    metas = []

    for idx, item in enumerate(data):
        query = extract_query(item)
        if not query:
            continue

        queries.append(query)
        metas.append(
            {
                "query_id": extract_query_id(item, idx),
                "query": query,
                "ground_truth": extract_ground_truth(item),
            }
        )

    if not queries:
        raise RuntimeError("No valid queries were found.")

    print(f"Loading retriever model from: {args.model_path}")
    model = SentenceTransformer(args.model_path)

    print(f"Loading FAISS index from: {args.index_path}")
    index = faiss.read_index(args.index_path)

    print(f"Loading SQLite context store from: {args.sqlite_path}")
    store = SQLiteContextStore(args.sqlite_path)

    print(f"Encoding {len(queries)} queries...")
    query_embeddings = encode_queries(
        model=model,
        queries=queries,
        batch_size=args.batch_size,
        normalize=args.normalize_embeddings,
        query_prefix=args.query_prefix,
    )

    print(f"Retrieving top-{args.top_k} contexts...")
    scores, indices = retrieve(
        index=index,
        store=store,
        query_embeddings=query_embeddings,
        top_k=args.top_k,
    )

    output_data = []

    for meta, row_scores, row_indices in tqdm(
        zip(metas, scores, indices),
        total=len(metas),
        desc="Building retrieval output",
    ):
        candidates = []

        for rank, (score, faiss_id) in enumerate(zip(row_scores, row_indices)):
            if int(faiss_id) < 0:
                continue

            candidate = store.get(int(faiss_id))
            candidate["score"] = float(score)
            candidate["rank"] = int(rank)
            candidates.append(candidate)

        output_data.append(
            {
                "query_id": meta["query_id"],
                "query": meta["query"],
                "ground_truth": meta["ground_truth"],
                "candidates": candidates,
            }
        )

    store.close()

    print(f"Saving retrieval output to: {args.output_jsonl}")
    write_json_or_jsonl(output_data, args.output_jsonl)

    print("Done.")


if __name__ == "__main__":
    main()