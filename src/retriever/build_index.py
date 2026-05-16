#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build FAISS IVF-PQ index for large-scale corpus.

This script constructs a vector index over a large text corpus
(e.g., Wikipedia) using E5 embeddings and FAISS IVF-PQ.

"""

import os
import json
import time
import argparse
import sqlite3
import gc
from typing import Dict, Any, Iterator

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
import faiss


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def jsonl_stream(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


@torch.no_grad()
def embed_passages(texts, tokenizer, model, device, batch_size, max_length):
    model.eval()
    outputs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # E5 model requirement: prefix with 'passage: '
        batch = ["passage: " + t for t in batch]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        out = model(**enc)
        hidden = out.last_hidden_state

        # Mean pooling based on attention mask
        mask = enc["attention_mask"].unsqueeze(-1)
        emb = (hidden * mask).sum(1) / mask.sum(1)

        # L2 normalization for cosine similarity via inner product
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        outputs.append(emb.cpu().numpy())

    return np.concatenate(outputs, axis=0).astype(np.float32)


def init_sqlite(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS contexts (
        vec_id INTEGER PRIMARY KEY,
        doc_id TEXT,
        contents TEXT
    )
    """)
    return conn


def build_faiss_index(config: Dict):
    data_cfg = config["data"]
    retriever_cfg = config["retriever"]

    jsonl_path = data_cfg["corpus_path"]
    out_dir = retriever_cfg["index_dir"]
    model_path = retriever_cfg["embedding_model"]

    embed_bs = retriever_cfg.get("embed_batch_size", 256)
    max_length = retriever_cfg.get("max_length", 512)

    # FAISS IVF-PQ Hyperparameters
    nlist = retriever_cfg.get("nlist", 4096)
    m_pq = retriever_cfg.get("m_pq", 64)
    nbits = retriever_cfg.get("nbits", 8)

    # Hard cap for batching to prevent RAM overflow
    add_bs = retriever_cfg.get("add_batch_size", 20000)

    ensure_dir(out_dir)

    index_path = os.path.join(out_dir, "faiss.index")
    db_path = os.path.join(out_dir, "contexts.sqlite")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[{now()}] Loading Encoder: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)

    # 1. Sample for IVF Training
    print(f"[{now()}] Collecting samples for IVF training...")
    train_texts = []
    for i, obj in enumerate(jsonl_stream(jsonl_path)):
        if i >= 50000:
            break
        train_texts.append(obj.get("contents", ""))

    train_embeddings = embed_passages(
        train_texts,
        tokenizer,
        model,
        device,
        embed_bs,
        max_length
    )

    dim = train_embeddings.shape[1]
    print(f"[{now()}] Vector dimension: {dim}")

    # Initialize IndexIVFPQ with Inner Product metric
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(
        quantizer,
        dim,
        nlist,
        m_pq,
        nbits,
        faiss.METRIC_INNER_PRODUCT
    )

    print(f"[{now()}] Training IVF centroids...")
    index.train(train_embeddings)

    del train_texts, train_embeddings
    gc.collect()

    # 2. Streaming Build
    conn = init_sqlite(db_path)
    cur = conn.cursor()

    buffer_texts = []
    buffer_meta = []
    total_added = 0

    print(f"[{now()}] Processing corpus in streams...")
    for i, obj in enumerate(tqdm(jsonl_stream(jsonl_path))):
        content = obj.get("contents", "")
        if not content:
            continue

        buffer_texts.append(content)
        # Ensure FAISS internal ID matches SQLite Primary Key (vec_id)
        buffer_meta.append((
            total_added + len(buffer_texts) - 1,
            obj.get("id", str(i)),
            content
        ))

        if len(buffer_texts) >= add_bs:
            embeddings = embed_passages(
                buffer_texts,
                tokenizer,
                model,
                device,
                embed_bs,
                max_length
            )

            index.add(embeddings)
            cur.executemany("INSERT INTO contexts VALUES (?, ?, ?)", buffer_meta)
            conn.commit()

            total_added += len(buffer_texts)
            buffer_texts, buffer_meta = [], []
            gc.collect()

    # 3. Process remaining items
    if buffer_texts:
        embeddings = embed_passages(buffer_texts, tokenizer, model, device, embed_bs, max_length)
        index.add(embeddings)
        cur.executemany("INSERT INTO contexts VALUES (?, ?, ?)", buffer_meta)
        conn.commit()
        total_added += len(buffer_texts)

    # 4. Persistence
    print(f"[{now()}] Saving FAISS index to {index_path}")
    faiss.write_index(index, index_path)
    conn.close()

    print(f"[{now()}] Build completed. Total contexts in C: {total_added}")



def main():
    parser = argparse.ArgumentParser(description="Build RAG retrieval index.")
    parser.add_argument("--corpus_path", required=True, help="Path to JSONL corpus file.")
    parser.add_argument("--index_dir", required=True, help="Output directory for index and database.")
    parser.add_argument("--model_path", required=True, help="Local path or HF hub ID of the embedding model.")

    args = parser.parse_args()

    config = {
        "data": {
            "corpus_path": args.corpus_path
        },
        "retriever": {
            "index_dir": args.index_dir,
            "embedding_model": args.model_path,
            "nlist": 4096,
            "m_pq": 64,
            "nbits": 8
        }
    }

    build_faiss_index(config)


if __name__ == "__main__":
    main()