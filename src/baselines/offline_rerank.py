#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified offline reranking entry point for standard baseline methods.
"""

import argparse
import os
from typing import Any, Dict, List

from tqdm import tqdm

try:
    from .standard_rag import StandardRAGReranker
    from .bge_reranker import BGEReranker, BGERerankerConfig
    from .rankgpt import RankGPTReranker, RankGPTConfig
    from .rqrag import RQRAGReranker, RQRAGConfig
    from .recomp_ext import ReCompExtReranker, ReCompExtConfig
    from .recomp_abs import ReCompAbsReranker, ReCompAbsConfig
    from .utils import (
        ensure_dir,
        extract_candidates,
        extract_ground_truth,
        extract_query,
        extract_query_id,
        read_json_or_jsonl,
        write_jsonl_line,
    )
except ImportError:
    from standard_rag import StandardRAGReranker
    from bge_reranker import BGEReranker, BGERerankerConfig
    from rankgpt import RankGPTReranker, RankGPTConfig
    from rqrag import RQRAGReranker, RQRAGConfig
    from recomp_ext import ReCompExtReranker, ReCompExtConfig
    from recomp_abs import ReCompAbsReranker, ReCompAbsConfig
    from utils import (
        ensure_dir,
        extract_candidates,
        extract_ground_truth,
        extract_query,
        extract_query_id,
        read_json_or_jsonl,
        write_jsonl_line,
    )


def build_reranker(args):
    """
    Build a baseline reranker from argparse arguments.

    Passing the entire args object keeps this entry point extensible for
    model-based rerankers that require model paths, quantization options,
    decoding parameters, or batch sizes.
    """
    if args.method == "standard_rag":
        return StandardRAGReranker(
            top_k_input=args.top_k_input,
            top_k_output=args.top_k_output,
        )

    if args.method == "bge_reranker":
        if not args.model_path:
            raise ValueError("--model_path is required for method=bge_reranker.")

        config = BGERerankerConfig(
            model_name_or_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            fp16=not args.no_fp16,
        )

        return BGEReranker(
            config=config,
            top_k_input=args.top_k_input,
            top_k_output=args.top_k_output,
        )

    if args.method == "rankgpt":
        if not args.model_path:
            raise ValueError("--model_path is required for method=rankgpt.")

        config = RankGPTConfig(
            model_name_or_path=args.model_path,
            tokenizer_name_or_path=args.tokenizer_path,
            quantization_mode=args.quantization_mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_input_chars_per_doc=args.max_input_chars_per_doc,
            listwise_k=args.listwise_k,
        )

        return RankGPTReranker(
            config=config,
            top_k_input=args.top_k_input,
            top_k_output=args.top_k_output,
        )

    if args.method == "rqrag":
        if not args.model_path:
            raise ValueError("--model_path is required as the query-refinement LLM for method=rqrag.")
        if not args.bge_model_path:
            raise ValueError("--bge_model_path is required as the scorer for method=rqrag.")

        rqrag_config = RQRAGConfig(
            generator_model_name_or_path=args.model_path,
            generator_tokenizer_name_or_path=args.tokenizer_path,
            quantization_mode=args.quantization_mode,
            num_refined_queries=args.num_refined_queries,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            aggregation=args.rqrag_aggregation,
        )

        bge_config = BGERerankerConfig(
            model_name_or_path=args.bge_model_path,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            fp16=not args.no_fp16,
        )

        return RQRAGReranker(
            config=rqrag_config,
            bge_config=bge_config,
            top_k_input=args.top_k_input,
            top_k_output=args.top_k_output,
        )

    if args.method == "recomp_ext":
        if not args.model_path:
            raise ValueError("--model_path is required for method=recomp_ext.")

        config = ReCompExtConfig(
            model_name_or_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size,
            max_input_length=args.max_length,
            fp16=not args.no_fp16,
            use_top_n_docs=args.recomp_use_top_n_docs,
            num_sentences=args.recomp_num_sentences,
        )

        return ReCompExtReranker(
            config=config,
            top_k_input=args.top_k_input,
            top_k_output=args.top_k_output,
        )

    if args.method == "recomp_abs":
        if not args.model_path:
            raise ValueError("--model_path is required for method=recomp_abs.")

        config = ReCompAbsConfig(
            model_name_or_path=args.model_path,
            device=args.device,
            fp16=not args.no_fp16,
            use_top_n_docs=args.recomp_use_top_n_docs,
            max_input_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
        )

        return ReCompAbsReranker(
            config=config,
            top_k_input=args.top_k_input,
            top_k_output=args.top_k_output,
        )

    raise ValueError(f"Unknown reranking method: {args.method}")


def build_output_path(args) -> str:
    input_stem = os.path.splitext(os.path.basename(args.input))[0]
    filename = f"{input_stem}.{args.method}_top{args.top_k_output}.jsonl"
    return os.path.join(args.output_dir, filename)


def normalize_selected_contexts(selected_contexts: List[str], top_k_output: int) -> List[str]:
    if selected_contexts is None:
        selected_contexts = []

    selected_contexts = [str(context) if context is not None else "" for context in selected_contexts]

    if len(selected_contexts) < top_k_output:
        selected_contexts += [""] * (top_k_output - len(selected_contexts))
    else:
        selected_contexts = selected_contexts[:top_k_output]

    return selected_contexts


def rerank_item(
    item: Dict[str, Any],
    reranker,
    top_k_output: int,
    method: str,
) -> Dict[str, Any]:
    query_id = extract_query_id(item)
    query = extract_query(item)
    ground_truth = extract_ground_truth(item)
    candidates = extract_candidates(item)

    selected_contexts = reranker.rank(query, candidates)
    selected_contexts = normalize_selected_contexts(
        selected_contexts=selected_contexts,
        top_k_output=top_k_output,
    )

    return {
        "query_id": query_id,
        "query": query,
        "ground_truth": ground_truth,
        "selected_contexts": selected_contexts,
        "score_method": method,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Unified offline baseline reranking.")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input retrieval results in JSON or JSONL format.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save reranked JSONL output.",
    )

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=[
            "standard_rag",
            "bge_reranker",
            "rankgpt",
            "rqrag",
            "recomp_ext",
            "recomp_abs",
        ],
        help="Baseline reranking method.",
    )

    parser.add_argument(
        "--top_k_input",
        type=int,
        default=10,
        help="Candidate pool size used by the reranker.",
    )

    parser.add_argument(
        "--top_k_output",
        type=int,
        default=5,
        help="Number of selected contexts for final generation.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help=(
            "Model path for model-based baselines. "
            "For bge_reranker, this is the cross-encoder model. "
            "For rankgpt and rqrag, this is the LLM path. "
            "For recomp_ext and recomp_abs, this is the compressor/scorer model path."
        ),
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional tokenizer path for LLM-based baselines.",
    )

    parser.add_argument(
        "--bge_model_path",
        type=str,
        default="",
        help="BGE cross-encoder scorer path for RQ-RAG.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model-based rerankers.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for model-based rerankers.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for encoder/compressor inputs.",
    )

    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="Disable fp16/autocast where supported.",
    )

    parser.add_argument(
        "--quantization_mode",
        type=str,
        default="none",
        choices=["none", "bnb4", "mxfp4"],
        help="Quantization mode for LLM-based rerankers.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum generated tokens for LLM-based rerankers.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature for LLM-based rerankers.",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling value for LLM-based rerankers.",
    )

    parser.add_argument(
        "--max_input_chars_per_doc",
        type=int,
        default=1200,
        help="Maximum number of characters per document for RankGPT.",
    )

    parser.add_argument(
        "--listwise_k",
        type=int,
        default=20,
        help="Number of documents ranked listwise by RankGPT.",
    )

    parser.add_argument(
        "--num_refined_queries",
        type=int,
        default=3,
        help="Number of refined queries generated by RQ-RAG.",
    )

    parser.add_argument(
        "--rqrag_aggregation",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="Aggregation method over refined-query scores in RQ-RAG.",
    )

    parser.add_argument(
        "--recomp_use_top_n_docs",
        type=int,
        default=10,
        help="Number of retrieved documents used by ReComp baselines.",
    )

    parser.add_argument(
        "--recomp_num_sentences",
        type=int,
        default=5,
        help="Number of extracted sentences for ReComp-Ext.",
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Beam size for ReComp-Abs generation.",
    )

    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.0,
        help="Length penalty for ReComp-Abs generation.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    output_path = build_output_path(args)
    reranker = build_reranker(args)

    num_items = 0
    num_empty_candidates = 0
    num_empty_selected_contexts = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for item in tqdm(read_json_or_jsonl(args.input), desc=f"rerank[{args.method}]"):
            num_items += 1

            candidates = extract_candidates(item)
            if not candidates:
                num_empty_candidates += 1

            output_item = rerank_item(
                item=item,
                reranker=reranker,
                top_k_output=args.top_k_output,
                method=args.method,
            )

            if all(not str(context).strip() for context in output_item["selected_contexts"]):
                num_empty_selected_contexts += 1

            write_jsonl_line(fout, output_item)

    print(f"Saved reranked output to: {output_path}")
    print(f"Processed examples: {num_items}")
    print(f"Examples with empty candidate lists: {num_empty_candidates}")
    print(f"Examples with empty selected contexts: {num_empty_selected_contexts}")


if __name__ == "__main__":
    main()