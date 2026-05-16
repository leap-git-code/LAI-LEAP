#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from .base import BaseReranker
    from .utils import get_candidate_text, clean_whitespace
    from .llm_utils import LocalCausalLM
    from .bge_reranker import BGEReranker, BGERerankerConfig
except ImportError:
    from base import BaseReranker
    from utils import get_candidate_text, clean_whitespace
    from llm_utils import LocalCausalLM
    from bge_reranker import BGEReranker, BGERerankerConfig


@dataclass
class RQRAGConfig:
    generator_model_name_or_path: str
    generator_tokenizer_name_or_path: Optional[str] = None
    quantization_mode: str = "none"
    num_refined_queries: int = 3
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 0.9
    aggregation: str = "max"


class RQRAGReranker(BaseReranker):
    def __init__(
        self,
        config: RQRAGConfig,
        bge_config: BGERerankerConfig,
        top_k_input: int = 10,
        top_k_output: int = 5,
    ):
        super().__init__(top_k_input=top_k_input, top_k_output=top_k_output)

        self.config = config

        self.generator = LocalCausalLM(
            model_name_or_path=config.generator_model_name_or_path,
            tokenizer_name_or_path=config.generator_tokenizer_name_or_path,
            quantization_mode=config.quantization_mode,
        )

        self.scorer = BGEReranker(
            config=bge_config,
            top_k_input=top_k_input,
            top_k_output=top_k_output,
        )

    def _build_prompt(self, query: str) -> str:
        system_prompt = (
            "You are a query refinement module for retrieval-augmented generation. "
            "Generate refined search queries that improve retrieval quality. "
            "Return one refined query per line. Do not include numbering or explanations."
        )

        user_prompt = (
            f"Question: {query}\n\n"
            f"Generate at most {self.config.num_refined_queries} refined queries."
        )

        return self.generator.apply_chat_template(system_prompt, user_prompt)

    def _parse_refined_queries(self, text: str) -> List[str]:
        if not text:
            return []

        lines = [clean_whitespace(line) for line in text.splitlines()]
        lines = [line for line in lines if line]

        refined_queries = []
        seen = set()

        for line in lines:
            line = re.sub(r"^\s*[\(\[]?\d+[\)\].:\-]\s*", "", line).strip()
            line = re.sub(r"^\s*[-*]\s*", "", line).strip()

            if not line:
                continue

            key = line.lower()
            if key not in seen:
                seen.add(key)
                refined_queries.append(line)

        return refined_queries[: max(1, self.config.num_refined_queries)]

    def refine_queries(self, query: str) -> List[str]:
        prompt = self._build_prompt(query)

        try:
            generated = self.generator.generate_text(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            refined = self._parse_refined_queries(generated)
        except Exception:
            refined = []

        if not refined:
            refined = [query]

        return refined

    def _aggregate_scores(self, score_matrix: np.ndarray) -> np.ndarray:
        aggregation = (self.config.aggregation or "max").lower()
        if aggregation == "mean":
            return score_matrix.mean(axis=0)
        return score_matrix.max(axis=0)

    def rank(self, query: str, candidates: List[Dict[str, Any]]) -> List[str]:
        candidate_pool = candidates[: self.top_k_input]
        docs = [get_candidate_text(candidate) for candidate in candidate_pool]
        docs = [doc for doc in docs if isinstance(doc, str) and doc.strip()]

        if not docs:
            return [""] * self.top_k_output

        refined_queries = self.refine_queries(query)

        score_rows = []
        for refined_query in refined_queries:
            scores = self.scorer.score_pairs(refined_query, docs)
            score_rows.append(scores)

        if not score_rows:
            selected = docs[: self.top_k_output]
        else:
            score_matrix = np.stack(score_rows, axis=0)
            scores = self._aggregate_scores(score_matrix)
            order = np.argsort(-scores)
            selected = [docs[int(idx)] for idx in order[: self.top_k_output]]

        if len(selected) < self.top_k_output:
            selected += [""] * (self.top_k_output - len(selected))

        return selected