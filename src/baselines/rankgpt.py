#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from .base import BaseReranker
    from .utils import get_candidate_text, strip_document_text
    from .llm_utils import LocalCausalLM
except ImportError:
    from base import BaseReranker
    from utils import get_candidate_text, strip_document_text
    from llm_utils import LocalCausalLM


@dataclass
class RankGPTConfig:
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = None
    quantization_mode: str = "none"
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 0.9
    max_input_chars_per_doc: int = 1200
    listwise_k: int = 20


class RankGPTReranker(BaseReranker):
    def __init__(
        self,
        config: RankGPTConfig,
        top_k_input: int = 10,
        top_k_output: int = 5,
    ):
        super().__init__(top_k_input=top_k_input, top_k_output=top_k_output)

        self.config = config
        self.generator = LocalCausalLM(
            model_name_or_path=config.model_name_or_path,
            tokenizer_name_or_path=config.tokenizer_name_or_path,
            quantization_mode=config.quantization_mode,
        )

    def _build_user_prompt(self, query: str, docs: List[str]) -> str:
        lines = []
        lines.append(f"Question: {query}")
        lines.append("")
        lines.append("Candidate documents:")

        for index, doc in enumerate(docs, start=1):
            doc = strip_document_text(doc)
            if self.config.max_input_chars_per_doc > 0:
                doc = doc[: self.config.max_input_chars_per_doc]
            lines.append(f"[{index}] {doc}")

        lines.append("")
        lines.append(
            "Rank the documents by their relevance to the question. "
            "Return ONLY the ranked document indices as a comma-separated list, "
            "for example: 3,1,5,2."
        )

        return "\n".join(lines)

    def _build_prompt(self, query: str, docs: List[str]) -> str:
        system_prompt = (
            "You are a ranking assistant for retrieval-augmented generation. "
            "Your task is to rank candidate documents by relevance to the question. "
            "Output only document indices. Do not provide explanations."
        )
        user_prompt = self._build_user_prompt(query, docs)
        return self.generator.apply_chat_template(system_prompt, user_prompt)

    @staticmethod
    def _parse_indices(text: str, num_docs: int) -> Optional[List[int]]:
        if not text:
            return None

        numbers = re.findall(r"\d+", text)
        if not numbers:
            return None

        indices = []
        seen = set()

        for number in numbers:
            value = int(number)
            if 1 <= value <= num_docs and value not in seen:
                seen.add(value)
                indices.append(value - 1)

        return indices if indices else None

    def _rank_listwise(self, query: str, docs: List[str]) -> List[int]:
        prompt = self._build_prompt(query, docs)

        try:
            output_text = self.generator.generate_text(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            parsed = self._parse_indices(output_text, num_docs=len(docs))
        except Exception:
            parsed = None

        if parsed is None:
            parsed = list(range(len(docs)))

        remaining = [idx for idx in range(len(docs)) if idx not in set(parsed)]
        return parsed + remaining

    def rank(self, query: str, candidates: List[Dict[str, Any]]) -> List[str]:
        candidate_pool = candidates[: self.top_k_input]
        docs = [get_candidate_text(candidate) for candidate in candidate_pool]
        docs = [doc for doc in docs if isinstance(doc, str) and doc.strip()]

        if not docs:
            return [""] * self.top_k_output

        listwise_k = min(self.config.listwise_k, len(docs))
        head_docs = docs[:listwise_k]
        tail_docs = docs[listwise_k:]

        order_head = self._rank_listwise(query, head_docs)
        ranked_docs = [head_docs[idx] for idx in order_head] + tail_docs

        selected = ranked_docs[: self.top_k_output]
        if len(selected) < self.top_k_output:
            selected += [""] * (self.top_k_output - len(selected))

        return selected