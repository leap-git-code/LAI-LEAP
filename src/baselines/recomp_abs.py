#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    from .base import BaseReranker
    from .utils import get_candidate_text, clean_whitespace
except ImportError:
    from base import BaseReranker
    from utils import get_candidate_text, clean_whitespace


@dataclass
class ReCompAbsConfig:
    model_name_or_path: str
    device: str = "cuda"
    fp16: bool = True
    use_top_n_docs: int = 5
    max_input_length: int = 1024
    max_new_tokens: int = 256
    num_beams: int = 4
    length_penalty: float = 1.0


class ReCompAbsReranker(BaseReranker):
    def __init__(
        self,
        config: ReCompAbsConfig,
        top_k_input: int = 10,
        top_k_output: int = 5,
    ):
        super().__init__(top_k_input=top_k_input, top_k_output=top_k_output)

        self.config = config

        if config.device == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            use_fast=True,
            trust_remote_code=True,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
        )

        self.model.to(self.device)
        self.model.eval()

        if config.fp16 and self.device.type == "cuda":
            self.model.half()

    @staticmethod
    def build_compressor_input(query: str, context: str) -> str:
        return f"Question: {query} Context: {context}"

    @torch.no_grad()
    def compress_batch(self, queries: List[str], contexts: List[str]) -> List[str]:
        inputs_text = [
            self.build_compressor_input(query, context)
            for query, context in zip(queries, contexts)
        ]

        encoded = self.tokenizer(
            inputs_text,
            padding=True,
            truncation=True,
            max_length=self.config.max_input_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        outputs = self.model.generate(
            **encoded,
            max_new_tokens=self.config.max_new_tokens,
            num_beams=self.config.num_beams,
            early_stopping=True,
            length_penalty=self.config.length_penalty,
        )

        summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [summary.strip() for summary in summaries]

    def compress(self, query: str, context: str) -> str:
        return self.compress_batch([query], [context])[0]

    def rank(self, query: str, candidates: List[Dict[str, Any]]) -> List[str]:
        candidate_pool = candidates[: self.top_k_input]

        docs = [
            get_candidate_text(candidate)
            for candidate in candidate_pool[: self.config.use_top_n_docs]
        ]
        docs = [clean_whitespace(doc) for doc in docs if doc and doc.strip()]

        if not docs:
            return [""] * self.top_k_output

        concat_context = " ".join(docs)
        summary = self.compress(query, concat_context)

        if not summary:
            return [""] * self.top_k_output

        selected = [summary]

        if len(selected) < self.top_k_output:
            selected += [""] * (self.top_k_output - len(selected))

        return selected[: self.top_k_output]