#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from .base import BaseReranker
    from .utils import get_candidate_text
except ImportError:
    from base import BaseReranker
    from utils import get_candidate_text


@dataclass
class ReCompExtConfig:
    model_name_or_path: str
    device: str = "cuda"
    batch_size: int = 16
    max_input_length: int = 128
    fp16: bool = True
    use_top_n_docs: int = 10
    num_sentences: int = 5


class ReCompExtReranker(BaseReranker):
    def __init__(
        self,
        config: ReCompExtConfig,
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

        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
        )

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 5]

    @torch.no_grad()
    def score_sentences(self, query: str, sentences: List[str]) -> torch.Tensor:
        if not sentences:
            return torch.empty(0)

        all_scores = []
        use_amp = bool(self.config.fp16 and self.device.type == "cuda")

        for start in range(0, len(sentences), self.config.batch_size):
            batch_sentences = sentences[start:start + self.config.batch_size]
            pairs = [[query, sentence] for sentence in batch_sentences]

            encoded = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.config.max_input_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(**encoded)
            else:
                output = self.model(**encoded)

            logits = output.logits

            if logits.dim() == 2 and logits.size(-1) >= 2:
                scores = logits[:, 1]
            else:
                scores = logits.view(-1)

            all_scores.append(scores.detach().float().cpu())

        return torch.cat(all_scores, dim=0)

    def compress(self, query: str, docs: List[str]) -> str:
        context = "\n\n".join([doc for doc in docs if doc and doc.strip()])
        sentences = self.split_sentences(context)

        if not sentences:
            return ""

        scores = self.score_sentences(query, sentences)
        if scores.numel() == 0:
            return ""

        num_select = min(self.config.num_sentences, len(sentences))
        top_indices = torch.topk(scores, k=num_select).indices.tolist()
        top_indices = sorted(top_indices)

        selected_sentences = [sentences[idx] for idx in top_indices]
        return " ".join(selected_sentences)

    def rank(self, query: str, candidates: List[Dict[str, Any]]) -> List[str]:
        candidate_pool = candidates[: self.top_k_input]
        docs = [get_candidate_text(candidate) for candidate in candidate_pool]
        docs = docs[: self.config.use_top_n_docs]

        compressed_context = self.compress(query, docs)

        if not compressed_context:
            return [""] * self.top_k_output

        selected = [compressed_context]
        if len(selected) < self.top_k_output:
            selected += [""] * (self.top_k_output - len(selected))

        return selected[: self.top_k_output]