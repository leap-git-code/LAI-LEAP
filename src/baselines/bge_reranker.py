#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from .base import BaseReranker
    from .utils import get_candidate_text
except ImportError:
    from base import BaseReranker
    from utils import get_candidate_text


class PairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.pairs[idx]


@dataclass
class BGERerankerConfig:
    model_name_or_path: str
    device: str = "cuda"
    batch_size: int = 32
    max_length: int = 512
    fp16: bool = True


class BGEReranker(BaseReranker):
    def __init__(
        self,
        config: BGERerankerConfig,
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

    def _collate_pairs(self, batch: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        queries, docs = zip(*batch)

        return self.tokenizer(
            list(queries),
            list(docs),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

    @torch.no_grad()
    def score_pairs(self, query: str, docs: List[str]) -> np.ndarray:
        pairs = [(query, doc) for doc in docs]
        dataset = PairDataset(pairs)

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_pairs,
        )

        all_scores = []
        use_amp = bool(self.config.fp16 and self.device.type == "cuda")

        for encoded in loader:
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(**encoded)
            else:
                output = self.model(**encoded)

            logits = output.logits

            if logits.dim() == 2 and logits.size(-1) == 1:
                scores = logits.squeeze(-1)
            elif logits.dim() == 2 and logits.size(-1) >= 2:
                scores = logits[:, 1]
            else:
                scores = logits.view(-1)

            all_scores.append(scores.detach().float().cpu())

        if not all_scores:
            return np.array([], dtype=np.float32)

        return torch.cat(all_scores, dim=0).numpy().astype(np.float32)

    def rank(self, query: str, candidates: List[Dict[str, Any]]) -> List[str]:
        candidate_pool = candidates[: self.top_k_input]
        docs = [get_candidate_text(candidate) for candidate in candidate_pool]

        valid_docs = [doc for doc in docs if isinstance(doc, str) and doc.strip()]

        if not valid_docs:
            return [""] * self.top_k_output

        scores = self.score_pairs(query, valid_docs)
        order = np.argsort(-scores)

        selected = [valid_docs[int(idx)] for idx in order[: self.top_k_output]]

        if len(selected) < self.top_k_output:
            selected += [""] * (self.top_k_output - len(selected))

        return selected