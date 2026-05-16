#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Any, Dict, List

try:
    from .base import BaseReranker
    from .utils import get_candidate_text
except ImportError:
    from base import BaseReranker
    from utils import get_candidate_text


class StandardRAGReranker(BaseReranker):
    def __init__(self, top_k_input: int = 10, top_k_output: int = 5):
        super().__init__(top_k_input=top_k_input, top_k_output=top_k_output)

    def rank(self, query: str, candidates: List[Dict[str, Any]]) -> List[str]:
        candidate_pool = candidates[: self.top_k_input]
        selected = candidate_pool[: self.top_k_output]
        return [get_candidate_text(candidate) for candidate in selected]