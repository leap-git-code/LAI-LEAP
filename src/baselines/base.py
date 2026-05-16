#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base interface for offline rerankers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseReranker(ABC):
    def __init__(self, top_k_input: int = 10, top_k_output: int = 5):
        self.top_k_input = top_k_input
        self.top_k_output = top_k_output

    @abstractmethod
    def rank(self, query: str, candidates: List[Dict[str, Any]]) -> List[str]:
        """Return selected contexts as a list of strings."""
        raise NotImplementedError