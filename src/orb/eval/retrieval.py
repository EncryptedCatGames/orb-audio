"""Retrieval baseline: TF-IDF over captions, no model.

This is the bar the planner has to beat. It is deliberately simple.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class RetrievalIndex:
    def __init__(self, captions: list[str]):
        self.captions = captions
        docs = [_tokens(c) for c in captions]
        vocab = sorted({t for doc in docs for t in doc})
        self.vocab = {t: i for i, t in enumerate(vocab)}
        df = np.zeros(len(vocab), dtype=np.float64)
        for doc in docs:
            for token in set(doc):
                df[self.vocab[token]] += 1.0
        self.idf = np.log((1.0 + len(docs)) / (1.0 + df)) + 1.0
        self.matrix = np.zeros((len(docs), len(vocab)), dtype=np.float64)
        for i, doc in enumerate(docs):
            for token, count in Counter(doc).items():
                self.matrix[i, self.vocab[token]] = count
            self.matrix[i] *= self.idf
            norm = np.linalg.norm(self.matrix[i])
            if norm > 0:
                self.matrix[i] /= norm

    def query(self, text: str, k: int = 5) -> list[tuple[int, float]]:
        vec = np.zeros(len(self.vocab), dtype=np.float64)
        for token in _tokens(text):
            if token in self.vocab:
                vec[self.vocab[token]] += 1.0
        vec *= self.idf
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        scores = self.matrix @ vec
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]
