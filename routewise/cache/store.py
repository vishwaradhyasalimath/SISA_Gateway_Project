"""
Prompt similarity cache.

Before every LLM call we check whether we've already answered something
close to this prompt. "Close" is defined by cosine similarity on TF-IDF
vectors — cheap, fast, no GPU needed.

The threshold is a deliberate research variable. Too strict and we get
almost no cache hits. Too loose and we start returning answers that don't
actually match the question. The default (0.85) sits in a comfortable
middle ground for the kinds of prompts we tested.

We keep everything in memory for simplicity. In a production setting
you'd swap this for Redis or a vector DB, but for a demo the dict is fine.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CacheResult:
    hit: bool
    response: Optional[str]
    matched_prompt: Optional[str]
    score: float
    threshold: float


@dataclass
class _Entry:
    prompt: str
    response: str
    model: str
    reason: str
    ts: float = field(default_factory=time.time)


class Cache:
    def __init__(
        self,
        threshold: float | None = None,
        on_event: Callable[[dict], None] | None = None,
    ):
        # Pull threshold from env if not explicitly passed.
        self.threshold = threshold or float(os.getenv("CACHE_THRESHOLD", "0.85"))
        self._entries: list[_Entry] = []
        self._notify  = on_event or (lambda _: None)

        self.lookups = 0
        self.hits    = 0
        self.misses  = 0

    def get(self, prompt: str) -> CacheResult:
        self.lookups += 1

        if not self._entries:
            self.misses += 1
            return CacheResult(hit=False, response=None,
                               matched_prompt=None, score=0.0,
                               threshold=self.threshold)

        entry, score = self._best_match(prompt)

        if score >= self.threshold:
            self.hits += 1
            self._notify({
                "event": "cache_hit",
                "query": prompt[:100],
                "matched": entry.prompt[:100],
                "score": round(score, 4),
            })
            return CacheResult(hit=True, response=entry.response,
                               matched_prompt=entry.prompt,
                               score=round(score, 4),
                               threshold=self.threshold)

        self.misses += 1
        self._notify({
            "event": "cache_miss",
            "query": prompt[:100],
            "best_score": round(score, 4),
            "threshold": self.threshold,
        })
        return CacheResult(hit=False, response=None, matched_prompt=None,
                           score=round(score, 4), threshold=self.threshold)

    def put(self, prompt: str, response: str, model: str, reason: str):
        self._entries.append(_Entry(
            prompt=prompt, response=response,
            model=model, reason=reason,
        ))

    def stats(self) -> dict:
        hr = round(self.hits / self.lookups * 100, 1) if self.lookups else 0.0
        return {
            "lookups":   self.lookups,
            "hits":      self.hits,
            "misses":    self.misses,
            "hit_rate":  hr,
            "size":      len(self._entries),
            "threshold": self.threshold,
        }

    def clear(self):
        self._entries.clear()
        self.lookups = self.hits = self.misses = 0

    def _best_match(self, query: str) -> tuple[_Entry, float]:
        prompts = [e.prompt for e in self._entries] + [query]
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            mat = vec.fit_transform(prompts)
        except ValueError:
            return self._entries[0], 0.0

        scores = cosine_similarity(mat[-1], mat[:-1]).flatten()
        idx    = int(np.argmax(scores))
        return self._entries[idx], float(scores[idx])
