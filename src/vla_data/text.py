"""Deterministic, dependency-free instruction encoder.

Hashed bag of word unigrams + bigrams (signed feature hashing, L2-normalized).
It needs no model download, runs identically on a phone, and is compositional:
an unseen instruction that reuses known words ("pick up the ketchup and place
it in the basket" vs "... the milk ...") lands near related training tasks.
It is deliberately weak compared with a pretrained language model (no
synonyms, no word order beyond bigrams) — the benchmark reports it as such.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, List

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = {"the", "a", "an", "it", "of", "to", "and"}


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def _h(s: str) -> int:
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "little")


def encode(text: str, dim: int = 64) -> np.ndarray:
    toks = tokenize(text)
    grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    v = np.zeros(dim, dtype=np.float32)
    for g in grams:
        h = _h(g)
        v[h % dim] += 1.0 if (h >> 32) & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def encode_many(texts: Iterable[str], dim: int = 64) -> np.ndarray:
    return np.stack([encode(t, dim) for t in texts])
