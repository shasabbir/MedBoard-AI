"""Deterministic local embeddings for reproducible offline demonstrations."""

from __future__ import annotations

import hashlib
import math
import re

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Map terms and bigrams into a fixed vector without model downloads."""

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 32:
            raise ValueError("embedding dimensions must be at least 32")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        tokens = TOKEN_PATTERN.findall(text.casefold())
        features = [*tokens, *(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))]
        vector = [0.0] * self.dimensions
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]
