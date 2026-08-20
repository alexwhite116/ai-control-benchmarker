"""
Pluggable similarity backends for the control benchmarker.

TfidfSimilarityEngine works offline out of the box and is what the
CLI uses by default.

EmbeddingSimilarityEngine is a stub showing how to swap in real
embeddings (Azure OpenAI, OpenAI, or a local sentence-transformers
model) once you have API access / a machine with unrestricted
internet. The rest of the codebase doesn't need to change — just
pass a different engine into ControlBenchmarker.
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityEngine(ABC):
    """Base interface: given two lists of strings, return a similarity matrix."""

    @abstractmethod
    def similarity_matrix(self, texts_a: list[str], texts_b: list[str]) -> np.ndarray:
        """Return an (len(texts_a) x len(texts_b)) matrix of similarity scores in [0, 1]."""
        raise NotImplementedError


class TfidfSimilarityEngine(SimilarityEngine):
    """Baseline lexical similarity. No external calls, no API key, runs anywhere.

    Good enough to prove the pipeline end-to-end and to demo the concept.
    Weaker than real embeddings at catching paraphrased/synonymous controls
    (e.g. "human oversight" vs "ability for a person to intervene") — that's
    exactly the gap the embedding backend below is meant to close.
    """

    def similarity_matrix(self, texts_a: list[str], texts_b: list[str]) -> np.ndarray:
        vectorizer = TfidfVectorizer(stop_words="english")
        combined = texts_a + texts_b
        tfidf = vectorizer.fit_transform(combined)
        vecs_a = tfidf[: len(texts_a)]
        vecs_b = tfidf[len(texts_a):]
        return cosine_similarity(vecs_a, vecs_b)


class EmbeddingSimilarityEngine(SimilarityEngine):
    """Stub for a real embedding backend (Azure OpenAI / OpenAI / sentence-transformers).

    To activate:
      1. pip install openai  (or azure-ai-inference, or sentence-transformers)
      2. Fill in `embed()` below to call your provider of choice.
      3. Pass EmbeddingSimilarityEngine() into ControlBenchmarker instead of
         TfidfSimilarityEngine() — no other code changes required.

    This is deliberately left unimplemented in this starter kit since it
    needs an API key / model download that this environment can't reach —
    but the interface is what matters for the portfolio: it shows you
    designed for swappable embedding backends from day one.
    """

    def __init__(self, embed_fn=None):
        self.embed_fn = embed_fn  # callable: list[str] -> np.ndarray of shape (n, dim)

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.embed_fn is None:
            raise NotImplementedError(
                "Provide an embed_fn, e.g. an Azure OpenAI or OpenAI "
                "text-embedding call, or a sentence-transformers model.encode()."
            )
        return self.embed_fn(texts)

    def similarity_matrix(self, texts_a: list[str], texts_b: list[str]) -> np.ndarray:
        emb_a = self.embed(texts_a)
        emb_b = self.embed(texts_b)
        return cosine_similarity(emb_a, emb_b)
