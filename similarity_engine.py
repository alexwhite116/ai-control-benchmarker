"""
Pluggable similarity backends for the control benchmarker.

TfidfSimilarityEngine works offline out of the box and is what the
CLI uses by default.

EmbeddingSimilarityEngine swaps in real
embeddings (Azure OpenAI, OpenAI, or a local sentence-transformers
model). The code is designed to work flexibly with any embedding backend,
so you can swap in your own.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityEngine(ABC):
    """Base interface: given two lists of strings, return a similarity matrix."""

    @abstractmethod
    def similarity_matrix(self, texts_a: list[str], texts_b: list[str]) -> np.ndarray:
        """Return an (len(texts_a) x len(texts_b)) matrix of similarity scores"""
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
    """Similarity based on real embeddings, defaulting to a local sentence-transformers model.

    You can swap in your own embedding function (e.g. Azure OpenAI or OpenAI) by passing
    an embed_fn to the constructor. The function should take a list of strings and return a 2D numpy array of embeddings.
    """

    def __init__(self, embed_fn: Callable[[list[str]], np.ndarray] = None):
        from sentence_transformers import SentenceTransformer
        self.embed_fn = embed_fn or SentenceTransformer("all-MiniLM-L6-v2").encode  # accept custom model or default to all-MiniLM-L6-v2

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

class CrossEncoderSimilarityEngine(SimilarityEngine):
    """Similarity based on a cross-encoder model from sentence-transformers.

    This is more accurate than the EmbeddingSimilarityEngine, but slower because it requires
    pairwise scoring of all combinations of texts_a and texts_b.
    """

    def __init__(self, model_name: str = "cross-encoder/stsb-roberta-base"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def similarity_matrix(self, texts_a: list[str], texts_b: list[str]) -> np.ndarray:
        pairs = [(a, b) for a in texts_a for b in texts_b]
        scores = self.model.predict(pairs)
        if type(scores) is not np.ndarray:
            raise AttributeError("CrossEncoder.predict() should return a numpy array, but got type: {}".format(type(scores)))
        return scores.reshape(len(texts_a), len(texts_b))