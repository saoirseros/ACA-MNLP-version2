"""
Lazy-loaded lightweight sentence embedding model.

Used by Adaptive Context Activation to measure semantic similarity between
the current message and recent conversation history. Uses
sentence-transformers/all-MiniLM-L6-v2: a small (~90MB), CPU-friendly
embedding model - not a generative Transformer, so it's cheap to run
per message compared to the classification/summarization models.
"""
from functools import lru_cache

import numpy as np

from app.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_CHECKPOINT = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache
def _get_model():
    # Imported lazily so importing this module never pulls in
    # sentence-transformers/torch until an embedding is actually needed.
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model (first use): %s", MODEL_CHECKPOINT)
    return SentenceTransformer(MODEL_CHECKPOINT)


def embed(texts: list[str]) -> np.ndarray:
    """Return an (n, dim) array of L2-normalized embeddings for `texts`."""
    if not texts:
        return np.empty((0, 0))
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def cosine_similarities(query_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    """
    Cosine similarity of one query embedding against many candidates.
    Assumes both are already L2-normalized (see embed()), so this is a
    plain dot product.
    """
    if candidate_embeddings.size == 0:
        return np.array([])
    return candidate_embeddings @ query_embedding
