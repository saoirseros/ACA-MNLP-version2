"""
Lightweight, non-Transformer topic/keyword extraction.

Per the project spec ("a lightweight topic modeling approach or
embedding-based approach where practical"), this uses classic TF-IDF
(via scikit-learn) over the conversation's messages rather than a large
Transformer model - topic extraction from a handful of short chat
messages doesn't need one, and this keeps memory/CPU cost minimal.
"""
import time
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer

from app.utils.logging import get_logger

logger = get_logger(__name__)

TOP_N_KEYWORDS = 5


def extract_topics(messages: list[str], context: Optional[dict] = None) -> dict:
    """
    Extract a small set of representative keywords/phrases from a
    conversation's messages, treating each message as one "document" so
    TF-IDF can weigh terms that are distinctive to this conversation.
    """
    start = time.perf_counter()

    non_empty_messages = [m for m in messages if m and m.strip()]
    if not non_empty_messages:
        return {
            "topic": {"label": None, "keywords": []},
            "processing": {"latencyMs": 0.0, "model": "tfidf"},
        }

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=200)

    try:
        matrix = vectorizer.fit_transform(non_empty_messages)
    except ValueError:
        # Happens if every message is made up entirely of stop words/punctuation.
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "topic": {"label": None, "keywords": []},
            "processing": {"latencyMs": round(latency_ms, 2), "model": "tfidf"},
        }

    # Sum each term's TF-IDF weight across all messages to find the terms
    # most distinctive/representative of the conversation as a whole.
    scores = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda pair: pair[1], reverse=True)

    keywords = [term for term, score in ranked[:TOP_N_KEYWORDS] if score > 0]
    label = keywords[0] if keywords else None

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "topic": {"label": label, "keywords": keywords},
        "processing": {"latencyMs": round(latency_ms, 2), "model": "tfidf"},
    }
