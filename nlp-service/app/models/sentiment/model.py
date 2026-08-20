"""
Lazy-loaded wrapper around a pretrained sentiment classifier.

Uses distilbert-base-uncased-finetuned-sst-2-english: a small (~260MB)
DistilBERT checkpoint fine-tuned on SST-2, producing POSITIVE/NEGATIVE
labels with a confidence score. This is the lightweight Transformer
baseline described in the project spec - CPU-friendly and appropriate
for an 8GB RAM development machine.

The model is only loaded into memory the first time analyze_sentiment()
is called (lazy loading), and then reused for all subsequent requests
(cached via lru_cache) rather than reloading it per request.
"""
import time
from functools import lru_cache
from typing import Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"


@lru_cache
def _get_pipeline():
    # Imported lazily so importing this module never pulls in torch/transformers
    # until a sentiment request actually arrives.
    from transformers import pipeline

    logger.info("Loading sentiment model (first use): %s", MODEL_CHECKPOINT)
    return pipeline("sentiment-analysis", model=MODEL_CHECKPOINT)


def analyze_sentiment(text: str, context: Optional[list[str]] = None) -> dict:
    """
    Run sentiment analysis on `text`.

    `context` is accepted now for interface stability (Adaptive Context
    Selection lands in Phase 6) but this simple single-message classifier
    does not yet use it.
    """
    classifier = _get_pipeline()

    start = time.perf_counter()
    result = classifier(text, truncation=True)[0]
    latency_ms = (time.perf_counter() - start) * 1000

    label = "positive" if result["label"] == "POSITIVE" else "negative"

    return {
        "sentiment": {"label": label, "confidence": round(float(result["score"]), 4)},
        "processing": {"latencyMs": round(latency_ms, 2), "model": MODEL_CHECKPOINT},
    }
