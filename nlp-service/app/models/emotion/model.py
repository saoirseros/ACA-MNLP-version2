"""
Lazy-loaded wrapper around a pretrained emotion classifier.

Uses j-hartmann/emotion-english-distilroberta-base: a DistilRoBERTa
checkpoint fine-tuned on an English emotion dataset, producing one of
7 labels (anger, disgust, fear, joy, neutral, sadness, surprise) with a
confidence score. Small and CPU-friendly, matching the project's resource
constraints.
"""
import time
from functools import lru_cache
from typing import Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_CHECKPOINT = "j-hartmann/emotion-english-distilroberta-base"


@lru_cache
def _get_pipeline():
    # Imported lazily so importing this module never pulls in torch/transformers
    # until an emotion request actually arrives.
    from transformers import pipeline

    logger.info("Loading emotion model (first use): %s", MODEL_CHECKPOINT)
    return pipeline("text-classification", model=MODEL_CHECKPOINT, top_k=None)


def analyze_emotion(text: str, context: Optional[list[str]] = None) -> dict:
    """
    Run emotion classification on `text`.

    `context` is accepted now for interface stability (Adaptive Context
    Selection lands in Phase 6) but this simple single-message classifier
    does not yet use it.
    """
    classifier = _get_pipeline()

    start = time.perf_counter()
    scores = classifier(text, truncation=True)[0]  # list of {label, score} across all 7 labels
    latency_ms = (time.perf_counter() - start) * 1000

    top = max(scores, key=lambda item: item["score"])

    return {
        "emotion": {"label": top["label"].lower(), "confidence": round(float(top["score"]), 4)},
        "processing": {"latencyMs": round(latency_ms, 2), "model": MODEL_CHECKPOINT},
    }
