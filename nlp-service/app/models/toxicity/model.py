"""
Lazy-loaded wrapper around a pretrained toxicity/offensive-language
classifier.

Uses unitary/toxic-bert: a BERT-base checkpoint fine-tuned on the Jigsaw
Toxic Comment Classification dataset. It is a multi-label model (a message
can simultaneously be "toxic" + "insult" + "obscene", etc, each with its
own independent probability) rather than a single mutually-exclusive
class, so we:
  1. use the general "toxic" label's probability to decide toxic vs
     non-toxic (thresholded at 0.5), and
  2. when flagged toxic, report the most likely specific category
     (insult, threat, obscene, ...) for extra context.
"""
import time
from functools import lru_cache
from typing import Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_CHECKPOINT = "unitary/toxic-bert"
TOXIC_THRESHOLD = 0.5


@lru_cache
def _get_pipeline():
    # Imported lazily so importing this module never pulls in torch/transformers
    # until a toxicity request actually arrives.
    from transformers import pipeline

    logger.info("Loading toxicity model (first use): %s", MODEL_CHECKPOINT)
    return pipeline("text-classification", model=MODEL_CHECKPOINT, top_k=None)


def analyze_toxicity(text: str, context: Optional[list[str]] = None) -> dict:
    """
    Run toxicity classification on `text`.

    `context` is accepted now for interface stability (Adaptive Context
    Selection lands in Phase 6) but this simple single-message classifier
    does not yet use it.
    """
    classifier = _get_pipeline()

    start = time.perf_counter()
    results = classifier(text, truncation=True)[0]
    latency_ms = (time.perf_counter() - start) * 1000

    scores = {item["label"].lower(): float(item["score"]) for item in results}
    toxic_score = scores.get("toxic", 0.0)
    is_toxic = toxic_score >= TOXIC_THRESHOLD

    category = None
    if is_toxic:
        specific_scores = {label: score for label, score in scores.items() if label != "toxic"}
        if specific_scores:
            category = max(specific_scores, key=specific_scores.get)

    confidence = toxic_score if is_toxic else (1 - toxic_score)

    return {
        "toxicity": {
            "label": "toxic" if is_toxic else "non-toxic",
            "confidence": round(confidence, 4),
            "category": category,
        },
        "processing": {"latencyMs": round(latency_ms, 2), "model": MODEL_CHECKPOINT},
    }
