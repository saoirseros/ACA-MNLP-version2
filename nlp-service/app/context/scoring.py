"""
Adaptive Context Activation: interpretable context-requirement scoring.

Combines a handful of simple, explainable signals into a single "context
requirement score" in [0, 1], then buckets it into low/medium/high. This
is a deliberately simple first implementation per the project spec ("do
not make this mathematically unnecessarily complicated"); the weights and
thresholds below are heuristic starting points, not tuned from data -
that tuning is exactly what the Phase 9 experiment framework is for.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

from app.context.embeddings import cosine_similarities, embed
from app.context.signals import brevity_signal, reference_signal
from app.models.baseline.sentiment_baseline import analyze_sentiment_baseline
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Weight of each signal in the final score. Must sum to 1.0.
WEIGHTS = {
    "reference": 0.35,     # conversational dependency: pronouns/back-references
    "brevity": 0.15,        # ambiguity: very short messages carry less standalone meaning
    "similarity": 0.30,     # semantic relevance / topic continuity with recent turns
    "uncertainty": 0.20,    # model uncertainty: how confident the sentiment model is alone
}

LOW_THRESHOLD = 0.35
HIGH_THRESHOLD = 0.65

# The lightweight TF-IDF + Logistic Regression baseline (see
# app/models/baseline/) is far less calibrated than the heavyweight
# Transformer it replaced for this signal: even on unambiguous text it
# rarely reports a max-class probability above ~0.70 (versus ~0.99+ for
# the Transformer). Empirically, confidence at or above this ceiling
# already reflects a clear-cut prediction *for this specific model*, so
# it is normalized to "fully certain" (uncertainty 0.0) beyond that point,
# while confidence near 0.5 (an actual coin flip) still registers full
# uncertainty (1.0). Without this, the uncertainty signal would be
# persistently high for nearly every message, since this baseline almost
# never approaches 1.0 confidence, which would over-inflate the context
# score across the board.
BASELINE_CONFIDENCE_CEILING = 0.60

# How many of the most recent history messages to consider at all when
# measuring similarity/topic continuity (older messages are ignored here,
# not because they can't matter, but to keep this first implementation
# cheap and interpretable).
MAX_HISTORY_CONSIDERED = 8


@dataclass
class ContextScore:
    score: float
    level: str  # "low" | "medium" | "high"
    signals: dict = field(default_factory=dict)
    # Per-message similarity to the current text, aligned with the
    # trailing slice of history actually considered (see considered_history).
    similarities: List[float] = field(default_factory=list)
    considered_history: List[str] = field(default_factory=list)
    # The lightweight baseline's sentiment prediction on `text` alone,
    # computed as a side effect of the uncertainty signal below. Exposed
    # here so the model-tier cascade in api/analysis.py can reuse it
    # directly for "low" context messages instead of recomputing it.
    baseline_sentiment: dict = field(default_factory=dict)


def _uncertainty_signal(text: str) -> Tuple[float, dict]:
    """
    How uncertain the lightweight baseline model is about this message on
    its own. Confidence near 0.5 -> high uncertainty (1.0); confidence at
    or above BASELINE_CONFIDENCE_CEILING -> low uncertainty (0.0).

    This deliberately calls the *lightweight* baseline (not the
    heavyweight Transformer) so that scoring a message "low" and routing
    it to the cheap tier never first pays the cost of the expensive model
    it was trying to avoid - the whole point of the model-tier cascade.
    """
    try:
        result = analyze_sentiment_baseline(text)
        confidence = result["sentiment"]["confidence"]
    except Exception:  # noqa: BLE001 - fall back to a neutral signal, never crash scoring
        logger.exception("Baseline sentiment model unavailable during context scoring; using neutral uncertainty")
        return 0.5, {}
    spread = BASELINE_CONFIDENCE_CEILING - 0.5
    uncertainty = max(0.0, 1.0 - abs(confidence - 0.5) / spread) if spread > 0 else 0.0
    return uncertainty, result


def _similarity_signal(text: str, history: List[str]) -> Tuple[float, List[float]]:
    """
    Highest cosine similarity between the current message and any of the
    recent history messages, plus the full per-message similarity list
    (reused later for context selection so we don't re-embed).
    """
    if not history:
        return 0.0, []

    embeddings = embed(history + [text])
    history_embeddings, query_embedding = embeddings[:-1], embeddings[-1]
    sims = cosine_similarities(query_embedding, history_embeddings)
    similarities = [max(0.0, float(s)) for s in sims]
    max_similarity = max(similarities) if similarities else 0.0
    return max_similarity, similarities


def score_context_requirement(text: str, history: List[str]) -> ContextScore:
    """
    Estimate how much conversational context `text` needs, given the
    preceding `history` messages (oldest first, current message excluded).
    """
    considered_history = history[-MAX_HISTORY_CONSIDERED:] if history else []

    reference = reference_signal(text)
    brevity = brevity_signal(text)
    similarity, similarities = _similarity_signal(text, considered_history)
    uncertainty, baseline_sentiment = _uncertainty_signal(text)

    signals = {
        "reference": reference,
        "brevity": brevity,
        "similarity": similarity,
        "uncertainty": uncertainty,
    }

    score = sum(WEIGHTS[name] * value for name, value in signals.items())
    score = round(min(1.0, max(0.0, score)), 4)

    if score < LOW_THRESHOLD:
        level = "low"
    elif score < HIGH_THRESHOLD:
        level = "medium"
    else:
        level = "high"

    return ContextScore(
        score=score,
        level=level,
        signals=signals,
        similarities=similarities,
        considered_history=considered_history,
        baseline_sentiment=baseline_sentiment,
    )
