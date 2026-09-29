"""
Optional model warm-up.

Since every chat message now runs through the combined /analyze/message
pipeline (Adaptive Context Activation + a lightweight/heavyweight model
cascade for sentiment + emotion + toxicity), it's worth paying the
one-time model-loading/training cost right after the service starts
rather than on the first real chat message. This runs in a background
thread so /health stays responsive immediately, even while warm-up is
still in progress.
"""
from app.context.embeddings import embed
from app.models.baseline.emotion_baseline import analyze_emotion_baseline
from app.models.baseline.sentiment_baseline import analyze_sentiment_baseline
from app.models.baseline.toxicity_baseline import analyze_toxicity_baseline
from app.models.emotion.model import analyze_emotion
from app.models.sentiment.model import analyze_sentiment
from app.models.toxicity.model import analyze_toxicity
from app.utils.logging import get_logger

logger = get_logger(__name__)

_WARM_UP_TEXT = "This is a warm-up request to preload the model into memory."


def warm_up_all_models() -> None:
    """Trigger lazy loading (and, for the lightweight baselines, training
    or cache loading) of every model used on the live per-message path."""
    logger.info("Warming up NLP models...")
    for name, warm_up_fn in (
        ("sentiment", lambda: analyze_sentiment(_WARM_UP_TEXT)),
        ("emotion", lambda: analyze_emotion(_WARM_UP_TEXT)),
        ("toxicity", lambda: analyze_toxicity(_WARM_UP_TEXT)),
        ("context-embeddings", lambda: embed([_WARM_UP_TEXT])),
        ("baseline-sentiment", lambda: analyze_sentiment_baseline(_WARM_UP_TEXT)),
        ("baseline-emotion", lambda: analyze_emotion_baseline(_WARM_UP_TEXT)),
        ("baseline-toxicity", lambda: analyze_toxicity_baseline(_WARM_UP_TEXT)),
    ):
        try:
            warm_up_fn()
        except Exception:  # noqa: BLE001 - warm-up failures shouldn't crash startup
            logger.exception("Failed to warm up %s model", name)
    logger.info("Model warm-up complete.")

