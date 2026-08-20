"""
Evaluates each task's pretrained Transformer - the exact model wrapper
used by the live nlp-service, imported directly rather than duplicated -
on the same held-out sample used for the baseline, so results are
directly comparable.

Note: these models are used as pretrained (not fine-tuned on the
specific evaluation datasets below). See experiments/README.md and
experiments/FINDINGS.md for what that does and doesn't mean for
interpreting the comparison against the baseline (which IS trained on
each dataset's own training split).
"""
from common.pathutil import ensure_nlp_service_on_path

ensure_nlp_service_on_path()

from app.models.emotion.model import analyze_emotion  # noqa: E402
from app.models.sentiment.model import analyze_sentiment  # noqa: E402
from app.models.toxicity.model import analyze_toxicity  # noqa: E402


def evaluate_sentiment(texts):
    predictions, latencies_ms = [], []
    for text in texts:
        result = analyze_sentiment(text)
        predictions.append(result["sentiment"]["label"])
        latencies_ms.append(result["processing"]["latencyMs"])
    return predictions, latencies_ms


def evaluate_emotion(texts):
    predictions, latencies_ms = [], []
    for text in texts:
        result = analyze_emotion(text)
        predictions.append(result["emotion"]["label"])
        latencies_ms.append(result["processing"]["latencyMs"])
    return predictions, latencies_ms


def evaluate_toxicity(texts):
    predictions, latencies_ms = [], []
    for text in texts:
        result = analyze_toxicity(text)
        predictions.append(result["toxicity"]["label"])
        latencies_ms.append(result["processing"]["latencyMs"])
    return predictions, latencies_ms
