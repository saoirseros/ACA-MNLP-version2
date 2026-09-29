"""
Lightweight emotion baseline: TF-IDF + Logistic Regression, trained on
the same dair-ai/emotion sample used in experiments/run_all.py (3000
training examples, seed 42).

Note the same label-space caveat already documented in
experiments/datasets_loader.py: dair-ai/emotion only covers
sadness/joy/anger/fear/surprise (no "love", no "neutral"/"disgust"),
whereas the heavyweight transformer (j-hartmann/emotion-english-
distilroberta-base) covers all 7 labels including neutral/disgust. This
is an existing, already-published discrepancy in the paper's evaluation,
not a new problem introduced here - it is simply now also visible on the
"lightweight" tier of the live cascade, and is called out in the trace
returned to the client so it's never silently hidden.
"""
import time
from functools import lru_cache
from typing import Tuple, List

from app.models.baseline._train import SEED, get_or_train_pipeline
from app.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_NAME = "tfidf-logreg-emotion"
N_TRAIN = 3000  # matches experiments/run_all.py SAMPLE_SIZES["emotion_train"]

# dair-ai/emotion's native label set used for training (see
# experiments/datasets_loader.py EMOTION_LABELS for why "love" is excluded).
LABEL_SPACE = ["sadness", "joy", "anger", "fear", "surprise"]


def _load_training_data() -> Tuple[List[str], List[str]]:
    """
    Identical loading logic to experiments/datasets_loader.py's
    load_emotion_split("train", N_TRAIN) - duplicated (not imported), see
    sentiment_baseline.py's _load_training_data for why.
    """
    from datasets import load_dataset

    ds = load_dataset("dair-ai/emotion", split="train")
    label_names = ds.features["label"].names  # ['sadness','joy','love','anger','fear','surprise']
    ds = ds.filter(lambda example: label_names[example["label"]] != "love")
    ds = ds.shuffle(seed=SEED).select(range(min(N_TRAIN, len(ds))))
    texts = list(ds["text"])
    labels = [label_names[label] for label in ds["label"]]
    return texts, labels


@lru_cache
def _get_pipeline():
    return get_or_train_pipeline(MODEL_NAME, _load_training_data)


def analyze_emotion_baseline(text: str) -> dict:
    """Run the lightweight emotion baseline on a single message."""
    pipeline = _get_pipeline()

    start = time.perf_counter()
    label = str(pipeline.predict([text])[0])
    proba = pipeline.predict_proba([text])[0]
    confidence = float(max(proba))
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "emotion": {"label": label, "confidence": round(confidence, 4)},
        "processing": {"latencyMs": round(latency_ms, 2), "model": MODEL_NAME},
    }
