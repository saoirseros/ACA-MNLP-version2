"""
Lightweight sentiment baseline: TF-IDF + Logistic Regression, trained on
the same SST-2 sample used in experiments/run_all.py (2000 training
examples, seed 42). This is the "cheap" tier of the model cascade - used
for messages that Adaptive Context Activation (ACA) judges to need no
conversational context at all (level == "low"), since a bag-of-n-grams
classifier is already sufficient for a message that stands on its own,
at a small fraction of the compute cost of the DistilBERT transformer.
"""
import time
from functools import lru_cache
from typing import Tuple, List

from app.models.baseline._train import SEED, get_or_train_pipeline
from app.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_NAME = "tfidf-logreg-sentiment"
N_TRAIN = 2000  # matches experiments/run_all.py SAMPLE_SIZES["sentiment_train"]


def _load_training_data() -> Tuple[List[str], List[str]]:
    """
    Identical loading logic to experiments/datasets_loader.py's
    load_sentiment_split("train", N_TRAIN) - duplicated (not imported)
    because nlp-service is a standalone deployable service and
    experiments/ is a sibling research/evaluation directory that isn't
    guaranteed to be on the Python path (or even present) at runtime.
    """
    from datasets import load_dataset

    ds = load_dataset("glue", "sst2", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(N_TRAIN, len(ds))))
    label_map = {0: "negative", 1: "positive"}
    texts = list(ds["sentence"])
    labels = [label_map[label] for label in ds["label"]]
    return texts, labels


@lru_cache
def _get_pipeline():
    return get_or_train_pipeline(MODEL_NAME, _load_training_data)


def analyze_sentiment_baseline(text: str) -> dict:
    """Run the lightweight sentiment baseline on a single message."""
    pipeline = _get_pipeline()

    start = time.perf_counter()
    # str() normalizes sklearn/numpy's numpy.str_ scalar to a plain
    # Python str so it serializes cleanly through Pydantic/JSON.
    label = str(pipeline.predict([text])[0])
    # LogisticRegression gives calibrated-ish class probabilities via
    # predict_proba; take the probability of the predicted class as the
    # confidence, mirroring the {label, confidence} shape of the heavy model.
    proba = pipeline.predict_proba([text])[0]
    confidence = float(max(proba))
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "sentiment": {"label": label, "confidence": round(confidence, 4)},
        "processing": {"latencyMs": round(latency_ms, 2), "model": MODEL_NAME},
    }
