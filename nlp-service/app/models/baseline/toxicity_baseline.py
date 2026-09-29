"""
Lightweight toxicity baseline: TF-IDF + Logistic Regression, trained on
the same tweet_eval (offensive) sample used in experiments/run_all.py
(3000 training examples, seed 42).

Unlike the heavyweight unitary/toxic-bert model, this baseline is a
single binary classifier (toxic vs non-toxic) rather than a multi-label
one, so it never reports a specific category (insult/threat/obscene...)
- `category` is always None here, which mirrors how the heavy model
itself only reports a category when it flags a message as toxic.
"""
import time
from functools import lru_cache
from typing import Tuple, List

from app.models.baseline._train import SEED, get_or_train_pipeline
from app.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_NAME = "tfidf-logreg-toxicity"
N_TRAIN = 3000  # matches experiments/run_all.py SAMPLE_SIZES["toxicity_train"]


def _load_training_data() -> Tuple[List[str], List[str]]:
    """
    Identical loading logic to experiments/datasets_loader.py's
    load_toxicity_split("train", N_TRAIN) - duplicated (not imported), see
    sentiment_baseline.py's _load_training_data for why.
    """
    from datasets import load_dataset

    ds = load_dataset("tweet_eval", "offensive", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(N_TRAIN, len(ds))))
    label_map = {0: "non-toxic", 1: "toxic"}
    texts = list(ds["text"])
    labels = [label_map[label] for label in ds["label"]]
    return texts, labels


@lru_cache
def _get_pipeline():
    return get_or_train_pipeline(MODEL_NAME, _load_training_data)


def analyze_toxicity_baseline(text: str) -> dict:
    """Run the lightweight toxicity baseline on a single message."""
    pipeline = _get_pipeline()

    start = time.perf_counter()
    label = str(pipeline.predict([text])[0])
    proba = pipeline.predict_proba([text])[0]
    confidence = float(max(proba))
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "toxicity": {"label": label, "confidence": round(confidence, 4), "category": None},
        "processing": {"latencyMs": round(latency_ms, 2), "model": MODEL_NAME},
    }
