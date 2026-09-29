"""
Shared train-or-load helper for the lightweight baseline models used by
the Adaptive Context Activation (ACA) model-tier cascade.

Every baseline task (sentiment/emotion/toxicity) is a TF-IDF +
Logistic Regression pipeline trained with the *exact* recipe already
published and evaluated in experiments/baselines.py and
experiments/run_all.py (same vectorizer settings, same classifier, same
sample sizes, same random seed). Reusing that recipe here - rather than
inventing a new one - means the "lightweight model" this service now
routes low-context messages to is provably the same model whose
accuracy/latency was already measured and reported in the paper
(70.3% / 67.3% / 80.7% for sentiment/emotion/toxicity respectively),
not an unevaluated stand-in.

Training runs once per task (a few seconds on CPU, using the datasets
already cached locally by experiments/) and the fitted pipeline is then
cached to disk with joblib, so subsequent service restarts load it
instantly instead of retraining.
"""
from pathlib import Path
from typing import Callable, List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.utils.logging import get_logger

logger = get_logger(__name__)

# Cached alongside the service code but never committed to git (see
# .gitignore) - these are regenerable training artifacts, not source.
CACHE_DIR = Path(__file__).resolve().parent / "_artifacts"

SEED = 42  # matches experiments/datasets_loader.py's SEED, for reproducibility


def _build_pipeline() -> Pipeline:
    """Same architecture as experiments/baselines.py's train_baseline()."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def get_or_train_pipeline(name: str, loader: Callable[[], Tuple[List[str], List[str]]]) -> Pipeline:
    """
    Return a fitted TF-IDF + Logistic Regression pipeline for `name`,
    loading it from an on-disk cache if present, otherwise training it
    from scratch via `loader()` (which returns (train_texts, train_labels))
    and caching the result to disk.

    This is called once per task from an `lru_cache`-wrapped, zero-argument
    function in each `*_baseline.py` module, which is what keeps the
    fitted pipeline warm in memory for the process's lifetime - this
    function itself is not memoized, since the `loader` passed in is
    typically a fresh closure each call and would defeat an in-memory cache.
    """
    cache_path = CACHE_DIR / f"{name}.joblib"

    if cache_path.exists():
        try:
            logger.info("Loading cached lightweight baseline model: %s", cache_path)
            return joblib.load(cache_path)
        except Exception:  # noqa: BLE001 - corrupt/incompatible cache, retrain instead
            logger.exception("Failed to load cached baseline %s; retraining", name)

    logger.info("Training lightweight baseline model %s (first use)...", name)
    train_texts, train_labels = loader()
    pipeline = _build_pipeline()
    pipeline.fit(train_texts, train_labels)

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, cache_path)
    except Exception:  # noqa: BLE001 - caching is an optimization, not a hard requirement
        logger.exception("Failed to persist cached baseline %s", name)

    logger.info("Lightweight baseline model %s ready.", name)
    return pipeline
