"""
TF-IDF + Logistic Regression baseline: trained per-task on that task's
own training split, then evaluated on the same held-out sample used for
the Transformer comparison. This is the "lightweight, computationally
inexpensive reference point" the project spec calls for - we do not
assume it performs badly, we measure it.
"""
import time

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def train_baseline(train_texts, train_labels) -> Pipeline:
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(train_texts, train_labels)
    return pipeline


def evaluate_baseline(pipeline: Pipeline, test_texts):
    """Predict one example at a time so latency is measured comparably
    to the Transformer path (which also processes one message at a time
    in the live chat pipeline)."""
    predictions = []
    latencies_ms = []
    for text in test_texts:
        start = time.perf_counter()
        prediction = pipeline.predict([text])[0]
        latencies_ms.append((time.perf_counter() - start) * 1000)
        predictions.append(prediction)
    return predictions, latencies_ms
