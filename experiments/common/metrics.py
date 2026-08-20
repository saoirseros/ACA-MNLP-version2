"""
Shared evaluation metrics and chart generation for all Phase 9
experiments. Every number here is computed from real model predictions
against real dataset labels - nothing in this module accepts or produces
hardcoded results (project spec: "Do not fake results").
"""
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")  # headless - no display available/needed
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def compute_classification_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict:
    """
    Accuracy plus macro/weighted precision/recall/F1 (macro-F1 matters
    most for imbalanced label sets - see project spec: "Do not blindly
    report accuracy when it is misleading"), a full classification
    report, and a confusion matrix, all computed from the given
    predictions vs. ground-truth labels.
    """
    return {
        "n_samples": len(y_true),
        "labels": list(labels),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def latency_stats(latencies_ms: Sequence[float]) -> dict:
    """Mean/p50/p95/min/max over a list of per-example latencies (ms)."""
    if not latencies_ms:
        return {"mean": None, "p50": None, "p95": None, "min": None, "max": None, "n": 0}
    ordered = sorted(latencies_ms)
    n = len(ordered)
    return {
        "mean": sum(ordered) / n,
        "p50": ordered[n // 2],
        "p95": ordered[min(n - 1, int(n * 0.95))],
        "min": ordered[0],
        "max": ordered[-1],
        "n": n,
    }


def save_json(name: str, data: dict) -> Path:
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def plot_confusion_matrix(cm: List[List[int]], labels: Sequence[str], title: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    max_val = max((max(row) for row in cm), default=1) or 1
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            color = "white" if value > max_val / 2 else "black"
            ax.text(j, i, value, ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_grouped_bar(categories: Sequence[str], series: Dict[str, Sequence[float]], ylabel: str, title: str, filename: str) -> Path:
    """series: {"Baseline": [...], "Transformer": [...]}, each aligned with `categories`."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    n_series = max(len(series), 1)
    width = 0.8 / n_series
    x = list(range(len(categories)))
    for idx, (name, values) in enumerate(series.items()):
        offsets = [xi + idx * width for xi in x]
        ax.bar(offsets, values, width=width, label=name)
    ax.set_xticks([xi + width * (n_series - 1) / 2 for xi in x])
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
