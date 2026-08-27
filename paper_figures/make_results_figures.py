"""
Generates the "IV. Results" figures for the research paper, reading
real numbers directly from experiments/results/*.json (produced by
experiments/run_all.py) rather than hardcoding any value, so the charts
can never drift out of sync with the actual experiment output.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"
OUT_DIR = Path(__file__).resolve().parent

TASKS = ["sentiment", "emotion", "toxicity"]
TASK_LABELS = ["Sentiment", "Emotion", "Toxicity"]


def load(name):
    with open(RESULTS_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def grouped_bar(categories, series, ylabel, title, filename, value_fmt="{:.3f}", log=False):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    n_series = len(series)
    width = 0.8 / n_series
    x = np.arange(len(categories))
    for i, (name, values) in enumerate(series.items()):
        offsets = x + i * width
        bars = ax.bar(offsets, values, width=width, label=name)
        for b, v in zip(bars, values):
            ax.annotate(value_fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8.5, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x + width * (n_series - 1) / 2)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
    else:
        top = max(max(v) for v in series.values())
        ax.set_ylim(0, top * 1.22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved", filename)


def main():
    task_data = {t: load(f"{t}_comparison") for t in TASKS}
    ctx = load("context_experiment_comparison")

    # --- Figure 3: Accuracy, Baseline vs Transformer, across all 3 tasks ---
    baseline_acc = [task_data[t]["baseline"]["metrics"]["accuracy"] for t in TASKS]
    transformer_acc = [task_data[t]["transformer"]["metrics"]["accuracy"] for t in TASKS]
    grouped_bar(
        TASK_LABELS,
        {"Baseline (TF-IDF + LR)": baseline_acc, "Transformer": transformer_acc},
        "Accuracy", "Fig. 3. Accuracy: Baseline vs. Transformer",
        "figure3_accuracy_comparison.png",
    )

    # --- Figure 4: Macro F1, Baseline vs Transformer, across all 3 tasks ---
    baseline_f1 = [task_data[t]["baseline"]["metrics"]["f1_macro"] for t in TASKS]
    transformer_f1 = [task_data[t]["transformer"]["metrics"]["f1_macro"] for t in TASKS]
    grouped_bar(
        TASK_LABELS,
        {"Baseline (TF-IDF + LR)": baseline_f1, "Transformer": transformer_f1},
        "F1-score (macro)", "Fig. 4. Macro F1-Score: Baseline vs. Transformer",
        "figure4_f1_comparison.png",
    )

    # --- Figure 5: Mean inference latency (log scale), Baseline vs Transformer ---
    baseline_lat = [task_data[t]["baseline"]["latency_ms"]["mean"] for t in TASKS]
    transformer_lat = [task_data[t]["transformer"]["latency_ms"]["mean"] for t in TASKS]
    grouped_bar(
        TASK_LABELS,
        {"Baseline (TF-IDF + LR)": baseline_lat, "Transformer": transformer_lat},
        "Mean latency (ms, log scale)", "Fig. 5. Mean Inference Latency: Baseline vs. Transformer",
        "figure5_latency_comparison.png", value_fmt="{:.1f}", log=True,
    )

    # --- Figure 7: Full-context vs Adaptive-context efficiency (dual panel) ---
    full_ctx_size = ctx["full_context"]["average_context_size_words"]
    adaptive_ctx_size = ctx["adaptive_context"]["average_context_size_words"]
    full_latency = ctx["full_context"]["latency_ms"]["mean"]
    adaptive_latency = ctx["adaptive_context"]["latency_ms"]["mean"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))
    labels = ["Full context", "Adaptive context\n(ACA)"]
    colors = ["#1f77b4", "#ff7f0e"]

    bars0 = axes[0].bar(labels, [full_ctx_size, adaptive_ctx_size], color=colors)
    axes[0].set_ylabel("Avg. context size (words)")
    axes[0].set_title("Context size sent to classifiers")
    for b, v in zip(bars0, [full_ctx_size, adaptive_ctx_size]):
        axes[0].annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                          ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")

    bars1 = axes[1].bar(labels, [full_latency, adaptive_latency], color=colors)
    axes[1].set_ylabel("Mean latency (ms)")
    axes[1].set_title("End-to-end analysis latency")
    for b, v in zip(bars1, [full_latency, adaptive_latency]):
        axes[1].annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                          ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")

    fig.suptitle("Fig. 7. Full-Context vs. Adaptive-Context: Context Size and Latency", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure7_context_efficiency.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved figure7_context_efficiency.png")

    # --- Figure 8: side-by-side confusion matrices, Full vs Adaptive context ---
    emotion_labels = ctx["labels"]
    cm_full = np.array(ctx["full_context"]["metrics"]["confusion_matrix"])
    cm_adaptive = np.array(ctx["adaptive_context"]["metrics"]["confusion_matrix"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, cm, title in zip(axes, [cm_full, cm_adaptive], ["Full-context strategy", "Adaptive-context (ACA)"]):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(emotion_labels)))
        ax.set_yticks(range(len(emotion_labels)))
        ax.set_xticklabels(emotion_labels, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticklabels(emotion_labels, fontsize=8.5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title, fontsize=10.5)
        max_val = cm.max() if cm.max() > 0 else 1
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                color = "white" if val > max_val / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Fig. 8. Confusion Matrices: Full-Context vs. Adaptive-Context (Emotion, DailyDialog)", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure8_context_confusion_matrices.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved figure8_context_confusion_matrices.png")


if __name__ == "__main__":
    main()
