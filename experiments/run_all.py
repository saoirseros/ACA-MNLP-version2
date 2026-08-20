"""
Phase 9 experiment runner - reproducibly generates every table/chart
referenced in experiments/FINDINGS.md from real model predictions on
real, public datasets.

Run from inside experiments/, using the nlp-service virtual environment
(it already has torch/transformers/scikit-learn/sentence-transformers,
plus this adds `datasets`/`matplotlib`):

    C:\\nlp-venvs\\ptp-nlp-service\\Scripts\\python.exe run_all.py

Sample sizes are kept modest (see SAMPLE_SIZES below) so this completes
in a reasonable time on a CPU-only development machine, while still
being large enough to give directionally meaningful results. Nothing in
the output is hardcoded - change SAMPLE_SIZES/SEED and re-run, and the
numbers will change accordingly (project spec: "Do not fake results").
"""
import time

from common import pathutil

pathutil.ensure_nlp_service_on_path()

from common.metrics import (  # noqa: E402
    compute_classification_metrics,
    latency_stats,
    plot_confusion_matrix,
    plot_grouped_bar,
    save_json,
)
from baselines import evaluate_baseline, train_baseline  # noqa: E402
from context_experiment import run_context_experiment  # noqa: E402
from datasets_loader import (  # noqa: E402
    load_dialogue_sample,
    load_emotion_split,
    load_sentiment_split,
    load_toxicity_split,
)
from transformer_eval import evaluate_emotion, evaluate_sentiment, evaluate_toxicity  # noqa: E402

SENTIMENT_LABELS = ["negative", "positive"]
TOXICITY_LABELS = ["non-toxic", "toxic"]
# Full live-model label space (see datasets_loader.py for why dair-ai's
# "love" label is excluded from the ground truth, and why "neutral" is
# still included - the transformer can predict it even though this
# particular dataset's ground truth doesn't contain it).
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

SAMPLE_SIZES = {
    "sentiment_train": 2000,
    "sentiment_test": 300,
    "emotion_train": 3000,
    "emotion_test": 300,
    "toxicity_train": 3000,
    "toxicity_test": 300,
    "context_dialogues": 40,
}


def run_task_experiment(task_name, labels, train_texts, train_labels, test_texts, test_labels, transformer_eval_fn):
    print(f"\n=== {task_name.upper()}: training TF-IDF + Logistic Regression baseline ===")
    baseline_pipeline = train_baseline(train_texts, train_labels)
    baseline_preds, baseline_latencies = evaluate_baseline(baseline_pipeline, test_texts)
    baseline_metrics = compute_classification_metrics(test_labels, baseline_preds, labels)
    baseline_latency_stats = latency_stats(baseline_latencies)

    print(f"=== {task_name.upper()}: evaluating pretrained Transformer ===")
    transformer_preds, transformer_latencies = transformer_eval_fn(test_texts)
    transformer_metrics = compute_classification_metrics(test_labels, transformer_preds, labels)
    transformer_latency_stats = latency_stats(transformer_latencies)

    result = {
        "task": task_name,
        "n_train": len(train_texts),
        "n_test": len(test_texts),
        "labels": labels,
        "baseline": {"metrics": baseline_metrics, "latency_ms": baseline_latency_stats},
        "transformer": {"metrics": transformer_metrics, "latency_ms": transformer_latency_stats},
    }

    save_json(f"{task_name}_comparison", result)

    plot_confusion_matrix(
        baseline_metrics["confusion_matrix"], labels,
        f"{task_name.title()} - TF-IDF+LogReg Baseline (n={len(test_texts)})",
        f"{task_name}_baseline_confusion_matrix.png",
    )
    plot_confusion_matrix(
        transformer_metrics["confusion_matrix"], labels,
        f"{task_name.title()} - Pretrained Transformer (n={len(test_texts)})",
        f"{task_name}_transformer_confusion_matrix.png",
    )
    plot_grouped_bar(
        ["Accuracy", "F1 (macro)", "F1 (weighted)"],
        {
            "Baseline (TF-IDF+LR)": [baseline_metrics["accuracy"], baseline_metrics["f1_macro"], baseline_metrics["f1_weighted"]],
            "Transformer": [transformer_metrics["accuracy"], transformer_metrics["f1_macro"], transformer_metrics["f1_weighted"]],
        },
        "Score", f"{task_name.title()}: Baseline vs Transformer", f"{task_name}_metric_comparison.png",
    )
    plot_grouped_bar(
        ["Mean latency (ms)"],
        {
            "Baseline (TF-IDF+LR)": [baseline_latency_stats["mean"] or 0],
            "Transformer": [transformer_latency_stats["mean"] or 0],
        },
        "Latency (ms)", f"{task_name.title()}: Inference Latency", f"{task_name}_latency_comparison.png",
    )

    print(
        f"{task_name}: baseline acc={baseline_metrics['accuracy']:.3f} f1_macro={baseline_metrics['f1_macro']:.3f} | "
        f"transformer acc={transformer_metrics['accuracy']:.3f} f1_macro={transformer_metrics['f1_macro']:.3f}"
    )

    return result


def run_sentiment_experiment():
    train_texts, train_labels = load_sentiment_split("train", SAMPLE_SIZES["sentiment_train"])
    test_texts, test_labels = load_sentiment_split("validation", SAMPLE_SIZES["sentiment_test"])
    return run_task_experiment(
        "sentiment", SENTIMENT_LABELS, train_texts, train_labels, test_texts, test_labels, evaluate_sentiment
    )


def run_emotion_experiment():
    train_texts, train_labels = load_emotion_split("train", SAMPLE_SIZES["emotion_train"])
    test_texts, test_labels = load_emotion_split("test", SAMPLE_SIZES["emotion_test"])
    return run_task_experiment(
        "emotion", EMOTION_LABELS, train_texts, train_labels, test_texts, test_labels, evaluate_emotion
    )


def run_toxicity_experiment():
    train_texts, train_labels = load_toxicity_split("train", SAMPLE_SIZES["toxicity_train"])
    test_texts, test_labels = load_toxicity_split("test", SAMPLE_SIZES["toxicity_test"])
    return run_task_experiment(
        "toxicity", TOXICITY_LABELS, train_texts, train_labels, test_texts, test_labels, evaluate_toxicity
    )


def run_adaptive_context_experiment():
    print("\n=== ADAPTIVE CONTEXT: Full-context vs Adaptive Context Activation (DailyDialog) ===")
    dialogues = load_dialogue_sample(SAMPLE_SIZES["context_dialogues"])
    outcome = run_context_experiment(dialogues)

    full_metrics = compute_classification_metrics(outcome["full"]["y_true"], outcome["full"]["y_pred"], EMOTION_LABELS)
    adaptive_metrics = compute_classification_metrics(
        outcome["adaptive"]["y_true"], outcome["adaptive"]["y_pred"], EMOTION_LABELS
    )
    full_latency = latency_stats(outcome["full"]["latencies_ms"])
    adaptive_latency = latency_stats(outcome["adaptive"]["latencies_ms"])

    def avg(values):
        return sum(values) / len(values) if values else None

    result = {
        "n_dialogues": SAMPLE_SIZES["context_dialogues"],
        "n_turns_evaluated": len(outcome["full"]["y_true"]),
        "labels": EMOTION_LABELS,
        "full_context": {
            "metrics": full_metrics,
            "latency_ms": full_latency,
            "average_context_size_words": avg(outcome["full"]["context_sizes"]),
        },
        "adaptive_context": {
            "metrics": adaptive_metrics,
            "latency_ms": adaptive_latency,
            "average_context_size_words": avg(outcome["adaptive"]["context_sizes"]),
        },
    }

    save_json("context_experiment_comparison", result)

    plot_confusion_matrix(
        full_metrics["confusion_matrix"], EMOTION_LABELS, "Full-Context Strategy (emotion)", "context_full_confusion_matrix.png"
    )
    plot_confusion_matrix(
        adaptive_metrics["confusion_matrix"], EMOTION_LABELS, "Adaptive-Context Strategy (emotion)",
        "context_adaptive_confusion_matrix.png",
    )
    plot_grouped_bar(
        ["Accuracy", "F1 (macro)", "F1 (weighted)"],
        {
            "Full context": [full_metrics["accuracy"], full_metrics["f1_macro"], full_metrics["f1_weighted"]],
            "Adaptive context": [adaptive_metrics["accuracy"], adaptive_metrics["f1_macro"], adaptive_metrics["f1_weighted"]],
        },
        "Score", "Full-Context vs Adaptive-Context: Predictive Quality", "context_metric_comparison.png",
    )
    plot_grouped_bar(
        ["Avg context size (words)", "Avg latency (ms)"],
        {
            "Full context": [result["full_context"]["average_context_size_words"] or 0, full_latency["mean"] or 0],
            "Adaptive context": [result["adaptive_context"]["average_context_size_words"] or 0, adaptive_latency["mean"] or 0],
        },
        "Value", "Full-Context vs Adaptive-Context: Efficiency", "context_efficiency_comparison.png",
    )

    print(
        f"Full context:     acc={full_metrics['accuracy']:.3f} f1_macro={full_metrics['f1_macro']:.3f} "
        f"avg_context_words={result['full_context']['average_context_size_words']:.1f} avg_latency={full_latency['mean']:.1f}ms"
    )
    print(
        f"Adaptive context: acc={adaptive_metrics['accuracy']:.3f} f1_macro={adaptive_metrics['f1_macro']:.3f} "
        f"avg_context_words={result['adaptive_context']['average_context_size_words']:.1f} avg_latency={adaptive_latency['mean']:.1f}ms"
    )

    return result


def main():
    overall_start = time.perf_counter()

    sentiment_result = run_sentiment_experiment()
    emotion_result = run_emotion_experiment()
    toxicity_result = run_toxicity_experiment()
    context_result = run_adaptive_context_experiment()

    total_time = time.perf_counter() - overall_start
    print(f"\nAll experiments complete in {total_time:.1f}s. Results saved to experiments/results/.")

    return {
        "sentiment": sentiment_result,
        "emotion": emotion_result,
        "toxicity": toxicity_result,
        "context": context_result,
    }


if __name__ == "__main__":
    main()
