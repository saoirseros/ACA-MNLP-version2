"""
Fine-tunes a DistilBERT base checkpoint for sentiment classification on
SST-2.

NOTE: the live service's sentiment model
(distilbert-base-uncased-finetuned-sst-2-english) is *already* fine-tuned
on this exact dataset by its original authors, so this script mainly
serves as a working, reproducible template for the fine-tuning pipeline
required by the project spec - it does not by itself change what the
live nlp-service uses. To actually use a checkpoint produced here, update
MODEL_CHECKPOINT in nlp-service/app/models/sentiment/model.py to point at
the saved output_dir this script prints.

Not executed as part of Phase 9 in this session (CPU-only Transformer
fine-tuning takes far longer than is practical for a single working
session) - see training/README.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # training/
from shared.finetune_utils import fine_tune_classifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))  # reuse the dataset loader
from datasets_loader import load_sentiment_split  # noqa: E402

BASE_CHECKPOINT = "distilbert-base-uncased"
LABELS = ["negative", "positive"]


def main():
    train_texts, train_label_names = load_sentiment_split("train", n=2000)
    eval_texts, eval_label_names = load_sentiment_split("validation", n=300)

    label_to_id = {name: i for i, name in enumerate(LABELS)}
    train_labels = [label_to_id[name] for name in train_label_names]
    eval_labels = [label_to_id[name] for name in eval_label_names]

    result = fine_tune_classifier(
        task_name="sentiment",
        base_checkpoint=BASE_CHECKPOINT,
        label_names=LABELS,
        train_texts=train_texts,
        train_labels=train_labels,
        eval_texts=eval_texts,
        eval_labels=eval_labels,
    )
    print(result)


if __name__ == "__main__":
    main()
