"""
Fine-tunes a DistilBERT base checkpoint for emotion classification on
dair-ai/emotion. Unlike sentiment, this IS a meaningful in-domain
fine-tune to try: the live service's emotion model
(j-hartmann/emotion-english-distilroberta-base) was fine-tuned on a
different emotion dataset, so it is evaluated out-of-domain on
dair-ai/emotion in experiments/FINDINGS.md - this script is the natural
next experiment referenced there.

To actually use a checkpoint produced here, update MODEL_CHECKPOINT in
nlp-service/app/models/emotion/model.py to point at the saved output_dir
this script prints (and update its label set if different).

Not executed as part of Phase 9 in this session (CPU-only Transformer
fine-tuning takes far longer than is practical for a single working
session) - see training/README.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # training/
from shared.finetune_utils import fine_tune_classifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))  # reuse the dataset loader
from datasets_loader import load_emotion_split  # noqa: E402

BASE_CHECKPOINT = "distilbert-base-uncased"
# dair-ai/emotion's 6 labels minus "love" (see experiments/datasets_loader.py for why).
LABELS = ["sadness", "joy", "anger", "fear", "surprise"]


def main():
    train_texts, train_label_names = load_emotion_split("train", n=3000)
    eval_texts, eval_label_names = load_emotion_split("test", n=300)

    label_to_id = {name: i for i, name in enumerate(LABELS)}
    train_labels = [label_to_id[name] for name in train_label_names]
    eval_labels = [label_to_id[name] for name in eval_label_names]

    result = fine_tune_classifier(
        task_name="emotion",
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
