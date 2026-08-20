"""
Fine-tunes a DistilBERT base checkpoint for toxicity/offensive-language
classification on tweet_eval (offensive config). Like emotion, this is a
meaningful in-domain fine-tune: the live service's toxicity model
(unitary/toxic-bert) was trained on Jigsaw (Wikipedia comments), so it is
evaluated out-of-domain on tweet_eval (Twitter) in
experiments/FINDINGS.md - this script is the natural next experiment
referenced there.

To actually use a checkpoint produced here, update MODEL_CHECKPOINT in
nlp-service/app/models/toxicity/model.py to point at the saved
output_dir this script prints (and simplify its multi-label handling,
since this produces a single binary toxic/non-toxic head rather than
Jigsaw's multi-label category set).

Not executed as part of Phase 9 in this session (CPU-only Transformer
fine-tuning takes far longer than is practical for a single working
session) - see training/README.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # training/
from shared.finetune_utils import fine_tune_classifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))  # reuse the dataset loader
from datasets_loader import load_toxicity_split  # noqa: E402

BASE_CHECKPOINT = "distilbert-base-uncased"
LABELS = ["non-toxic", "toxic"]


def main():
    train_texts, train_label_names = load_toxicity_split("train", n=3000)
    eval_texts, eval_label_names = load_toxicity_split("test", n=300)

    label_to_id = {name: i for i, name in enumerate(LABELS)}
    train_labels = [label_to_id[name] for name in train_label_names]
    eval_labels = [label_to_id[name] for name in eval_label_names]

    result = fine_tune_classifier(
        task_name="toxicity",
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
