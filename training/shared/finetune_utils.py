"""
Shared fine-tuning pipeline used by every training/<task>/train.py
script: dataset -> preprocessing/tokenizer -> train/validation split ->
pretrained Transformer -> fine-tuning -> evaluation -> saved model, per
the project spec's training workflow.

This is infrastructure the live nlp-service never calls directly at
startup (spec: "Do not retrain models every time the application
starts. The live application should load saved models."). A saved model
produced here would need a deliberate, separate change to
nlp-service/app/models/<task>/model.py's MODEL_CHECKPOINT to actually be
used by the live chat pipeline - fine-tuning and inference are
intentionally decoupled.
"""
from pathlib import Path
from typing import Sequence

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Fine-tuned models are saved here, never inside nlp-service/app/ itself,
# so they don't get confused with the live pretrained checkpoints.
MODELS_DIR = Path(__file__).resolve().parents[2] / "nlp-service" / "trained_models"


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
    }


def fine_tune_classifier(
    task_name: str,
    base_checkpoint: str,
    label_names: Sequence[str],
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    eval_texts: Sequence[str],
    eval_labels: Sequence[int],
    num_train_epochs: float = 2.0,
):
    """
    Fine-tunes `base_checkpoint` as a sequence classifier on the given
    train/eval splits and saves the result under
    nlp-service/trained_models/<task_name>/final/.
    """
    output_dir = MODELS_DIR / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_checkpoint,
        num_labels=len(label_names),
        id2label={i: name for i, name in enumerate(label_names)},
        label2id={name: i for i, name in enumerate(label_names)},
        ignore_mismatched_sizes=True,
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    train_ds = Dataset.from_dict({"text": list(train_texts), "label": list(train_labels)}).map(tokenize, batched=True)
    eval_ds = Dataset.from_dict({"text": list(eval_texts), "label": list(eval_labels)}).map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    final_model_dir = output_dir / "final"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    return {"output_dir": str(final_model_dir), "eval_metrics": eval_metrics}
