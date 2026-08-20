# Training / Fine-Tuning Pipeline (Phase 9)

Scaffolding for fine-tuning task-specific Transformers, per the project
spec's requested workflow:

```
Dataset -> Preprocessing -> Tokenizer -> Train/Validation split ->
Pretrained Transformer -> Fine-tuning -> Evaluation -> Saved model ->
(separately) Inference service
```

## Status: infrastructure built, not executed this session

These scripts are complete and syntax-checked, but **have not been run**
in this session. CPU-only fine-tuning of a Transformer (even for 2
epochs on a few thousand examples) typically takes far longer than is
practical for a single interactive working session. The project spec
explicitly allows this:

> "For experiments where fine-tuning is too computationally expensive
> initially, create the architecture so pretrained inference can be
> demonstrated first and fine-tuning can be performed separately."

Phase 4-8 already demonstrated pretrained inference, live, end-to-end.
This directory is the "fine-tuning can be performed separately" half.

## Structure

```
training/
    shared/
        finetune_utils.py   # shared Trainer-based fine-tuning function, reused by every task
    sentiment/train.py       # fine-tunes distilbert-base-uncased on SST-2
    emotion/train.py          # fine-tunes distilbert-base-uncased on dair-ai/emotion
    toxicity/train.py         # fine-tunes distilbert-base-uncased on tweet_eval (offensive)
```

Each `train.py` is independently runnable and modifies nothing else -
"Each model should be independently replaceable" per the project spec.

## Why start from `distilbert-base-uncased` rather than the live checkpoints

The live nlp-service already uses task-specific pretrained checkpoints
(see [nlp-service/README.md](../nlp-service/README.md)'s model table).
These scripts fine-tune a *generic* base checkpoint instead of further
tuning those, so that:

- Sentiment: this is a working template for the pipeline (the live
  checkpoint is already fine-tuned on SST-2 itself, so re-fine-tuning it
  further isn't the interesting case).
- Emotion and toxicity: these are genuinely useful **in-domain**
  fine-tunes, directly motivated by [experiments/FINDINGS.md](../experiments/FINDINGS.md)'s
  finding that the live emotion/toxicity models are evaluated
  **out-of-domain** relative to the datasets used in Phase 9's
  experiments. Fine-tuning from a clean base on those same datasets is
  the natural next step to test whether that gap closes.

## How to run (when you have time for a full fine-tuning pass)

Using the nlp-service virtual environment (already has
torch/transformers installed; `datasets`/`scikit-learn` too, from
Phase 9):

```powershell
cd training/emotion
<path-to-nlp-service-venv>\Scripts\python.exe train.py
```

Each script prints the saved model directory and final evaluation
metrics (accuracy, F1 macro/precision/recall) when done - computed for
real from the fine-tuned model, not fabricated, same as every other
metric in this project.

## Wiring a fine-tuned model into the live service

Fine-tuning and inference are intentionally decoupled (spec: "Do not
retrain models every time the application starts. The live application
should load saved models."). After running a script above:

1. Note the printed `output_dir` (under `nlp-service/trained_models/<task>/final/`).
2. Update the corresponding `MODEL_CHECKPOINT` constant in
   `nlp-service/app/models/<task>/model.py` to that local path instead
   of the Hugging Face Hub checkpoint name.
3. Restart the nlp-service - the lazy-loading `@lru_cache` wrapper
   already in place will load the new local checkpoint on first use, no
   other code changes required.
