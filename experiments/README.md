# Experiments (Phase 9)

Reproducible evaluation framework that generates **real** accuracy/precision/
recall/F1/latency numbers - nothing here is hardcoded. Re-running
`run_all.py` re-downloads/re-samples the datasets (fixed seed for
reproducibility) and re-runs every model, so the numbers reflect whatever
code is currently in `nlp-service/app/`.

For the actual results and their interpretation (written for the
project's research paper), see **[FINDINGS.md](FINDINGS.md)**. This file
is the technical "how it works / how to reproduce it" documentation.

## What this answers

1. **Baseline vs Transformer**, per task (sentiment, emotion, toxicity):
   TF-IDF + Logistic Regression vs. the exact pretrained Transformer used
   by the live chat pipeline.
2. **Full-context vs Adaptive Context Activation (ACA)**: the project's
   central research question - does ACA maintain/improve predictive
   quality while reducing how much conversation history gets processed?

## Datasets used (all public, no login required)

| Task | Dataset | Source |
| --- | --- | --- |
| Sentiment | SST-2 (validation split used as test set) | [glue/sst2](https://huggingface.co/datasets/glue) |
| Emotion | dair-ai/emotion (6-class; "love" excluded, see below) | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) |
| Toxicity (proxy) | tweet_eval, "offensive" config | [tweet_eval](https://huggingface.co/datasets/tweet_eval) |
| Adaptive context | DailyDialog, per-utterance emotion labels | [roskoN/dailydialog](https://huggingface.co/datasets/roskoN/dailydialog) mirror (the original `daily_dialog` dataset's loading script is broken on the Hub as of writing) |

Datasets are downloaded via the Hugging Face `datasets` library the first
time a script runs, then cached locally (`~/.cache/huggingface/`).

### Why these particular substitutions

- **"Toxicity"**: the live model (`unitary/toxic-bert`) was trained on
  Jigsaw's Wikipedia-comment data. Jigsaw's Kaggle dataset requires a
  Kaggle login to download automatically, so `tweet_eval`'s
  offensive-language subset (Twitter) is used here as an openly
  accessible proxy. This means the toxicity numbers reflect
  cross-domain (Wikipedia-trained model evaluated on Twitter text)
  performance, not an in-domain Jigsaw evaluation - see FINDINGS.md.
- **Emotion label mismatch**: `dair-ai/emotion` has 6 labels (sadness,
  joy, love, anger, fear, surprise); the live model has 7 (anger,
  disgust, fear, joy, neutral, sadness, surprise). "love" has no
  equivalent in the live model's label space, so examples labeled "love"
  are excluded from both the training and test samples. "disgust" and
  "neutral" never appear as true labels in this particular dataset
  (their row is present in the label list for completeness, but always
  has 0 support) - this mechanically affects macro-averaged metrics
  (see FINDINGS.md).
- **DailyDialog emotion labels** are remapped to the live model's label
  space: `happiness -> joy`, `no_emotion -> neutral`, others unchanged.

## Structure

```
experiments/
    common/
        pathutil.py    # adds nlp-service/ to sys.path so scripts reuse the live model wrappers
        metrics.py      # accuracy/precision/recall/F1/confusion-matrix/latency + chart generation
    datasets_loader.py  # loads + samples each public dataset
    baselines.py         # TF-IDF + Logistic Regression: train + evaluate
    transformer_eval.py  # evaluates the live pretrained Transformers (imported from nlp-service)
    context_experiment.py  # Full-context vs Adaptive Context Activation, on DailyDialog
    run_all.py            # orchestrates everything, saves results/
    results/               # generated JSON + PNG output (see below)
```

## How to reproduce

Uses the same Python environment as `nlp-service/` (this repo's
[nlp-service/README.md](../nlp-service/README.md) has notes on the
Windows long-path venv workaround if you hit that):

```powershell
cd experiments
<path-to-nlp-service-venv>\Scripts\python.exe -m pip install -r ..\nlp-service\requirements.txt
<path-to-nlp-service-venv>\Scripts\python.exe run_all.py
```

This takes roughly 10-15 minutes on a CPU-only machine (mostly spent on
per-example Transformer inference and the ~250-turn context experiment,
all run one example at a time to match how the live chat pipeline
actually processes messages).

## Output

`results/` contains, per task (`sentiment`, `emotion`, `toxicity`):
- `<task>_comparison.json` - full metrics (accuracy, precision/recall/F1
  macro & weighted, full classification report, confusion matrix,
  latency distribution) for both baseline and Transformer.
- `<task>_baseline_confusion_matrix.png`, `<task>_transformer_confusion_matrix.png`
- `<task>_metric_comparison.png`, `<task>_latency_comparison.png`

Plus for the context experiment:
- `context_experiment_comparison.json`
- `context_full_confusion_matrix.png`, `context_adaptive_confusion_matrix.png`
- `context_metric_comparison.png`, `context_efficiency_comparison.png`

## Sample sizes

Kept modest so the whole suite finishes in minutes rather than hours on
a CPU-only development machine (`run_all.py`'s `SAMPLE_SIZES` dict):
300 test examples per classification task (2000-3000 for baseline
training), 40 DailyDialog dialogues (~250 evaluable turns) for the
context experiment. Increase these and re-run for higher-confidence
numbers if you have more compute time available - the code does not
change based on sample size.
