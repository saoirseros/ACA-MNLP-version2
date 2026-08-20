"""
Loads small, standard, public, no-login-required benchmark datasets for
each task, and returns a fixed-size random sample so evaluation stays
CPU-feasible in this project's constrained environment. Every dataset
used here is a real, published benchmark - see experiments/README.md for
sources/citations. Sampling uses a fixed seed for reproducibility.
"""
from datasets import load_dataset

SEED = 42

# Standard DailyDialog emotion label ordering (from the dataset's own
# documentation / original paper), mapped to our live emotion model's
# label set (j-hartmann/emotion-english-distilroberta-base): happiness
# -> joy, no_emotion -> neutral are the only renamings needed.
DAILYDIALOG_EMOTION_LABELS = ["no_emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
DAILYDIALOG_TO_MODEL_LABEL = {
    "no_emotion": "neutral",
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "joy",
    "sadness": "sadness",
    "surprise": "surprise",
}

# dair-ai/emotion's "love" label has no equivalent in our live emotion
# model's label set (anger/disgust/fear/joy/neutral/sadness/surprise), so
# it is excluded from both train and test samples for an honest, defined
# comparison rather than silently mismatching.
EMOTION_LABELS = ["sadness", "joy", "anger", "fear", "surprise"]


def load_sentiment_split(split: str, n: int):
    """
    SST-2 (Stanford Sentiment Treebank v2), part of GLUE.
    Source: https://huggingface.co/datasets/glue (config "sst2")
    GLUE's official "test" split is unlabeled, so "validation" is used as
    the held-out evaluation set here, following common practice.
    """
    ds = load_dataset("glue", "sst2", split=split)
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))
    label_map = {0: "negative", 1: "positive"}
    texts = list(ds["sentence"])
    labels = [label_map[label] for label in ds["label"]]
    return texts, labels


def load_emotion_split(split: str, n: int):
    """
    dair-ai/emotion: English Twitter messages labeled with one of 6 basic
    emotions. Source: https://huggingface.co/datasets/dair-ai/emotion
    "love" examples are excluded (see EMOTION_LABELS note above).
    """
    ds = load_dataset("dair-ai/emotion", split=split)
    label_names = ds.features["label"].names  # ['sadness','joy','love','anger','fear','surprise']
    ds = ds.filter(lambda example: label_names[example["label"]] != "love")
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))
    texts = list(ds["text"])
    labels = [label_names[label] for label in ds["label"]]
    return texts, labels


def load_toxicity_split(split: str, n: int):
    """
    tweet_eval (offensive-language subset): tweets labeled offensive or
    not. Source: https://huggingface.co/datasets/tweet_eval (config
    "offensive"). Used as a proxy for "toxicity" - see experiments/
    README.md for why this is not a perfect match with the live model's
    original Jigsaw training distribution.
    """
    ds = load_dataset("tweet_eval", "offensive", split=split)
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))
    label_map = {0: "non-toxic", 1: "toxic"}
    texts = list(ds["text"])
    labels = [label_map[label] for label in ds["label"]]
    return texts, labels


def load_dialogue_sample(n_dialogues: int):
    """
    DailyDialog: multi-turn, human-written dialogues with per-utterance
    emotion labels. Source (mirror used because the original dataset's
    loading script is broken on the Hub as of writing):
    https://huggingface.co/datasets/roskoN/dailydialog

    Returns a list of dialogues, each a list of (utterance, emotion_label)
    tuples in the live emotion model's label space.
    """
    ds = load_dataset("roskoN/dailydialog", split="test", trust_remote_code=True)
    ds = ds.shuffle(seed=SEED).select(range(min(n_dialogues, len(ds))))

    dialogues = []
    for row in ds:
        utterances = [u.strip() for u in row["utterances"]]
        emotions = [DAILYDIALOG_TO_MODEL_LABEL[DAILYDIALOG_EMOTION_LABELS[e]] for e in row["emotions"]]
        dialogues.append(list(zip(utterances, emotions)))
    return dialogues
